import sys

import matplotlib.pyplot as plt
import torch
import torch.onnx
from datasets import BarDataset
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding="same"),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DownSample, self).__init__()

        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        c = self.conv(x)
        p = self.pool(c)
        return c, p


class UpSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(UpSample, self).__init__()

        self.upConv = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1.size = [batch_size, channels, H, W]
        # x2.size = [batch_size, channels // 2, H, W]

        x1 = self.upConv(x1)
        x = torch.cat((x1, x2), 1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, encoder_depth: int = 4, firstl_channels: int = 64):
        super(UNet, self).__init__()

        self.encoder_depth = encoder_depth

        self.down = nn.ModuleList([DownSample(in_channels, firstl_channels)])
        channels = firstl_channels
        for _ in range(1, encoder_depth):
            self.down.append(DownSample(channels, channels * 2))
            channels *= 2

        self.bridge = DoubleConv(channels, channels * 2)
        channels *= 2

        self.up = nn.ModuleList([])
        for _ in range(encoder_depth):
            self.up.append(UpSample(channels, channels // 2))
            channels //= 2

        self.outl = nn.Conv2d(channels, num_classes, 1)

    def forward(self, x):
        fm, y = self.down[0](x)
        fms = [fm]
        for d in self.down[1:]:
            fm, y = d(y)
            fms.insert(0, fm)

        y = self.bridge(y)

        for up, fm in zip(self.up, fms):
            y = up(y, fm)

        assert y.size(dim=2) == x.size(dim=2) and y.size(dim=3) == x.size(dim=3)
        return self.outl(y)


def dice_coefficient(pred, target, epsilon=1e-6):
    # pred.size = [batch_size, num_labels, H, W]
    # target.size = [batch_size, 1, H, W]

    dice = 0
    pred = torch.argmax(torch.softmax(pred, dim=1), dim=1)
    num_labels = pred.max() + 1

    for i in range(num_labels):
        pred_mask = pred == i
        target_mask = target == i

        intersection = torch.sum(pred_mask * target_mask)
        union = torch.sum(pred_mask) + torch.sum(target_mask)

        dice += (2 * intersection + epsilon) / (union + epsilon)

    return dice / num_labels


def train_loop(dataloader, model, loss_fn, optimizer, device):
    model.train()

    size = len(dataloader.dataset)
    train_loss, train_dice = 0, 0

    for batch, (imgs, masks) in enumerate(dataloader):
        imgs = imgs.to(device)
        masks = masks.to(device).squeeze(1)

        pred = model(imgs)
        loss = loss_fn(pred, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_dice += dice_coefficient(pred, masks).item()

        if batch % 7 == 0:
            loss, current = loss.item(), batch * len(imgs)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return train_loss / len(dataloader), train_dice / len(dataloader)


def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    test_loss, test_dice = 0, 0

    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs = imgs.to(device)
            masks = masks.to(device).squeeze(1)
            pred = model(imgs)
            test_loss += loss_fn(pred, masks).item()
            test_dice += dice_coefficient(pred, masks).item()

    return test_loss / len(dataloader), test_dice / len(dataloader)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on the {device} device")

    save_file_name = sys.argv[2] if len(sys.argv) > 2 else "unet"

    num_labels = 12
    label_map = {
        0: "_background_",
        1: "q1",
        2: "q2",
        3: "q3",
        4: "q4",
        5: "q5",
        6: "h1",
        7: "h2",
        8: "h3",
        9: "h4",
        10: "hell",
        11: "heaven",
    }

    dataset = BarDataset("./cv/game_images_voc/")
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])

    train_dataset.dataset.transform = v2.Compose(
        [
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(),
            v2.RandomRotation((-7, 7)),
            v2.RandomGrayscale(),
        ]
    )

    batch_size = 4
    lr = 1e-4
    epochs = 200
    t = 0

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = UNet(3, num_labels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=255)

    train_losses, train_dices, test_losses, test_dices = [], [], [], []

    if len(sys.argv) > 1:
        checkpoint = torch.load(sys.argv[1])
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optim_state_dict"])
        train_losses = checkpoint["train_loss"]
        train_dices = checkpoint["train_dices"]
        test_losses = checkpoint["test_loss"]
        test_dices = checkpoint["test_dices"]
        t = len(train_losses)
        epochs = t + 100

    for t in range(t, epochs):
        print(f"Epoch {t + 1}\n----------------------------")
        train_loss, train_dice = train_loop(train_dataloader, model, loss_fn, optimizer, device)
        train_losses.append(train_loss)
        train_dices.append(train_dice)
        print(f"train loss: {train_loss}, train DICE: {train_dice}")
        test_loss, test_dice = test_loop(test_dataloader, model, loss_fn, device)
        test_losses.append(test_loss)
        test_dices.append(test_dice)
        print(f"test_loss: {test_loss}, test DICE: {test_dice}")

    model.eval()
    example_inputs = torch.randn(1, 3, 640, 480)
    torch.onnx.export(model, example_inputs, save_file_name + ".onnx", verbose=True, dynamo=True, optimize=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "train_dices": train_dices,
            "train_loss": train_losses,
            "test_dices": test_dices,
            "test_loss": test_losses,
        },
        save_file_name + ".pth",
    )

    epoch_list = list(range(1, epochs + 1))

    plt.subplot(1, 2, 1)
    plt.plot(epoch_list, train_losses, label="Training loss")
    plt.plot(epoch_list, test_losses, label="Test loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epoch_list, train_dices, label="Training dice")
    plt.plot(epoch_list, test_dices, label="Test dice")
    plt.xlabel("Epochs")
    plt.ylabel("DICE")
    plt.legend()

    plt.tight_layout()
    plt.show()
