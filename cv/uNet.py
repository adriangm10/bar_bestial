import torch
from torch import nn


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

    def forward(self, x1, x2: torch.Tensor):
        # x1.size = [batch_size, channels, H, W]
        # x2.size = [batch_size, channels // 2, H, W]

        x1 = self.upConv(x1)
        if x1.shape[2:] != x2.shape[2:]:
            x1 = nn.functional.interpolate(x1, size=x2.shape[2:], mode="bilinear")
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
    pred = torch.argmax(pred, dim=1)
    num_labels = pred.max() + 1

    for i in range(num_labels):
        pred_mask = pred == i
        target_mask = target == i

        intersection = torch.sum(pred_mask * target_mask)
        union = torch.sum(pred_mask) + torch.sum(target_mask)

        dice += (2 * intersection + epsilon) / (union + epsilon)

    return dice / num_labels


def train_loop(dataloader, model, loss_fn, optimizer, device, trfm_img=None):
    model.train()

    size = len(dataloader.dataset)
    train_loss, train_dice = 0, 0

    for batch, (imgs, masks) in enumerate(dataloader):
        if trfm_img:
            imgs = trfm_img(imgs)
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


def test_loop(dataloader, model, loss_fn, device, trfm_img=None):
    model.eval()
    test_loss, test_dice = 0, 0

    with torch.no_grad():
        for imgs, masks in dataloader:
            if trfm_img:
                imgs = trfm_img(imgs)
            imgs = imgs.to(device)
            masks = masks.to(device).squeeze(1)
            pred = model(imgs)
            test_loss += loss_fn(pred, masks).item()
            test_dice += dice_coefficient(pred, masks).item()

    return test_loss / len(dataloader), test_dice / len(dataloader)
