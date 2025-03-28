import matplotlib.pyplot as plt
import torch
from datasets import BarDataset
from torchvision.transforms import v2
from uNet import UNet

if __name__ == "__main__":
    checkpoint = torch.load("./models/unet.pth")
    model = UNet(3, 12).eval()
    model.load_state_dict(checkpoint["model_state_dict"])

    train_loss = checkpoint["train_loss"]
    train_dice = checkpoint["train_dices"]

    test_loss = checkpoint["test_loss"]
    test_dice = checkpoint["test_dices"]

    epochs = list(range(1, 201))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Training loss")
    plt.plot(epochs, test_loss, label="Test loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_dice, label="Training DICE")
    plt.plot(epochs, test_dice, label="Test DICE")
    plt.xlabel("Epochs")
    plt.ylabel("DICE")
    plt.legend()
    plt.tight_layout()

    dataset = BarDataset("./cv/game_images_voc/", transform=v2.RandomHorizontalFlip())

    figure = plt.figure()
    for i in range(1, 11, 2):
        idx = torch.randint(len(dataset), size=(1,)).item()
        img, mask = dataset[idx]
        pred_mask = torch.argmax(torch.softmax(model(img.unsqueeze(0)), dim=1), dim=1)

        figure.add_subplot(5, 2, i)
        plt.imshow(img.byte().permute((1, 2, 0)))
        plt.imshow(mask.permute((1, 2, 0)), alpha=0.7)
        plt.axis("off")

        figure.add_subplot(5, 2, i + 1)
        plt.imshow(img.byte().permute((1, 2, 0)))
        plt.imshow(pred_mask.permute((1, 2, 0)), alpha=0.7)
        plt.axis("off")

    figure.tight_layout()
    plt.show()
