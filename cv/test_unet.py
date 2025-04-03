import os
import sys

import matplotlib.pyplot as plt
import torch
from datasets import BarDataset
from torchvision.io import decode_image
from torchvision.transforms import v2
from uNet import UNet

if __name__ == "__main__":

    checkpoint = torch.load("./models/unet_mini_gray.pth")
    model = UNet(1, 5, encoder_depth=3, firstl_channels=32)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    train_loss = checkpoint["train_loss"]
    train_dice = checkpoint["train_dices"]

    test_loss = checkpoint["test_loss"]
    test_dice = checkpoint["test_dices"]

    epochs = list(range(1, len(train_loss) + 1))

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

    dataset = BarDataset("./cv/game_images_voc/", transform=v2.RandomHorizontalFlip(), combine_qs_and_hs=True)

    figure = plt.figure()
    for i in range(1, 11, 2):
        idx = torch.randint(len(dataset), size=(1,)).item()
        img, mask = dataset[idx]
        pred_mask = torch.argmax(model(v2.functional.rgb_to_grayscale(img).unsqueeze(0)), dim=1)

        figure.add_subplot(5, 2, i)
        plt.imshow(img.byte().permute((1, 2, 0)))
        plt.imshow(mask.permute((1, 2, 0)), alpha=0.7)
        plt.axis("off")

        figure.add_subplot(5, 2, i + 1)
        plt.imshow(img.byte().permute((1, 2, 0)))
        plt.imshow(pred_mask.permute((1, 2, 0)), alpha=0.7)
        plt.axis("off")

    figure.tight_layout()

    if len(sys.argv) > 1:
        label_map = {
            0: "_background_",
            1: "q",
            2: "h",
            3: "hell",
            4: "heaven",
        }

        figure = plt.figure()
        files = sorted(os.listdir(sys.argv[1]), reverse=True)
        for i, f in enumerate(files[:6]):
            img = decode_image(os.path.join(sys.argv[1], f)).float()
            mask = torch.argmax(model(v2.functional.rgb_to_grayscale(img.unsqueeze(0))), dim=1)

            figure.add_subplot(2, 3, i + 1)
            plt.imshow(img.byte().permute((1, 2, 0)))
            plt.imshow(mask.permute((1, 2, 0)), alpha=0.7)
            plt.axis("off")

        img = decode_image(os.path.join(sys.argv[1], files[-1])).float()
        mask = torch.argmax(model(v2.functional.rgb_to_grayscale(img.unsqueeze(0))), dim=1)

        figure = plt.figure()
        figure.add_subplot(3, 4, 1)
        plt.imshow(img.byte().permute((1, 2, 0)))
        plt.imshow(mask.permute((1, 2, 0)), alpha=0.7)
        for i in range(1, 5):
            figure.add_subplot(3, 4, i + 1)
            plt.imshow((mask == i).permute((1, 2, 0)))
            plt.title(label_map[i])

    plt.show()


