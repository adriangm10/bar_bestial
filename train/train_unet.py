import argparse
import sys

sys.path.insert(0, "../bar_bestial/")

import matplotlib.pyplot as plt
import torch
import torch.onnx
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2

from cv.datasets import BarDataset
from cv.uNet import *

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on the {device} device")

    parser = argparse.ArgumentParser(prog="uNet")
    parser.add_argument("--load", type=str, default=None, help="File to load the model from")
    parser.add_argument(
        "--save-file", type=str, default="unet", help="File to save the trained model without extension"
    )
    parser.add_argument(
        "--combine-qs-hs",
        action="store_true",
        help="whether to combine the hand cards in one mask and the queue cards in other mask",
    )
    args = parser.parse_args()
    save_file_name = args.save_file

    if args.combine_qs_hs:
        num_labels = 5
        label_map = {
            0: "_background_",
            1: "q1",
            2: "h1",
            3: "hell",
            4: "heaven",
        }
    else:
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

    dataset = BarDataset("./cv/game_images_voc/", combine_qs_and_hs=args.combine_qs_hs)
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])

    train_dataset.dataset.transform = v2.Compose(
        [
            v2.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation((-5, 5)),
            v2.Grayscale(),
        ]
    )

    batch_size = 8
    lr = 1e-4
    epochs = 200
    t = 0

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = UNet(1, num_labels, encoder_depth=3, firstl_channels=32).to(device)
    print("Number of parameters of the model:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=255)

    train_losses, train_dices, test_losses, test_dices = [], [], [], []

    if args.load:
        checkpoint = torch.load(args.load)
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
        train_loss, train_dice = train_loop(
            train_dataloader, model, loss_fn, optimizer, device, trfm_img=v2.ToDtype(torch.float32, scale=True)
        )
        train_losses.append(train_loss)
        train_dices.append(train_dice)
        print(f"train loss: {train_loss}, train DICE: {train_dice}")
        test_loss, test_dice = test_loop(
            test_dataloader, model, loss_fn, device, trfm_img=v2.ToDtype(torch.float32, scale=True)
        )
        test_losses.append(test_loss)
        test_dices.append(test_dice)
        print(f"test_loss: {test_loss}, test DICE: {test_dice}")

    model.eval()
    # example_inputs = torch.randn(1, 3, 640, 480)
    # torch.onnx.export(model, example_inputs, save_file_name + ".onnx", verbose=True, dynamo=True, optimize=True)
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
