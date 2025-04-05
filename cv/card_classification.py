import argparse
import os

import cv2
import numpy as np
import torch
from cv2.typing import MatLike
from datasets import CardDataset
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import models
from torchvision.transforms import InterpolationMode, v2


def card_color(img: MatLike) -> int:
    # (h, s, v)
    boundaries = [
        ((20, 20, 130), (90, 125, 210)),  # yellow
        ((100, 120, 110), (120, 200, 255)),  # blue
        ((120, 10, 80), (180, 120, 170)),  # red
        ((60, 60, 80), (95, 120, 210)),  # green
    ]

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return np.argmax([cv2.inRange(img_hsv, lowb, highb).sum() for lowb, highb in boundaries])  # type: ignore


def train_loop(dataloader, model, loss_fn, optimizer, device) -> tuple[float, float]:
    model.train()

    size = len(dataloader.dataset)
    train_loss, train_acc = 0, 0

    for batch, (imgs, (_, force)) in enumerate(dataloader):
        loss = 0
        imgs = imgs.to(device)
        force = force.to(device)

        pred = model(imgs)
        loss += loss_fn(pred, force)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += (pred.argmax(1) == force).float().sum().item()

        # if batch % 2 == 0:
        #     loss, current = loss.item(), batch * len(imgs)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return train_loss / len(dataloader), train_acc / len(dataloader.dataset)


def test_loop(dataloader, model, loss_fn, device) -> tuple[float, float]:
    model.eval()
    test_loss, test_acc = 0, 0

    with torch.no_grad():
        for imgs, (_, force) in dataloader:
            imgs = imgs.to(device)
            force = force.to(device)
            pred = model(imgs)
            test_loss += loss_fn(pred, force).item()
            test_acc += (pred.argmax(1) == force).float().sum().item()

    return test_loss / len(dataloader), test_acc / len(dataloader.dataset)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on the {device} device")

    parser = argparse.ArgumentParser(prog="card classification")
    parser.add_argument("--load", type=str, default=None, help="File to load the model from")
    parser.add_argument(
        "--train", type=str, default=None, help="File to save the trained model (without file extension)"
    )
    args = parser.parse_args()

    if args.train:
        dataset = CardDataset("./cv/card_images/")
        train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])

        train_dataset.dataset.transform = v2.Compose(
            [
                v2.ColorJitter(brightness=0.3, contrast=0.5, saturation=0.1, hue=0),
                v2.RandomVerticalFlip(),
                v2.RandomRotation((-10, 10), interpolation=InterpolationMode.BILINEAR, expand=True),
                v2.RandomCrop((70, 50)),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

        batch_size = 20
        lr = 1e-4
        epochs = 1000
        t = 0

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, 12)
        model = model.to(device)
        print("Number of parameters of the model:", sum(p.numel() for p in model.parameters() if p.requires_grad))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        train_losses, train_accs, test_losses, test_accs = [], [], [], []

        if args.load:
            checkpoint = torch.load(args.load)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optim_state_dict"])
            train_losses = checkpoint["train_loss"]
            train_accs = checkpoint["train_accs"]
            test_losses = checkpoint["test_loss"]
            test_accs = checkpoint["test_accs"]
            t = len(train_losses)
            epochs = t + 50

        for t in range(t, epochs):
            print(f"Epoch {t + 1}\n----------------------------")
            train_loss, train_acc = train_loop(train_dataloader, model, loss_fn, optimizer, device)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            print(f"train loss: {train_loss}, train acc: {train_acc}")
            test_loss, test_dice = test_loop(test_dataloader, model, loss_fn, device)
            test_losses.append(test_loss)
            test_accs.append(test_dice)
            print(f"test_loss: {test_loss}, test acc: {test_dice}")

        model.eval()
        # example_inputs = torch.randn(1, 3, 640, 480)
        # torch.onnx.export(model, example_inputs, save_file_name + ".onnx", verbose=True, dynamo=True, optimize=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "train_accs": train_accs,
                "train_loss": train_losses,
                "test_accs": test_accs,
                "test_loss": test_losses,
            },
            args.train + ".pth",
        )
        print(f"model saved in {args.train}.pth")
    else:
        assert args.load is not None, "at least --load or --train must be defined"

        cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
        img_dir = "./cv/card_images/"
        imgs = sorted(os.listdir(img_dir))
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, 12)

        checkpoint = torch.load(args.load)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.eval().to(device)
        train_losses = checkpoint["train_loss"]
        train_accs = checkpoint["train_accs"]
        test_losses = checkpoint["test_loss"]
        test_accs = checkpoint["test_accs"]
        t = len(train_losses)

        color_map = {
            0: "yellow",
            1: "blue",
            2: "red",
            3: "green",
        }

        for i in range(0, len(imgs), 4):
            img = cv2.imread(os.path.join(img_dir, imgs[i]))
            if torch.rand((1,)).item() > 0.5:
                img = cv2.flip(img, 0)
            color = card_color(img)

            trfm = v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                ]
            )
            blob = trfm(img).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(blob)
                pred = torch.argmax(pred, dim=1)

            cv2.putText(
                img,
                text=color_map[color] + "_" + str(pred.item() + 1),
                org=(img.shape[0] // 8, 20),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=0.8,
                color=(0, 0, 0),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
            cv2.imshow("preview", img)
            k = cv2.waitKey()
            if k == 27:
                break

        cv2.destroyAllWindows()
