import cv2
import numpy as np
import torch
from cv2.typing import MatLike
from torch import nn
from torchvision import models


def card_color(card: MatLike) -> int:
    # (h, s, v)
    boundaries = [
        ((20, 100, 100), (40, 255, 255)),  # yellow
        ((100, 100, 100), (120, 255, 255)),  # blue
        ((160, 50, 50), (180, 255, 255)),  # red
        ((50, 70, 70), (80, 255, 255)),  # green
    ]

    card = card.copy()
    height, width = card.shape[:2]

    top = int(0.15 * height)
    bottom = height - top
    left = int(0.15 * width)
    right = width - left

    card[top:bottom, left:right, :] = 0

    img_hsv = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)
    return np.argmax([cv2.countNonZero(cv2.inRange(img_hsv, lowb, highb)) for lowb, highb in boundaries])  # type: ignore


def get_class_model(
    num_class: int = 12, ws_file: str | None = None, pretrained: bool = False, device: str = "cuda"
) -> models.ResNet:
    ws = "DEFAULT" if pretrained or ws_file is None else None
    model = models.resnet18(weights=ws)
    model.fc = nn.Linear(model.fc.in_features, num_class)
    if ws_file:
        ws = torch.load(ws_file, map_location=device)
        model.load_state_dict(ws["model_state_dict"])
    return model


def train_loop(dataloader, model, loss_fn, optimizer, device) -> tuple[float, float]:
    model.train()

    size = len(dataloader.dataset)
    train_loss, train_acc = 0, 0

    for batch, (imgs, (_, force)) in enumerate(dataloader):
        imgs = imgs.to(device)
        force = force.to(device)

        pred = model(imgs)
        loss = loss_fn(pred, force)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += (pred.argmax(1) == force).float().sum().item()

        if batch % 5 == 0:
            loss, current = loss.item(), batch * len(imgs)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return train_loss / len(dataloader), train_acc / size


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
