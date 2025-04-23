import cv2
import numpy as np
import torch
from cv2.typing import MatLike
from torch import nn
from torchvision import models


def card_color(card: MatLike) -> int:
    # # (h, s, v)
    boundaries = [
        ((20, 80, 80), (40, 255, 255)),  # yellow
        ((100, 80, 80), (120, 255, 255)),  # blue
        ((160, 80, 80), (180, 255, 255)),  # red
        ((50, 80, 80), (80, 255, 255)),  # green
    ]

    height, width = card.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    top = int(0.15 * height)
    bottom = height - top
    left = int(0.20 * width)
    right = width - left

    mask[:top, :] = 255
    mask[bottom:, :] = 255
    mask[:, :left] = 255
    mask[:, right:] = 255

    masked_card = cv2.bitwise_and(card, card, mask=mask)

    img_hsv = cv2.cvtColor(masked_card, cv2.COLOR_BGR2HSV)
    return np.argmax([cv2.countNonZero(cv2.inRange(img_hsv, lowb, highb)) for lowb, highb in boundaries])  # type: ignore


def get_class_model(num_class: int = 12, ws_file: str | None = None) -> models.ResNet:
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, num_class)
    if ws_file:
        ws = torch.load(ws_file)
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
