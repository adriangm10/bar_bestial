import argparse
import os
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import InterpolationMode, v2

sys.path.insert(0, "../bar_bestial/")
from cv.card_classification import *
from cv.datasets import CardDataset

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

        trfm = v2.Compose(
            [
                v2.ColorJitter(brightness=0.5, contrast=0.6, saturation=0.2, hue=0),
                v2.RandomVerticalFlip(),
                v2.RandomRotation((-10, 10), interpolation=InterpolationMode.BILINEAR, expand=True),
                v2.RandomCrop((70, 50)),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

        train_dataset.dataset.transform = trfm

        batch_size = 20
        lr = 1e-4
        epochs = 1000
        t = 0

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = get_class_model().to(device)

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
            epochs = t + 1000

            for t in range(t, epochs):
                print(f"Epoch {t + 1}\n----------------------------")
                train_loss, train_acc = train_loop(train_dataloader, model, loss_fn, optimizer, device)
                train_losses.append(train_loss)
                train_accs.append(train_acc)
                print(f"train loss: {train_loss}, train force acc: {train_acc}")
                test_loss, test_acc = test_loop(test_dataloader, model, loss_fn, device)
                test_losses.append(test_loss)
                test_accs.append(test_acc)
                print(f"test_loss: {test_loss}, test force acc: {test_acc}\n")

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
        import cv2

        cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
        img_dir = "./cv/card_images/"
        imgs = sorted(os.listdir(img_dir))

        model = get_class_model()
        trfm = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

        checkpoint = torch.load(args.load)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.eval().to(device)

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

            blob = trfm(img).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(blob)
                pred = torch.argmax(pred, dim=1).item()

            cv2.putText(
                img,
                text=color_map[color] + "_" + str(pred + 1),
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
