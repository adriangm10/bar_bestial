import sys
import time
from typing import Sequence

import cv2
import numpy as np
import torch
from card_classification import card_color
from card_detection import (binarize_image, card_contours, card_positions,
                            separate_cards)
from cv2.dnn import blobFromImage
from cv2.typing import MatLike
from torchvision import models
from torchvision.transforms import v2
from uNet import UNet


def camera_idxs():
    arr = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            arr.append(i)
            cap.release()
    return arr


def is_horizontal_rect(rect: tuple[float, float, float, float], aspect_ratio_thresh: float = 1.5):
    _, _, w, h = rect
    aspect_ratio = w / h if h != 0 else 0
    return aspect_ratio > aspect_ratio_thresh


def put_labels(img: MatLike, cnts: Sequence[MatLike], pos_name: str, model, dest: MatLike, device: str) -> MatLike:
    trfm = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    color_map = {
        0: "yellow",
        1: "blue",
        2: "red",
        3: "green",
    }

    for i, c in enumerate(cnts):
        x, y, w, h = cv2.boundingRect(c)
        # rect = cv2.minAreaRect(c)
        # ignore horizontal cards (heaven and hell)

        if is_horizontal_rect((x, y, w, h)):
            dest = cv2.putText(
                dest,
                text=pos_name,
                org=(x, y),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=0.6,
                color=(255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
            continue

        card = img[y : y + h, x : x + w, :]
        ccolor = card_color(card)
        blob = trfm(card).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = torch.argmax(model(blob), dim=1).item()
        dest = cv2.putText(
            dest,
            text=pos_name + str(i + 1) + ": " + color_map[ccolor] + "_" + str(pred + 1),
            org=(x, y),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=0.6,
            color=(255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    return dest


def main():
    seg_model, device = None, None
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Running on the {device}")

    if len(sys.argv) > 1:
        seg_model = UNet(1, 5, encoder_depth=3, firstl_channels=16).to(device)
        data = torch.load(sys.argv[1])
        seg_model.load_state_dict(data["model_state_dict"])
        seg_model = seg_model.eval()
        cv2.namedWindow("model_preview", cv2.WINDOW_NORMAL)

    cam_idxs = camera_idxs()
    # color, force, img_count = 0, 1, 0
    class_model = models.resnet18()
    class_model.fc = torch.nn.Linear(class_model.fc.in_features, 12)
    # class_model = CNN(1, 12)
    class_model_dict = torch.load("./training_models/resnet.pth")
    class_model.load_state_dict(class_model_dict["model_state_dict"])
    class_model = class_model.eval().to(device)

    cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("cropped", cv2.WINDOW_NORMAL)

    cam = cv2.VideoCapture(cam_idxs[-1])
    rval = True
    while rval:
        rval, frame = cam.read()

        if not rval:
            break
        # cropped = frame.copy()

        if rval:
            if seg_model:
                blob = blobFromImage(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                img = torch.Tensor(blob).to(device)

                with torch.no_grad():
                    pred = torch.argmax(seg_model(img), dim=1).permute((1, 2, 0)).cpu().numpy()

                pred = cv2.applyColorMap((pred * 40).astype(np.uint8), cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(frame, 0.2, pred, 0.8, 0)
                cv2.imshow("model_preview", overlay)

            # gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            # _, bin = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
            # cnts, _ = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # cnts = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)
            # if len(cnts) > 0:
            #     x, y, w, h = cv2.boundingRect(cnts[0])
            #     cropped = frame[y : y + h, x : x + w]
            # cv2.imshow("cropped", cropped)
            bw = binarize_image(frame)
            cnts = card_contours(bw)
            cnts = separate_cards(frame, cnts)
            frame_cnts = frame.copy()
            cv2.drawContours(frame_cnts, cnts, -1, (0, 255, 0), 2)
            heaven, q, hell, hand = card_positions(cnts)
            if heaven is not None:
                frame_cnts = put_labels(frame, [heaven], "heaven", class_model, frame_cnts, device)
            frame_cnts = put_labels(frame, q, "q", class_model, frame_cnts, device)
            if hell is not None:
                frame_cnts = put_labels(frame, [hell], "hell", class_model, frame_cnts, device)
            frame_cnts = put_labels(frame, hand, "h", class_model, frame_cnts, device)

            cv2.imshow("preview", frame_cnts)

        key = cv2.pollKey()
        if key == 27:
            break
        elif key == 13:
            name = f"./opencv_images/opencv_img_{time.time_ns()}.png"
            # name = f"./opencv_images/card_{color}_{force}_{img_count}.png"
            if cv2.imwrite(name, frame):
                print(name, "written")
                # img_count += 1
            else:
                print(f"ERROR: Couldn't save  {name} file", file=sys.stderr)
            # elif key == ord('c'):
            #     color = int(input("new color (0-3): "))
            #     img_count = 0
            # elif key == ord('f'):
            #     force = int(input("new force (1-12): "))
            #     img_count = 0
        elif key != -1:
            print(key)

    cv2.destroyAllWindows()
    cam.release()


if __name__ == "__main__":
    main()
