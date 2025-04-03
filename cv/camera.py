import sys
import time

import cv2
import numpy as np
import torch
from card_detection import binarize_image, card_contours, separate_cards, card_positions, put_labels
from cv2.dnn import blobFromImage
from uNet import UNet


def camera_idxs():
    arr = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            arr.append(i)
            cap.release()
    return arr


if __name__ == "__main__":
    cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("cropped", cv2.WINDOW_NORMAL)

    model, device = None, None
    if len(sys.argv) > 1:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Running on the {device}")

        model = UNet(1, 5, encoder_depth=3, firstl_channels=16).to(device)
        data = torch.load(sys.argv[1])
        model.load_state_dict(data["model_state_dict"])
        model = model.eval()
        cv2.namedWindow("model_preview", cv2.WINDOW_NORMAL)

    cam_idxs = camera_idxs()
    # color, force, img_count = 0, 1, 0

    cam = cv2.VideoCapture(cam_idxs[-1])
    rval = True
    while rval:
        rval, frame = cam.read()
        # cropped = frame.copy()

        if rval:
            if model:
                blob = blobFromImage(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
                img = torch.Tensor(blob).to(device)

                with torch.no_grad():
                    pred = torch.argmax(model(img), dim=1).permute((1, 2, 0)).cpu().numpy()

                pred = cv2.applyColorMap((pred * 40).astype(np.uint8), cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(frame, 0.2, pred, 0.8, 0)
                cv2.imshow("model_preview", overlay)

            # gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            # _, bin = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
            # cnts, _ = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # cnts = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)
            # if len(cnts) > 0:
            #     x, y, w, h = cv2.boundingRect(cnts[0])
            #     cropped = frame[y:y + h, x: x + w]
            # cv2.imshow("cropped", cropped)
            bw = binarize_image(frame)
            cnts = card_contours(bw)
            cnts = separate_cards(frame, cnts)
            frame_cnts = frame.copy()
            cv2.drawContours(frame_cnts, cnts, -1, (0, 255, 0), 2)
            heaven, q, hell, hand = card_positions(cnts)
            if heaven is not None:
                frame_cnts = put_labels([heaven], "heaven", frame_cnts)
            frame_cnts = put_labels(q, "q", frame_cnts)
            if hell is not None:
                frame_cnts = put_labels([hell], "hell", frame_cnts)
            frame_cnts = put_labels(hand, "h", frame_cnts)
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
