import os
from typing import Sequence

import cv2
import numpy as np
from cv2.typing import MatLike

MIN_CARD_AREA = 5000
MAX_CARD_AREA = int(2.5 * MIN_CARD_AREA)


def binarize_image(img: MatLike) -> MatLike:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    h, w = gray.shape[:2]
    boty, topy, botx, topx = h // 100, h - h // 100, w // 100, w - w // 100
    thresh = max(gray[boty][w // 2], gray[topy][w // 2], gray[boty][botx], gray[topy][botx], gray[boty][topx], gray[topy][topx]) + 10  # type: ignore
    _, th = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY)
    return th


def card_contours(thimg: MatLike) -> Sequence[MatLike]:
    contours, hier = cv2.findContours(thimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnts = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        if area > MIN_CARD_AREA and hier[0][i][3] == -1:
            cnts.append(contour)

    return cnts


def separate_cards(img: MatLike, contours: Sequence[MatLike]):
    total_cnts = []
    for cnt in contours:
        mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
        cv2.drawContours(mask, [cnt], 0, (255, 255, 255), cv2.FILLED)
        cards = cv2.bitwise_and(img, img, mask=mask)

        gray_cards = cv2.cvtColor(cards, cv2.COLOR_RGB2GRAY)
        thresh = gray_cards.max() - 50
        _, th = cv2.threshold(gray_cards, thresh, 255, cv2.THRESH_BINARY_INV)
        cnts, hier = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        filtered_cnts = []
        for i, cnt in enumerate(cnts):
            area = cv2.contourArea(cnt)
            epsilon = 0.1 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            if MAX_CARD_AREA >= area >= MIN_CARD_AREA and hier[0][i][3] == 0 and len(approx) == 4:
                filtered_cnts.append(cnt)

        total_cnts.extend(filtered_cnts)

    return total_cnts


def card_positions(
    contours: Sequence[MatLike],
) -> tuple[MatLike | None, Sequence[MatLike], MatLike | None, Sequence[MatLike]]:
    """return: heaven, queue, hell, hand"""
    sorted_cnts = sorted([(i, *cv2.boundingRect(cnt)) for i, cnt in enumerate(contours)], key=lambda b: b[2])

    top_group = []
    hand_group = []
    prev_y = None
    append_top = True
    for i, _, y, _, _ in sorted_cnts:
        if append_top and (prev_y is None or abs(prev_y - y) < 50):
            top_group.append(contours[i])
        else:
            append_top = False
            hand_group.append(contours[i])

        prev_y = y

    top_group.sort(key=lambda cnt: cv2.boundingRect(cnt)[0])
    hand_group.sort(key=lambda cnt: cv2.boundingRect(cnt)[0])
    if len(top_group) < 2:
        print("ERROR: heaven and hell not detected")
        return None, [], None, hand_group
    return top_group[0], top_group[1:-1], top_group[-1], hand_group


def put_labels(cnts: Sequence[MatLike], label: str, img: MatLike) -> MatLike:
    for i, c in enumerate(cnts):
        x, y, w, _ = cv2.boundingRect(c)
        img = cv2.putText(
            img,
            text=label + str(i + 1),
            org=(x + w // 4, y),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1,
            color=(255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    return img


if __name__ == "__main__":
    cv2.namedWindow("preview", cv2.WINDOW_NORMAL)

    img_dir = "./cv/game_images/"
    imgs = os.listdir(img_dir)
    for f in imgs:
        img = cv2.imread(os.path.join(img_dir, f))
        th = binarize_image(img)

        cnts = card_contours(th)
        cnts = separate_cards(img, cnts)
        img_cnt = img.copy()
        cv2.drawContours(img_cnt, cnts, -1, (0, 255, 0), 1)

        heaven, q, hell, hand = card_positions(cnts)
        if heaven is not None:
            img_cnt = put_labels([heaven], "heaven", img_cnt)
        img_cnt = put_labels(q, "q", img_cnt)
        if hell is not None:
            img_cnt = put_labels([hell], "hell", img_cnt)
        img_cnt = put_labels(hand, "h", img_cnt)

        cv2.imshow("preview", img_cnt)
        cv2.waitKey()

    cv2.destroyAllWindows()
