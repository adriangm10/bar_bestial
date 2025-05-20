import logging
from typing import Sequence

import cv2
import numpy as np
from cv2.typing import MatLike

MIN_CARD_AREA = 5000
MAX_CARD_AREA = int(2.5 * MIN_CARD_AREA)

logger = logging.getLogger(__name__)


def binarize_image(img: MatLike) -> MatLike:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    h, w = gray.shape[:2]
    boty, topy, botx, topx = h // 100, h - h // 100, w // 100, w - w // 100
    thresh = np.clip(max(gray[boty][w // 2], gray[topy][w // 2], gray[boty][botx], gray[topy][botx], gray[boty][topx], gray[topy][topx]) + 10, 0, 255)  # type: ignore
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

        gray_cards = cv2.cvtColor(cards, cv2.COLOR_BGR2GRAY)
        thresh = np.clip(gray_cards.max() - 50, 0, 255)
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
    # sorted card bboxes by y coordinate
    sorted_bboxs = sorted([(i, *cv2.boundingRect(cnt)) for i, cnt in enumerate(contours)], key=lambda b: b[2])

    current_group = []
    card_groups = []  # queue, hell/heaven, hand from top to bottom of the img
    prev_y = None
    for i, _, y, _, _ in sorted_bboxs:
        if prev_y is None or abs(prev_y - y) < 50:
            current_group.append(contours[i])
        else:
            card_groups.append(current_group)
            current_group = [contours[i]]

        prev_y = y

    if current_group:
        card_groups.append(current_group)

    # order cards by x coordinate
    for group in card_groups:
        group.sort(key=lambda cnt: cv2.boundingRect(cnt)[0])

    # no card is detected
    if not card_groups:
        return None, [], None, []

    # only queue and hand or queue and heaven and hell
    if len(card_groups) <= 2:
        if len(card_groups) == 2 and len(card_groups[1]) == 2:
            logger.debug("Only two groups detected and only 2 cards in the second one, checking if it's the hand or the hell and heaven")
            xheaven, _, _, _ = cv2.boundingRect(card_groups[0][0])
            xhell, _, _, _ = cv2.boundingRect(card_groups[0][-1])

            if abs(xheaven - cv2.boundingRect(card_groups[1][0])[0]) < 25 and abs(xhell - cv2.boundingRect(card_groups[1][-1])[0]) < 25:
                logger.debug("The second group corresponds to the heaven and hell")
                return card_groups[1][0], card_groups[0], card_groups[1][1], []
            logger.debug("The second group corresponds to the hand")

        return None, card_groups[0], None, card_groups[1] if len(card_groups) == 2 else []

    # heaven, queue, hell and hand detected
    queue = card_groups[0]
    heaven, hell = None, None
    if len(card_groups[1]) == 2:
        heaven, hell = card_groups[1]
    else:
        x, _, _, _ = cv2.boundingRect(card_groups[1][0])
        if abs(x - cv2.boundingRect(queue[0])[0]) < abs(x - cv2.boundingRect(queue[-1])[0]):
            heaven = card_groups[1][0]
        else:
            hell = card_groups[1][0]

    return heaven, queue, hell, card_groups[2]


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
