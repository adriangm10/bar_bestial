import argparse
import logging
from random import choice
from typing import Literal

import numpy as np
from sb3_contrib import QRDQN, TRPO
from stable_baselines3 import DQN, PPO

from game.bar import Card, CardType, Color
from game.bar_gym import BarEnv

type RLModel = QRDQN | DQN | TRPO | PPO
type CardRep = tuple[int, int]  # color, force

logger = logging.getLogger(__name__)


def load_model(file: str, model_class: str):
    match model_class:
        case "DQN":
            model = DQN.load(file)
        case "PPO":
            model = PPO.load(file)
        case "QRDQN":
            model = QRDQN.load(file)
        case "TRPO":
            model = TRPO.load(file)
        case _:
            raise ValueError(f"Not supported agent class: {model_class}")
    return model


def model_v_model(model1, model2, num_games: int, env: BarEnv) -> tuple[int, int, int]:
    wins1, wins2, draws = 0, 0, 0

    for _ in range(num_games):
        obs, _ = env.reset()
        while True:
            poss_actions = env.game.possible_actions()
            if env.game.turn == env.agent_color.value:
                action, _ = model1.predict(obs)
            else:
                action, _ = model2.predict(obs)

            if poss_actions and action not in poss_actions:
                action = choice(poss_actions)

            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                winners = env.game.winners()
                if len(winners) == env.num_players or not winners:
                    draws += 1
                elif env.agent_color in winners:
                    wins1 += 1
                else:
                    wins2 += 1
                break

    return wins1, wins2, draws


def create_state_rep(
    heaven: list[CardRep],
    hell: list[CardRep],
    q: list[CardRep],
    hand: list[CardRep],
    color_pos_map: list[int],
    chosen_card: CardRep | None = None,
    num_players: Literal[2, 3, 4] = 2,
    game_mode: Literal["full", "medium", "basic"] = "full",
    t: int = 1,
    history: list[np.ndarray] | None = None,
) -> np.ndarray:
    assert t >= 1
    assert len(q) <= 4
    L = 1 if game_mode == "basic" else 2  # hand + selected card
    M = 6  # heaven, hell and queue
    MM = M * num_players
    state_rep = np.zeros((t * MM + L, len(CardType.toList())), dtype=np.int32)

    # hand row
    for _, f in hand:
        state_rep[0][f - 1] = 1

    # selected card
    if game_mode != "basic" and chosen_card:
        state_rep[1][chosen_card[1] - 1] = 1

    # num_players rows for heaven
    for c, f in heaven:
        state_rep[color_pos_map.index(c) * M + L][f - 1] = 1

    # num_players rows for hell
    for c, f in hell:
        state_rep[color_pos_map.index(c) * M + L + 1][f - 1] = 1

    # 4 * num_players rows for the board, there will never be 5 cards in the queue
    for i, (c, f) in enumerate(q):
        state_rep[color_pos_map.index(c) * M + L + 2 + i][f - 1] = 1

    if history is not None:
        for i in range(1, min(t, len(history))):
            state_rep[L + i * MM : L + (i + 1) * MM] = history[-i]

        if chosen_card is None:
            history.append(state_rep[L : L + MM])

    return state_rep


def cv_loop(model: RLModel, num_players: Literal[2, 3, 4], game_mode: Literal["full", "medium", "basic"]) -> None:
    import cv2
    import torch

    from cv.camera import camera_idxs, put_labels
    from cv.card_classification import get_class_model
    from cv.card_detection import binarize_image, card_contours, card_positions, separate_cards

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    heaven, hell, q, hand, color_pos_map = [], [], [], [], []
    chosen_card = None
    num_cards = 12 * num_players if game_mode != "basic" else 9 * num_players

    cam_idxs = camera_idxs()
    cam = cv2.VideoCapture(cam_idxs[-1])
    class_model = get_class_model(ws_file="./training_models/resnet.pth").eval().to(device)
    cv2.namedWindow("preview", cv2.WINDOW_NORMAL)

    while True:
        rval, frame = cam.read()

        if not rval:
            break

        bw = binarize_image(frame)
        cnts = card_contours(bw)
        cnts = separate_cards(frame, cnts)
        frame_heaven, frame_q, frame_hell, frame_hand = card_positions(cnts)

        frame_cnts = frame.copy()
        cv2.drawContours(frame_cnts, cnts, -1, (0, 255, 0), 1)

        hell_lbl, heaven_lbl = None, None
        if frame_heaven is not None:
            frame_cnts, heaven_lbl = put_labels(frame, [frame_heaven], "heaven", class_model, frame_cnts, device)
        frame_cnts, q_lbls = put_labels(frame, frame_q, "q", class_model, frame_cnts, device)
        if frame_hell is not None:
            frame_cnts, hell_lbl = put_labels(frame, [frame_hell], "hell", class_model, frame_cnts, device)
        frame_cnts, hand_lbls = put_labels(frame, frame_hand, "h", class_model, frame_cnts, device)

        key = cv2.pollKey()
        if key == 27:  # escape
            break
        elif key == 13:  # intro
            # after all the movements are made in the board but opening heaven's doors
            # so if there are five cards in queue the first 2 will enter heaven and the last goes to hell
            if (l := len([c for c in q_lbls if c not in q])) > 1:
                logger.error(
                    f"""{l} new cards have been detected in the queue, which
                means 2 or more turns have passed since last state actualization or
                there has been a detection error
                previous queue: {q}, new queue: {q_lbls}"""
                )
                continue

            if (l := len([c for c in hand_lbls if c not in hand])) > 1 and hand:
                logger.error(
                    f"""{l} new cards have been detected in the hand, which
                means 2 or more turns have passed since last state actualization or
                there has been a detection error
                previous hand: {hand}, new hand: {hand_lbls}"""
                )
                continue

            if (heaven_lbl and not heaven) or (heaven_lbl and heaven_lbl[0] not in heaven):
                logger.error(
                    """Change in heaven detected, if the heaven's door have
                been opened please actualize the state BEFORE that, when all movements
                on the queue have been made but all the five cards are still in it"""
                )
                continue

            if not hell_lbl and hell or not heaven_lbl and heaven:
                logger.error("Detection error, the hell or heaven card disapeared")
                continue

            if len(q_lbls) == 5:
                heaven.extend(q_lbls[:2])
                hell.append(q_lbls[-1])
                new_heaven_cards = list(map(lambda x: Card(CardType(x[1]), Color(x[0])), q_lbls[:2]))
                new_hell_card = Card(CardType(q_lbls[-1][1]), Color(q_lbls[-1][0]))
                logger.info(f"{new_heaven_cards} enter in heaven")
                logger.info(f"{new_hell_card} goes to hell")
                q = q_lbls[2:-1]
            elif (hell_lbl and not hell) or (hell_lbl and hell_lbl[0] not in hell):
                new_hell_cards = [c for c in q if c not in q_lbls]
                if hell_lbl[0] not in new_hell_cards:
                    new_hell_cards.append(hell_lbl[0])
                hell.extend(new_hell_cards)
                logger.info(f"{new_hell_cards} went to hell")
                hand = hand_lbls
                q = q_lbls
            else:
                hand = hand_lbls
                q = q_lbls

            if len(color_pos_map) < num_players:
                ia_color = hand[0][0]
                color_pos_map = list(set([c for c, _ in heaven + q + hell if c != ia_color]))
                color_pos_map.insert(0, ia_color)

            hand.sort(key=lambda x: x[1])
            while True:
                state = create_state_rep(
                    heaven, hell, q, hand, color_pos_map, chosen_card=chosen_card, num_players=num_players, game_mode=game_mode
                )

                if state[1:].sum() == num_cards:
                    print("game has ended, count cards in heaven to see the winner")
                    return

                logger.debug(state)
                action, _ = model.predict(state)
                print(f"The AI chooses action {action}")

                if chosen_card is None:
                    card = Card(CardType(hand[action][1]), Color(hand[action][0]))
                    print(f"The AI plays {card}")

                    if card.has_action():
                        chosen_card = hand.pop(action)
                    else:
                        chosen_card = None
                        break
                else:
                    # camaleon
                    if chosen_card[1] == 5:
                        action = min(action, len(q))
                        card = Card(CardType(q[action][1]), Color(q[action][0]))
                        print(f"The camaleon transforms into {card}")
                        if card.has_action():
                            chosen_card = q[action]
                        else:
                            chosen_card = None
                            break
                    else:
                        match chosen_card[1]:
                            case 3:
                                print(f"the canguro jumps {action} cards")
                            case 2:
                                print(f"the loro scares the card in position {action}")
                        chosen_card = None
                        break

        cv2.imshow("preview", frame_cnts)


def main():
    parser = argparse.ArgumentParser(prog="main")

    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help="File of the model to load in human vs AI option",
    )
    parser.add_argument(
        "--agent1-class",
        type=str,
        choices=["DQN", "PPO", "QRDQN", "TRPO"],
        default="DQN",
        help='First agent\'s class ["DQN", "PPO", "QRDQN", "TRPO"], this argument is ignored if --agent argument is not defined',
    )
    parser.add_argument(
        "--agent2-class",
        type=str,
        choices=["DQN", "PPO", "QRDQN", "TRPO"],
        default="DQN",
        help='Second agent\'s class ["DQN", "PPO", "QRDQN", "TRPO"], this argument is ignored if --agent-v-agent is not defined',
    )
    parser.add_argument(
        "--agent-v-agent",
        type=str,
        metavar=("AGENT1", "AGENT2"),
        nargs=2,
        default=None,
        help="AI vs AI option, they play a total of num-games games and prints the results",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=1000,
        help="Number of games to be played, only used in AI vs AI option",
    )
    parser.add_argument(
        "--game-mode",
        type=str,
        default="full",
        choices=["full", "medium", "basic"],
        help='Game mode option to play, possible options: "basic", "medium", "full"',
    )
    parser.add_argument(
        "--num-players",
        type=int,
        choices=[2, 3, 4],
        default=2,
        help="Number of players in game, possible numbers: 2, 3, 4",
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="""Uses a webcam and CV to detect the cards in the board and create the board state to
        use as input for the model in real time, if this argument is defined only one agent is available to play,
        the rest must be humans""",
    )

    args = parser.parse_args()
    model = None
    logging.basicConfig(level=logging.DEBUG)

    if args.agent:
        model = load_model(args.agent, args.agent1_class)

    if args.cv:
        assert model is not None, "--agent must be defined to play"
        cv_loop(model, args.num_players, args.game_mode)
        exit(0)

    if args.agent_v_agent:
        model1 = load_model(args.agent_v_agent[0], args.agent1_class)
        model2 = load_model(args.agent_v_agent[1], args.agent2_class)
        env = BarEnv(num_players=args.num_players, game_mode=args.game_mode)
        wins1, wins2, draws = model_v_model(model1, model2, args.num_games, env)
        print(f"Model1: {args.agent_v_agent[0]} wins {wins1} times.")
        print(f"Model2: {args.agent_v_agent[1]} wins {wins2} times.")
        print(f"They draw {draws} times.")
        exit(0)


    env = BarEnv(game_mode=args.game_mode, render_mode="human", t=1, num_players=args.num_players)
    obs, _ = env.reset()

    while True:
        if env.game.turn == env.agent_color.value and model:
            action = model.predict(obs, deterministic=True)[0]
        else:
            msg = env.game.action_msg()
            msg = msg if msg else ""
            action = int(input(msg + ": "))

        obs, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            break

    print(f"Cards in hell: {", ".join([str(c) for c in env.game.hell])}")
    print(f"Cards in heaven: {", ".join([str(c) for c in env.game.heaven])}")
    print(f"The winners are: {", ".join([c.name for c in env.game.winners()])}")


if __name__ == "__main__":
    main()
