import argparse
import logging
import random
from typing import Literal

import numpy as np
from sb3_contrib import QRDQN, TRPO
from stable_baselines3 import DQN, PPO

from game.bar import QUEUE_LEN, Card, CardType, Color, TableCards
from game.bar_gym import BarEnv

type RLModel = QRDQN | DQN | TRPO | PPO
type CardRep = tuple[int, int]  # color, force

logger = logging.getLogger("bar bestial")


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


def model_v_model(models, num_games: int, env: BarEnv) -> tuple[list[int], int]:
    wins = [0 for _ in models]
    draws = 0

    for _ in range(num_games):
        obs, _ = env.reset()

        while True:
            poss_actions = env.game.possible_actions()
            action, _ = models[env.game.turn].predict(obs)

            if poss_actions and action not in poss_actions:
                action = random.choice(poss_actions)

            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                winners = env.game.winners()
                if len(winners) == env.num_players or not winners:
                    draws += 1
                for c in winners:
                    wins[c.value] += 1
                break

    return wins, draws


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


def log_error(condition: bool, msg: str, printed_error: bool) -> bool:
    if condition:
        if not printed_error:
            logger.error(msg)
        return True
    else:
        return False


def turn_result(
    queue: list[Card], hell: list[Card], heaven: list[Card], actions: list[int] = []
) -> tuple[list[TableCards], list[list[Card]], list[list[Card]]]:
    if actions:
        logger.debug(f"turn_result: turn actions: {actions}")

    poss_queues, poss_hells, poss_heavens = [], [], []

    played_card = queue[-1]
    action_fun = played_card.action()
    correct_queue = queue + [None] * (QUEUE_LEN - len(queue))

    if not actions and (actions := played_card.card_type.possible_options(queue[:-1])):  # type: ignore
        if played_card.card_type == CardType.CAMALEON:
            new_actions = []
            for action in actions:
                if correct_queue[action] is None:
                    logger.error(
                        "FATAL ERROR: correct_queue[action] is None, the action of the camaleon is not possible"
                    )
                    exit(1)

                if acts := correct_queue[action].card_type.possible_options(queue[:-1]):  # type: ignore
                    for act in acts:
                        new_actions.append([action, act])
                else:
                    new_actions.append([action])

            actions = new_actions

        for action in actions:
            if isinstance(action, int):
                action = [action]

            queue_cpy = correct_queue.copy()
            hell_cpy = hell.copy()

            queue_cpy, hell_cpy = action_fun(len(queue) - 1, queue_cpy, hell_cpy, action)
            poss_queues.append(queue_cpy)
            poss_hells.append(hell_cpy)
            poss_heavens.append(heaven.copy())
    else:
        queue_cpy = correct_queue.copy()
        hell_cpy = hell.copy()

        queue_cpy, hell_cpy = action_fun(len(queue) - 1, queue_cpy, hell_cpy, actions)  # type: ignore
        poss_queues.append(queue_cpy)
        poss_hells.append(hell_cpy)
        poss_heavens.append(heaven.copy())

    assert len(poss_heavens) == len(poss_queues) == len(poss_hells)

    for i, q in enumerate(poss_queues):
        for j, c in enumerate(q):
            if c and c != played_card and c.recursive:
                poss_queues[i], poss_hells[i] = c.action()(j, q, poss_hells[i], [])

        if len(q) == 5 and all([c is not None for c in q]):
            poss_heavens[i].extend([q[0], q[1]])
            poss_hells[i].append(q[-1])
            q[0], q[1] = q[2], q[3]
            q[2], q[3], q[4] = None, None, None

    logger.info(f"possible queues: {poss_queues},\n possible hells: {poss_hells},\n possible heavens: {poss_heavens}")
    return poss_queues, poss_hells, poss_heavens


def cv_auto(model: RLModel, num_players: Literal[2, 3, 4], game_mode: Literal["full", "medium", "basic"]) -> None:
    import cv2
    import torch
    from cv2.typing import MatLike

    from cv.camera import camera_idxs, put_labels
    from cv.card_classification import get_class_model
    from cv.card_detection import binarize_image, card_contours, card_positions, separate_cards

    def is_stable(prev_gray_img: MatLike, gray_img: MatLike, threshold: int = 1000) -> bool:
        assert len(prev_gray_img.shape) == 2 and len(gray_img.shape) == 2
        diff = cv2.absdiff(prev_gray_img, gray_img)
        _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        count = cv2.countNonZero(th)

        return count <= threshold

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running in {device} device")

    heaven, hell, q, hand, color_pos_map = [], [], [], [], []
    chosen_card = None
    num_cards = 12 * num_players if game_mode != "basic" else 9 * num_players
    turn = random.randint(0, num_players - 1)
    agent_turn = 0
    poss_hells, poss_queues, poss_heavens = None, None, None
    agent_actions = []
    ai_color = 0
    played_card = None

    cam_idxs = camera_idxs()
    cam = cv2.VideoCapture(cam_idxs[-1])
    class_model = get_class_model(ws_file="./training_models/prueba_sin_subsets.pth", device=device).eval().to(device)
    cv2.namedWindow("bar bestial", cv2.WINDOW_NORMAL)
    error = False
    game_started = False
    frame_cnts = None

    logger.info(f"The agent will be player number {agent_turn}")
    logger.info(f"Starts player number {turn}")
    logger.info("Press [Enter] to start")
    logger.info("the scene is not stable")

    rval, frame = cam.read()
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    stable_frame_counter = 0
    STABLE_FRAME_TRESHOLD = 20

    while True:
        rval, frame = cam.read()

        if not rval:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if is_stable(prev_gray, gray_frame):
            stable_frame_counter += 1
        else:
            if stable_frame_counter >= STABLE_FRAME_TRESHOLD:
                logger.info("the scene is not stable")
            stable_frame_counter = 0
        prev_gray = gray_frame

        if stable_frame_counter == STABLE_FRAME_TRESHOLD:
            logger.info("The scene is stable")

        if frame_cnts is not None:
            cv2.imshow("bar bestial", frame_cnts)

        bw = binarize_image(frame)
        cnts = card_contours(bw)
        cnts = separate_cards(frame, cnts)
        frame_heaven, frame_q, frame_hell, frame_hand = card_positions(cnts)
        frame_cnts = frame.copy()
        frame_cnts = cv2.drawContours(frame_cnts, cnts, -1, (0, 255, 0), 1)

        key = cv2.pollKey()
        if key == 27:  # escape
            break

        if log_error(
            len(frame_q) < 2 and game_started and stable_frame_counter >= STABLE_FRAME_TRESHOLD,
            "Heaven door and kick cards aren't being detected",
            error,
        ):
            error = True
            continue
        elif game_started:
            frame_q = frame_q[1:-1]

        # hell_lbl, heaven_lbl = None, None
        # if frame_heaven is not None:
        #     frame_cnts, heaven_lbl = put_labels(frame, [frame_heaven], "heaven", class_model, frame_cnts, device)
        frame_cnts, q_lbls = put_labels(frame, frame_q, "q", class_model, frame_cnts, device)
        # if frame_hell is not None:
        #     frame_cnts, hell_lbl = put_labels(frame, [frame_hell], "hell", class_model, frame_cnts, device)
        frame_cnts, hand_lbls = put_labels(frame, frame_hand, "h", class_model, frame_cnts, device)

        if game_started and stable_frame_counter >= STABLE_FRAME_TRESHOLD:
            # if log_error(
            #     not bool(frame_hell) and bool(hell) or not bool(frame_heaven) and bool(heaven),
            #     "Detection error, the hell or heaven card disapeared",
            #     error,
            # ):
            #     error = True
            #     continue

            length = len([c for c in q_lbls if c not in q])
            if log_error(length > 1, f"{length} changes have been detected in the queue which is not possible.", error):
                error = True
                continue

            if log_error(len(hand) > 1 and not hand_lbls, "Hand cards are not being detected", error):
                error = True
                continue

            if log_error(
                len([c for c in hand_lbls if c not in hand]) > 1
                or abs(len(hand) - len(hand_lbls)) > 1
                or any([c != ai_color for c, _ in hand_lbls]),
                f"The hand is not being correctly detected; hand: {hand}, hand_lbls: {hand_lbls}",
                error,
            ):
                error = True
                continue

            hand = hand_lbls

            if poss_queues and poss_hells and poss_heavens:
                queue = [Card(CardType(f), Color(c)) for c, f in q_lbls] + ([None] * (QUEUE_LEN - len(q_lbls)))
                try:
                    i = poss_queues.index(queue)
                    q = q_lbls
                    hell = [(c.color.value, c.card_type.value) for c in poss_hells[i]]
                    heaven = [(c.color.value, c.card_type.value) for c in poss_heavens[i]]

                    if hell:
                        logger.info(f"updated hell: {hell}")

                    if heaven:
                        logger.info(f"updated heaven: {heaven}")

                    turn = (turn + 1) % num_players
                    logger.info(
                        f"----------The queue has been correctly updated, next turn is for player {turn}.-------------"
                    )
                    poss_hells, poss_queues, poss_heavens = None, None, None
                    agent_actions.clear()
                    played_card = None
                except ValueError:
                    continue

            # there is a new card in q
            elif len(q_lbls) - len(q) == 1:
                if log_error(
                    len(color_pos_map) == num_players and q_lbls[-1][0] not in color_pos_map,
                    f"""The color of the new card is being wrongly detected or there are too many players.
                            Number of players that were supposed to be: {num_players}""",
                    error,
                ):
                    error = True
                    continue

                if log_error(
                    turn < len(color_pos_map) and color_pos_map[turn] != q_lbls[-1][0],
                    f"Is not your turn, is turn of player {turn}",
                    error,
                ):
                    error = True
                    continue

                if log_error(
                    played_card is not None and played_card != q_lbls[-1],
                    f"The agent played card {played_card} but the last card in the queue is {q_lbls[-1]}",
                    error,
                ):
                    error = True
                    continue

                logger.info(f"Card {q_lbls[-1]} has been played.")

                if len(color_pos_map) < num_players and q_lbls[-1][0] not in color_pos_map:
                    color_pos_map.append(q_lbls[-1][0])
                    logger.debug(f"color_pos_map updated: {color_pos_map}")

                q = q_lbls
                queue = [Card(CardType(f), Color(c)) for c, f in q]
                queue_cmp = queue + [None] * (QUEUE_LEN - len(queue))
                c_hell = [Card(CardType(f), Color(c)) for c, f in hell]
                c_heaven = [Card(CardType(f), Color(c)) for c, f in heaven]
                poss_queues, poss_hells, poss_heavens = turn_result(queue, c_hell, c_heaven, agent_actions)

                if (
                    turn != agent_turn
                    and any([qi == queue_cmp for qi in poss_queues])
                    and any([qi != queue_cmp for qi in poss_queues])
                ):
                    print("¡¡¡ The queue might change or not depending on the human action !!!")
                    decision = None
                    while decision is None:
                        yn = input("Will the queue change after your action?[y/n] ")
                        yn = yn.lower()
                        if yn == "y" or yn == "n":
                            decision = yn
                        else:
                            print("just y/n answer")

                    if decision == "y":
                        indxs = [i for i, qi in enumerate(poss_queues) if qi != queue_cmp]
                        poss_queues = [poss_queues[i] for i in indxs]
                        poss_hells = [poss_hells[i] for i in indxs]
                        poss_heavens = [poss_heavens[i] for i in indxs]
                    else:
                        indxs = [i for i, qi in enumerate(poss_queues) if qi == queue_cmp]
                        poss_queues = [poss_queues[i] for i in indxs]
                        poss_hells = [poss_hells[i] for i in indxs]
                        poss_heavens = [poss_heavens[i] for i in indxs]

            elif turn == agent_turn and played_card is None:
                assert agent_actions == []
                if not hand:
                    logger.info("There are no cards in hand, game has ended")
                    return

                sorted_hand = sorted(hand_lbls, key=lambda x: x[1])
                while True:
                    state = create_state_rep(
                        heaven,
                        hell,
                        q,
                        sorted_hand,
                        color_pos_map,
                        chosen_card=chosen_card,
                        num_players=num_players,
                        game_mode=game_mode,
                    )

                    logger.debug(f"generated state:\n{state}")
                    option, _ = model.predict(state)
                    logger.info(f"The AI chooses action {option}")

                    if chosen_card is None:
                        if option >= len(sorted_hand):
                            option = random.choice(list(range(len(sorted_hand))))
                        card = Card(CardType(sorted_hand[option][1]), Color(sorted_hand[option][0]))
                        played_card = sorted_hand[option]
                        logger.info(f"The AI plays {card}")

                        if card.has_options():
                            chosen_card = sorted_hand.pop(option)
                        else:
                            break
                    else:
                        poss_options = CardType(chosen_card[1]).possible_options(
                            [Card(CardType(f), Color(c)) for c, f in q]
                        )
                        if option not in poss_options and poss_options:
                            option = random.choice(poss_options)
                        agent_actions.append(option)

                        # camaleon
                        if chosen_card[1] == 5 and poss_options:
                            card = Card(CardType(q[option][1]), Color(q[option][0]))
                            logger.info(f"The camaleon transforms into {card}")
                            if card.has_options():
                                chosen_card = q[option]
                            else:
                                chosen_card = None
                                break
                        else:
                            match chosen_card[1]:
                                case 3:
                                    logger.info(f"the canguro jumps {option} cards")
                                case 2:
                                    logger.info(f"the loro scares the card in position {option}")
                            chosen_card = None
                            break

            if len(q) <= 4 and not hand:
                state = create_state_rep(
                    heaven,
                    hell,
                    q,
                    hand,
                    color_pos_map,
                    num_players=num_players,
                    game_mode=game_mode,
                )

                if state[1:].sum() == num_cards:
                    print("GAME ENDED, count cards in heaven to see the winner")
                    return

        error = False
        if key == 13:  # intro
            if stable_frame_counter < STABLE_FRAME_TRESHOLD:
                logger.error("There is movement on the scene, the game cannot start.")
                continue

            if game_started:
                continue

            if len(frame_q) != 2:
                logger.error(
                    "The queue is not being correctly detected, more or less than 2 cards are being detected. The game cannot be started."
                )
                continue

            if frame_hell or frame_heaven:
                logger.error(
                    "Hell or Heaven is being detected, which shouldn't be possible at start of the game. The game cannot be started."
                )
                continue

            if (length := len(hand_lbls)) != 4:
                logger.error(f"{length} cards detected in hand, which should be 4. The game cannot be started.")
                continue

            if any([c != hand_lbls[0][0] for c, _ in hand_lbls]):
                logger.error("different colors detected in hand cards.")
                continue

            assert color_pos_map == []
            color_pos_map.append(hand_lbls[0][0])
            ai_color = hand_lbls[0][0]
            logger.debug(f"color_pos_map updated: {color_pos_map}")
            logger.info(f"GAME STARTED: starts player {turn} (the bot is player {agent_turn})")
            hand = hand_lbls
            game_started = True


def main():
    parser = argparse.ArgumentParser(prog="main")

    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help="File of the model to load in human vs AI option",
    )
    parser.add_argument(
        "--algs",
        type=str,
        choices=["DQN", "PPO", "QRDQN", "TRPO"],
        nargs="+",
        metavar="LIST OF ALGORITHMS",
        help='The agent\'s algorithm ["DQN", "PPO", "QRDQN", "TRPO"]',
    )
    parser.add_argument(
        "--agent-v-agent",
        type=str,
        metavar="LIST OF AGENTS",
        nargs="+",
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
        "--cv-auto",
        action="store_true",
        help="""Uses a webcam and CV to detect the cards in the board and create the board state to
        use as input for the model in real time, if this argument is defined only one agent is available to play,
        the rest must be humans. The turn is detected automatically.""",
    )
    parser.add_argument("--debug", action="store_true", help="Set logging level to debug")

    args = parser.parse_args()
    model = None

    if args.agent_v_agent:
        assert len(args.agent_v_agent) == args.num_players and len(args.algs) == args.num_players
        models = [load_model(args.agent_v_agent[i], args.algs[i]) for i in range(args.num_players)]
        env = BarEnv(num_players=args.num_players, game_mode=args.game_mode)
        wins, draws = model_v_model(models, args.num_games, env)
        for w, name in zip(wins, args.agent_v_agent):
            print(f"Model1: {name} wins {w} times.")
        print(f"They draw {draws} times.")
        exit(0)

    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level)

    if args.agent:
        model = load_model(args.agent, args.algs[0])

    if args.cv_auto:
        assert model is not None, "--agent must be defined to play"
        cv_auto(model, args.num_players, args.game_mode)
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
