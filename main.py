import argparse
import logging
from random import choice

from stable_baselines3 import DQN, PPO

from bar_gym import BarEnv


def load_model(file: str, model_class: str):
    match model_class:
        case "DQN":
            model = DQN.load(file)
        case "PPO":
            model = PPO.load(file)
        case _:
            raise ValueError(f"Not supported agent class: {args.agent1_class}")
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


if __name__ == "__main__":
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
        choices=["DQN", "PPO"],
        default="DQN",
        help='First agent\'s class "DQN" or "PPO", this argument is ignored if --agent argument is not defined',
    )
    parser.add_argument(
        "--agent2-class",
        type=str,
        choices=["DQN", "PPO"],
        default="DQN",
        help='Second agent\'s class "DQN" or "PPO", this argument is ignored if --agent-v-agent is not defined',
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

    args = parser.parse_args()
    model = None
    if args.agent:
        model = load_model(args.agent, args.agent1_class)
    if args.agent_v_agent:
        model1 = load_model(args.agent_v_agent[0], args.agent1_class)
        model2 = load_model(args.agent_v_agent[1], args.agent2_class)
        env = BarEnv(num_players=args.num_players, game_mode=args.game_mode)
        wins1, wins2, draws = model_v_model(model1, model2, args.num_games, env)
        print(f"Model1: {args.agent_v_agent[0]} wins {wins1} times.")
        print(f"Model2: {args.agent_v_agent[1]} wins {wins2} times.")
        print(f"They draw {draws} times.")
        exit(0)

    logging.basicConfig(level=logging.DEBUG)

    env = BarEnv(game_mode=args.game_mode, render_mode="human", t=1, num_players=args.num_players)
    obs, _ = env.reset()

    while True:
        if env.game.turn == env.agent_color.value and model:
            action = model.predict(obs, deterministic=True)[0]
        else:
            msg = env.game.action_msg()
            msg = msg if msg else ""
            action = int(input(msg + ": "))

        obs, r, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            break

    print(f"Cards in hell: {", ".join([str(c) for c in env.game.hell])}")
    print(f"Cards in heaven: {", ".join([str(c) for c in env.game.heaven])}")
    print(f"The winners are: {", ".join([c.name for c in env.game.winners()])}")
