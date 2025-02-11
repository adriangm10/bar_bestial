import logging
import sys
from random import randint

from stable_baselines3 import DQN, PPO

from bar import CardType, Game
from bar_gym import BarEnv

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    ai = False
    if len(sys.argv) > 1:
        model = DQN.load(sys.argv[1])
        ai = True

    env = BarEnv(game_mode="full", render_mode="human", t=1, num_players=2)
    obs, _ = env.reset()

    while True:
        if env.game.turn == env.agent_color.value and ai:
            action = model.predict(obs, deterministic=True)[0]
        else:
            msg = env.game.action_msg()
            msg = msg if msg else ""
            action = int(input(msg + ": "))

        obs, r, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            break

    print(f"agent reward: {r}")
    print(f"Cards in hell: {", ".join([str(c) for c in env.game.hell])}")
    print(f"Cards in heaven: {", ".join([str(c) for c in env.game.heaven])}")
    print(f"The winners are: {", ".join([c.name for c in env.game.winners()])}")
