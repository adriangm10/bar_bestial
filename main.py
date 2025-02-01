import logging
import sys

from stable_baselines3 import DQN, PPO

from bar import CardType, Game
from bar_gym import BarEnv

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    ai = False
    if len(sys.argv) > 1:
        model = DQN.load(sys.argv[1])
        ai = True

    env = BarEnv(game_mode="basic", render_mode="human")
    obs, _ = env.reset()

    while True:
        if env.game.turn == env.agent_color.value and ai:
            action = model.predict(obs, deterministic=True)[0]
        else:
            pos = int(input("Select a card to play[0-3]: "))
            action = pos  # type: ignore[assignment]
            actions: list[int] = []
            msg = env.game.hands[env.game.turn][pos].action_msg()
            if msg:
                actions.append(int(input(msg + ": ")))
                if env.game.hand_card(pos).card_type == CardType.CAMALEON and (
                    msg := env.game.table_cards[actions[0]].action_msg()  # type: ignore[union-attr]
                ):
                    actions.append(int(input(msg + ": ")))

        obs, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            break

    print(f"Cards in hell: {", ".join([str(c) for c in env.game.hell])}")
    print(f"Cards in heaven: {", ".join([str(c) for c in env.game.heaven])}")
    print(f"The winners are: {", ".join([c.name for c in env.game.winners()])}")
