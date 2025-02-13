import random
from copy import deepcopy

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from bar_gym import BarEnv


def model_v_random(model, num_games=100, game_mode="full"):
    env = BarEnv(game_mode=game_mode, self_play=False)
    wins = 0
    losses = 0
    draws = 0

    for _ in range(num_games):
        obs, _ = env.reset()
        while True:
            poss_actions = env.game.possible_actions()
            if env.game.turn == env.agent_color:
                action = random.choice(poss_actions)
            else:
                action = random.choice(poss_actions)

            obs, r, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                winners = env.game.winners()
                if len(winners) == env.num_players or not winners:
                    draws += 1
                elif env.agent_color in winners:
                    wins += 1
                else:
                    losses += 1
                break

    return wins, losses, draws


class SelfPlayCallback(BaseCallback):
    def __init__(
        self,
        env: BarEnv,
        update_after_n_episodes: int = 100,
        temp_model_path: str = "/tmp/tmp_train_model",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.env = env
        self.episodes = 0
        self.opponent = None
        self.update_after_n_episodes = update_after_n_episodes
        self.temp_model_path = temp_model_path

    def _on_step(self) -> bool:
        if self.locals.get("done"):
            self.episodes += 1

            if self.episodes % self.update_after_n_episodes == 0:
                self.model.save(self.temp_model_path)
                self.env.opponent_model = self.model.__class__.load(self.temp_model_path)
                if self.verbose > 0:
                    wins, losses, draws = model_v_random(self.model, num_games=50)
                    print(f"Updated opponent model at episode {self.episodes}")
                    print(f"Stats against random actions: {{ wins: {wins}, losses: {losses}, draws: {draws} }}")

        return True


if __name__ == "__main__":
    def lr(a):
        if a > 0.90:
            return 0.1
        elif a > 0.6:
            return 0.01
        elif a > 0.25:
            return 0.001
        else:
            return 0.0001

    env = BarEnv(game_mode="full", self_play=True, t=3, num_players=2)
    # eval_env = BarEnv(game_mode="basic")
    # model = DQN.load("./models/dqn/t1p2fdqn.zip")
    # model.set_env(env)
    # model = DQN("MlpPolicy", env, verbose=0)
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=lr)
    # eval_callback = EvalCallback(eval_env, best_model_save_path="models/dqn", eval_freq=5000, n_eval_episodes=1000)
    selfplay_callback = SelfPlayCallback(env, update_after_n_episodes=1000, verbose=1)
    model.learn(total_timesteps=15_000_000, callback=selfplay_callback)

    model.save("models/ppo/t3p2fppo")

    # model = DQN.load("models/dqn/t3dqn")
    wins, losses, draws = model_v_random(model, num_games=1000)
    print(f"wins: {wins}, losses: {losses}, draws: {draws}")
