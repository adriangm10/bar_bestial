import random
from copy import deepcopy

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from bar_gym import BarEnv


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
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """

        if self.locals.get("done"):
            self.episodes += 1

            if self.episodes % self.update_after_n_episodes == 0:
                self.model.save(self.temp_model_path)
                self.env.opponent_model = self.model.__class__.load(self.temp_model_path)
                if self.verbose > 0:
                    print(f"Updated opponent model at episode {self.episodes}")

        return True


def model_v_random(model, num_games=100):
    env = BarEnv(game_mode="basic", self_play=False)
    wins = 0
    losses = 0
    draws = 0

    for _ in range(num_games):
        obs, _ = env.reset()
        while True:
            if env.game.turn == env.agent_color:
                action = model.predict(obs, deterministic=True)[0]
            else:
                action = random.randint(0, len(env.game.hands[env.game.turn]) - 1)

            obs, r, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                if r > 0:
                    wins += 1
                elif r == 0:
                    draws += 1
                else:
                    losses += 1
                break

    return wins, losses, draws


if __name__ == "__main__":
    env = BarEnv(game_mode="basic")
    model = DQN("MlpPolicy", env, buffer_size=10_000)
    eval_callback = EvalCallback(env, best_model_save_path="models/dqn", eval_freq=5000, n_eval_episodes=1000)
    selfplay_callback = SelfPlayCallback(env, update_after_n_episodes=1000, verbose=1)
    model.learn(total_timesteps=500_000, callback=[eval_callback, selfplay_callback])

    model.save("models/dqn/dqn_final")

    model = DQN.load("models/dqn/best_model")
    wins, losses, draws = model_v_random(model)
    print(f"wins: {wins}, losses: {losses}, draws: {draws}")
