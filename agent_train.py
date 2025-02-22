import random
from typing import Literal

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback

from bar_gym import BarEnv


def model_v_random(
    model,
    num_games=100,
    game_mode: Literal["basic", "medium", "full"] = "full",
    t=1,
    num_players: Literal[2, 3, 4] = 2,
    transfer_learning=None,
):
    env = BarEnv(
        game_mode=game_mode, self_play=False, t=t, num_players=num_players, transfer_learning=transfer_learning
    )
    wins = 0
    losses = 0
    draws = 0

    for _ in range(num_games):
        obs, _ = env.reset()
        while True:
            poss_actions = env.game.possible_actions()
            if env.game.turn == env.agent_color.value:
                action, _ = model.predict(obs)
            else:
                action = random.choice(poss_actions)

            obs, _, terminated, truncated, _ = env.step(action)
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
        evaluate_after_n_episodes: int = 1000,
        stats_file: str | None = None,
        stats_file_mode: str = "w",
        transfer_learning=None,
    ):
        super().__init__(verbose)
        self.env = env
        self.episodes = 0
        self.opponent = None
        self.update_after_n_episodes = update_after_n_episodes
        self.temp_model_path = temp_model_path
        self.evaluate_after_n_episodes = evaluate_after_n_episodes
        self.stats_file_name = stats_file
        self.stats_file_mode = stats_file_mode
        self.transfer_learning = transfer_learning

    def _on_training_start(self) -> None:
        if self.stats_file_name:
            self.stats_file = open(self.stats_file_name, self.stats_file_mode)
            if self.stats_file_mode != "a":
                self.stats_file.write("episode,wins,losses,draws\n")

    def _on_step(self) -> bool:
        if self.locals.get("done"):
            self.episodes += 1

            if self.episodes % self.update_after_n_episodes == 0:
                self.model.save(self.temp_model_path)
                self.env.set_opponent_model(self.model.__class__.load(self.temp_model_path))
                if self.verbose > 0:
                    print(f"Updated opponent model at episode {self.episodes}")

            if self.episodes % self.evaluate_after_n_episodes == 0:
                wins, losses, draws = model_v_random(
                    self.model,
                    num_games=100,
                    game_mode=self.env.game_mode,
                    num_players=self.env.num_players,
                    t=self.env.t,
                    transfer_learning=self.transfer_learning,
                )

                if self.verbose > 0:
                    print(f"Stats against random actions: {{ wins: {wins}, losses: {losses}, draws: {draws} }}")

                if self.stats_file_name:
                    self.stats_file.write(f"{self.episodes},{wins},{losses},{draws}\n")

        return True

    def _on_training_end(self) -> None:
        if self.stats_file:
            self.stats_file.close()


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

    env = BarEnv(game_mode="full", self_play=True, t=1, num_players=2, transfer_learning=None)
    # eval_env = BarEnv(game_mode="basic")
    model = DQN.load("./models/dqn/t1p2full2Mdqn.zip")
    model.set_env(env)
    # model = DQN("MlpPolicy", env, verbose=0)
    # model = PPO("MlpPolicy", env, verbose=0)
    # eval_callback = EvalCallback(eval_env, best_model_save_path="models/dqn", eval_freq=5000, n_eval_episodes=1000)
    selfplay_callback = SelfPlayCallback(
        env,
        update_after_n_episodes=1000,
        verbose=1,
        evaluate_after_n_episodes=2500,
        stats_file=None,
        stats_file_mode="w",
        transfer_learning=None,
    )
    model.learn(total_timesteps=10_500_000, callback=selfplay_callback, reset_num_timesteps=False)

    model.save("models/dqn/t1p2full2Mdqn")

    # model = DQN.load("models/dqn/t1p2finaldqn")
    wins, losses, draws = model_v_random(model, num_games=1000, game_mode="full", t=1, num_players=2)
    print(f"wins: {wins}, losses: {losses}, draws: {draws}")
