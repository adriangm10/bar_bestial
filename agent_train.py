import argparse
import random
from typing import Literal

from sb3_contrib import QRDQN, TRPO
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
        checkpoints_dir: str | None = None,
        save_checkpoint_after_n_episodes: int = 50_000,
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
        self.checkpoints_dir = checkpoints_dir
        self.save_checkpoint_after_n_episodes = save_checkpoint_after_n_episodes

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
                    num_games=1000,
                    game_mode=self.env.game_mode,
                    num_players=self.env.num_players,
                    t=self.env.t,
                    transfer_learning=self.transfer_learning,
                )

                if self.verbose > 0:
                    print(f"Stats against random actions: {{ wins: {wins}, losses: {losses}, draws: {draws} }}")

                if self.stats_file_name:
                    self.stats_file.write(f"{self.episodes},{wins},{losses},{draws}\n")

            if self.checkpoints_dir and self.episodes % self.save_checkpoint_after_n_episodes == 0:
                self.model.save(self.checkpoints_dir + f"/model_episode{self.episodes}")
                if self.verbose > 0:
                    print(f"New model checkpoint saved in {self.checkpoints_dir}/model_episode{self.episodes}")

        return True

    def _on_training_end(self) -> None:
        if self.stats_file_name:
            self.stats_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="train agent")
    algorithms = {
        "PPO": PPO,
        "DQN": DQN,
        "QRDQN": QRDQN,
        "TRPO": TRPO,
    }

    parser.add_argument(
        "--alg", type=str, choices=["DQN", "PPO", "QRDQN", "TRPO"], default="DQN", help="RL algorithm to use"
    )
    parser.add_argument(
        "--num-players",
        type=int,
        choices=[2, 3, 4],
        default=2,
        help="Number of players in game, possible numbers: 2, 3, 4. Default is set to 2",
    )
    parser.add_argument(
        "--game-mode",
        type=str,
        default="full",
        choices=["full", "medium", "basic"],
        help='Game mode option to play, possible options: "basic", "medium", "full". Default is set to full.',
    )
    parser.add_argument(
        "--num-timesteps",
        type=int,
        default=15_000_000,
        help="Number of timesteps to train for, each games is approximately 15 timesteps in 'medium' and 'full' gamemodes and 9 in 'basic'. Default is set to 15_000_000 (1M games)",
    )
    parser.add_argument(
        "--continue-training",
        type=str,
        default=None,
        metavar="FILE_NAME",
        help="If a file is given the script will continue training the model in that file",
    )
    parser.add_argument(
        "--stats-file",
        type=str,
        default=None,
        metavar="FILE_NAME",
        help="A file to save the evaluations against random actions during the training.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default=None,
        metavar="FILE_NAME",
        help="A file to save model checkpoints during the training.",
    )
    parser.add_argument(
        "--save-model-file",
        type=str,
        required=True,
        metavar="FILE_NAME",
        help="File to save the trained model",
    )

    args = parser.parse_args()
    env = BarEnv(game_mode=args.game_mode, self_play=True, t=1, num_players=args.num_players, transfer_learning=None)

    model = (
        algorithms[args.alg].load(args.continue_training)
        if args.continue_training
        else algorithms[args.alg]("MlpPolicy", env, verbose=0)
    )

    selfplay_callback = SelfPlayCallback(
        env,
        update_after_n_episodes=1000,
        verbose=1,
        evaluate_after_n_episodes=25_000,
        stats_file=args.stats_file,
        stats_file_mode="w" if args.continue_training is None else "a",
        transfer_learning=None,
        checkpoints_dir=args.checkpoints_dir,
    )
    model.learn(total_timesteps=args.num_timesteps, callback=selfplay_callback, reset_num_timesteps=False)

    model.save(args.save_model_file)

    # model = DQN.load("models/dqn/t1p2finaldqn")
    wins, losses, draws = model_v_random(model, num_games=1000, game_mode="full", t=1, num_players=2)
    print(f"wins: {wins}, losses: {losses}, draws: {draws}")
