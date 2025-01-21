from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from bar_gym import BarEnv

env = BarEnv(game_mode="basic")


# class CustomCallback(BaseCallback):
#     def __init__(self, verbose: int = 0):
#         super().__init__(verbose)
#         self.opponent = DQN("MlpPolicy", env)
#
#     def _on_training_start(self) -> None:
#         """
#         This method is called before the first rollout starts.
#         """
#         pass
#
#     def _on_rollout_start(self) -> None:
#         """
#         A rollout is the collection of environment interaction
#         using the current policy.
#         This event is triggered before collecting new samples.
#         """
#         pass
#
#     def _on_step(self) -> bool:
#         """
#         This method will be called by the model after each call to `env.step()`.
#
#         For child callback (of an `EventCallback`), this will be called
#         when the event is triggered.
#
#         :return: If the callback returns False, training is aborted early.
#         """
#         if self.n_calls % 1000 == 0:
#             self.opponent.set_parameters(self.model.get_parameters())  # type: ignore[arg-type]
#
#         # self.training_env.step(self.opponent.predict(self.training_env._get_obs()))
#         return True
#
#     def _on_rollout_end(self) -> None:
#         """
#         This event is triggered before updating the policy.
#         """
#         pass
#
#     def _on_training_end(self) -> None:
#         """
#         This event is triggered before exiting the `learn()` method.
#         """
#         pass


model = DQN("MlpPolicy", env, verbose=1)
eval_callback = EvalCallback(env, best_model_save_path="models/dqn", eval_freq=5000, n_eval_episodes=1000)
model.learn(total_timesteps=50000, callback=eval_callback)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=200, deterministic=True)
print(f"mean_reward: {mean_reward}, std_reward: {std_reward}")

model.save("models/dqn_final")
