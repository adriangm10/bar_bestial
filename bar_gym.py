from typing import Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from bar import Color, Game


class BarEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        opponent_model=None,
        num_players: Literal[2, 3, 4] = 2,
        game_mode: Literal["basic", "medium", "full"] = "full",
        agent_color: Color = Color(0),
        self_play: bool = True,
        render_mode=None,
    ):
        """
        observation space: (at the moment, with basic gamemode) a 9x3 matrix each row is a card
        that has (value, recursive, color), 5 for the table and 4 for the hand.
        action space: (for the basic gamemode) the card to be played Discrete(4)
        """
        if game_mode != "basic":
            raise NotImplementedError

        self.game = Game(num_players=num_players, game_mode=game_mode)
        self.num_players = num_players
        self.game_mode = game_mode
        self.agent_color = agent_color
        self.opponent_model = opponent_model
        self.self_play = self_play

        self.observation_space = spaces.Box(
            low=np.ones((9, 3)) * -1,
            high=np.array([np.array([12, 1, 3]) for _ in range(9)]),
            shape=(9, 3),
            dtype=np.int32,
        )
        self.action_space = spaces.Discrete(4)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        cards = [(c.value, int(c.recursive), c.color.value) if c else (-1, -1, -1) for c in self.game.table_cards]
        hand = [(c.value, int(c.recursive), c.color.value) for c in self.game.hands[self.game.turn]]
        hand += [(-1, -1, -1)] * (4 - len(hand))
        self.obs = np.array(cards + hand)

        return self.obs

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed, options=options)

        del self.game
        self.game = Game(self.num_players, self.game_mode)

        if self.render_mode == "human":
            self.game.print()

        return self._get_obs(), {}

    def step(self, action):
        assert self.action_space.contains(action)

        hand_count = len(self.game.hands[self.game.turn])
        if self.game.turn != self.agent_color.value and self.self_play:
            if self.opponent_model:
                action = self.opponent_model.predict(self.obs)[0]
            else:
                action = self.action_space.sample(np.array([1 if i < hand_count else 0 for i in range(4)], dtype=np.int8))

        action = min(hand_count - 1, action)
        reward = 0
        truncated = False
        if terminated := self.game.finished():
            winners = self.game.winners()
            if len(winners) == self.num_players:
                reward = 0
            elif self.agent_color in self.game.winners():
                reward = 1
            else:
                reward = -1
        else:
            try:
                self.game.play_card(action, [])
            except ValueError as e:
                print(action)
                print(e)
                reward = -1
                truncated, terminated = True, True

        observation = self._get_obs()

        if self.render_mode == "human":
            self.game.print()

        return observation, reward, terminated, truncated, {}
