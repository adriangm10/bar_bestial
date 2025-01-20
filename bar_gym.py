from typing import Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from bar import Game


class BarEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        num_players: Literal[2, 3, 4] = 2,
        game_mode: Literal["basic", "medium", "full"] = "full",
        render_mode=None,
    ):
        """
        observation space: (at the moment, with basic gamemode) a 9x3 matrix each row is a card
        that has (value, recursive, color), 5 for the table and 4 for the hand.
        action space: (for the basic gamemode) the card to be played Discrete(4)
        """
        self.game = Game(num_players=num_players, game_mode=game_mode)
        self.num_players = num_players
        self.game_mode = game_mode
        self.color = 0

        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]), high=np.array([12, 1, 3]), shape=(9, 3), dtype=np.int32
        )
        self.action_space = spaces.Discrete(4)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        cards = [(c.value, int(c.recursive), c.color.value) if c else (0, 0, 0) for c in self.game.table_cards]
        hand = [(c.value, int(c.recursive), c.color.value) for c in self.game.hands[self.game.turn]]
        hand += [(0, 0, 0)] * (4 - len(hand))

        return cards + hand

    def reset(self, seed=None, options=None):
        super().reset(seed, options)

        self.game = Game(self.num_players, self.game_mode)

        if self.render_mode == "human":
            self.game.print()

        return self._get_obs(), None

    def step(self, action):
        self.game.play_card(action, [])

        reward = 0
        if (terminated := self.game.finished()):
            if self.color in self.game.winners():
                reward = 1
            else:
                reward = -1

        observation = self._get_obs()

        if self.render_mode == "human":
            self.game.print()

        return observation, reward, terminated, False, None
