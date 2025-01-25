from sys import stderr
from typing import Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from bar import Card, CardType, Color, Game


class BarEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        opponent_model=None,
        num_players: Literal[2, 3, 4] = 2,
        game_mode: Literal["basic", "medium", "full"] = "full",
        self_play: bool = True,
        render_mode=None,
    ):
        if game_mode != "basic":
            raise NotImplementedError

        self.game = Game(num_players=num_players, game_mode=game_mode)
        self.game_mode = game_mode
        self.num_players = num_players
        self.agent_color = Color(np.random.randint(0, num_players))
        self.opponent_model = opponent_model
        self.self_play = self_play

        self.cardt_count = len(CardType.basicList()) if game_mode != "full" else len(CardType.toList())
        self.card_dim = 1 if game_mode != "full" else 2
        self.observation_space = spaces.Box(
            low=-1,
            high=12,
            shape=(num_players * self.cardt_count + 5 * (self.card_dim + 1) + 4 * self.card_dim + 1,),
            dtype=np.int32,
        )
        self.action_space = spaces.Discrete(4)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _card_pos(self, c: Card):
        if self.game_mode == "basic":
            if c.value > 5:
                return c.value - 3
            elif c.value > 1:
                return c.value - 2
        return c.value

    def _get_obs(self):
        table_cards = (
            np.array([(c.value, c.color.value) if c else (-1, -1) for c in self.game.table_cards]).flatten()
            if self.game_mode != "full"
            else np.array(
                [(c.value, int(c.recursive), c.color.value) if c else (-1, -1, -1) for c in self.game.table_cards],
                dtype=np.int32,
            ).flatten()
        )

        hand = (
            [c.value for c in self.game.hands[self.game.turn]]
            if self.game_mode != "full"
            else np.array(
                [(c.value, int(c.recursive)) if c else (-1, -1) for c in self.game.hands[self.game.turn]],
                dtype=np.int32,
            ).flatten()
        )

        hand += [-1] * (4 - len(hand)) if self.game_mode != "full" else [(-1, -1)] * (4 - len(hand))
        visible_cards = np.array(np.append(table_cards, hand)).flatten()
        visible_cards = np.append(visible_cards, [self.game.turn])

        cards_pos = np.zeros((self.cardt_count * self.num_players,), dtype=np.int32)
        for c in self.game.heaven:
            v = self._card_pos(c)
            cards_pos[c.color.value * self.cardt_count + v - 1] = 1
        for c in self.game.hell:
            v = self._card_pos(c)
            cards_pos[c.color.value * self.cardt_count + v - 1] = -1

        return np.append(cards_pos, visible_cards)

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed, options=options)

        del self.game
        self.game = Game(self.num_players, self.game_mode)
        self.agent_color = Color(np.random.randint(0, self.num_players))

        if self.render_mode == "human":
            self.game.print()

        self.obs = self._get_obs()
        return self.obs, {}

    def step(self, action):
        assert self.action_space.contains(action)

        hand_count = len(self.game.hands[self.game.turn])
        if self.game.turn != self.agent_color.value and self.self_play:
            if self.opponent_model:
                action = self.opponent_model.predict(self.obs)[0]
                if action > hand_count - 1:
                    action = np.random.randint(0, hand_count)
            else:
                action = np.random.randint(0, hand_count)

        # action = min(hand_count - 1, action)
        truncated = False
        reward = 0
        if action > hand_count - 1:
            return self.obs, -1, True, True, {}
        else:
            self.game.play_card(action, [])

        if terminated := self.game.finished():
            winners = self.game.winners()
            if len(winners) == self.num_players:
                reward = 0
            elif self.agent_color in winners:
                reward = 1
            else:
                reward = -1

        self.obs = self._get_obs()

        if self.render_mode == "human":
            self.game.print()

        return self.obs, reward, terminated, truncated, {}
