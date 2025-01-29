from typing import Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from bar import Card, CardType, Color, Game


class BarEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}
    HEAVEN = 5
    HELL = 6
    HAND = 7

    def __init__(
        self,
        opponent_model=None,
        num_players: Literal[2, 3, 4] = 2,
        game_mode: Literal["basic", "medium", "full"] = "full",
        self_play: bool = False,
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

        self.observation_space = spaces.Box(
            low=0,
            high=7,
            shape=(num_players, len(CardType)),
            dtype=np.int32,
        )
        self.action_space = spaces.Discrete(4)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        cardt_count = len(CardType.toList())
        cards_rep = np.zeros(self.observation_space.shape, dtype=np.int32)
        for c in self.game.heaven:
            cards_rep[c.color.value][c.value - 1] = self.HEAVEN
        for c in self.game.hell:
            cards_rep[c.color.value][c.value - 1] = self.HELL
        for i, c in enumerate(self.game.table_cards):
            if c is None:
                break
            cards_rep[c.color.value][c.value - 1] = i + 1
        for c in self.game.hands[self.game.turn]:
            cards_rep[c.color.value][c.value - 1] = self.HAND

        return cards_rep

    def _predict_opp(self):
        hand_count = len(self.game.hands[self.game.turn])
        if self.opponent_model:
            act, _ = self.opponent_model.predict(self.obs)
            if act > hand_count - 1:
                act = np.random.randint(0, hand_count)
            return act
        else:
            return np.random.randint(0, hand_count)

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed, options=options)

        del self.game, self.agent_color
        self.game = Game(self.num_players, self.game_mode)
        self.agent_color = Color(np.random.randint(0, self.num_players))

        while self.self_play and self.game.turn != self.agent_color.value:
            act = self._predict_opp()
            self.game.play_card(act, [])
            self.obs = self._get_obs()

        if self.render_mode == "human":
            self.game.print()

        self.obs = self._get_obs()
        return self.obs, {}

    def step(self, action):
        assert self.action_space.contains(action)
        if self.self_play:
            assert self.game.turn == self.agent_color.value

        hand_count = len(self.game.hands[self.game.turn])
        reward = 0
        if action > hand_count - 1:
            reward = -0.1
            action = np.random.randint(0, hand_count)

        self.game.play_card(action, [])

        while self.self_play and self.game.turn != self.agent_color.value and not self.game.finished():
            act = self._predict_opp()
            self.game.play_card(act, [])
            self.obs = self._get_obs()

        if done := self.game.finished():
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

        return self.obs, reward, done, False, {}
