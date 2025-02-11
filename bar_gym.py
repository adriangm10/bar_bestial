import logging
from typing import Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from bar import Card, CardType, Color, Game

logger = logging.getLogger(__name__)


class BarEnv(gym.Env):
    """ """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        opponent_model=None,
        num_players: Literal[2, 3, 4] = 2,
        game_mode: Literal["basic", "medium", "full"] = "full",
        self_play: bool = False,
        render_mode=None,
        t: int = 2,
    ):
        self.game = Game(num_players=num_players, game_mode=game_mode)
        self.game_mode = game_mode
        self.num_players = num_players
        self.agent_color = Color(0)
        self.opponent_model = opponent_model
        self.self_play = self_play
        # self.agent_heaven = 0
        # self.opponent_heaven = 0

        self.L = 1 if game_mode == "basic" else 2  # hand (1), chosen card (1)
        self.M = 6  # heaven (1), hell (1), and queue (4)
        self.MM = num_players * 6
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(
                self.MM * t + self.L,  # (heaven + hell, board) * t, hand, color
                len(CardType.toList()),
            ),
            dtype=np.int32,
        )
        self.action_space = spaces.Discrete(4)

        self.t = t
        self.history: list[int] = []

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        cardt_count = len(CardType.toList())
        cards_rep = np.zeros(self.observation_space.shape, dtype=np.int32)

        # hand row
        for c in self.game.hands[self.game.turn]:
            cards_rep[0][c.value - 1] = 1

        # selected card
        if self.game_mode != "basic" and self.game.chosen_card:
            cardt = self.game.transformed_cardt if self.game.transformed_cardt else self.game.chosen_card.card_type
            cards_rep[1][cardt.value - 1] = 1

        # color row
        # for i in range(cardt_count):
        #     cards_rep[1][i] = self.game.turn

        # num_players rows for heaven
        for c in self.game.heaven:
            cards_rep[((c.color.value - self.game.turn) % self.num_players) * self.M + self.L][c.value - 1] = 1

        # num_players rows for hell
        for c in self.game.hell:
            cards_rep[((c.color.value - self.game.turn) % self.num_players) * self.M + self.L + 1][c.value - 1] = 1

        # 4 * num_players rows for the board, there will never be 5 cards in the queue
        for i, c in enumerate(self.game.table_cards):
            if c is None:
                break
            cards_rep[((c.color.value - self.game.turn) % self.num_players) * self.M + self.L + 2 + i][c.value - 1] = 1

        for i in range(1, min(self.t, len(self.history))):
            cards_rep[self.L + i * self.MM : self.L + (i + 1) * self.MM] = self.history[-i]

        if self.game.chosen_card is None:
            self.history.append(cards_rep[self.L : self.L + self.MM])

        return cards_rep

    def _predict_opp(self):
        assert self.game.turn != self.agent_color.value

        poss_actions = self.game.possible_actions()
        if self.opponent_model:
            act, _ = self.opponent_model.predict(self.obs)
            if act not in poss_actions:
                act = np.random.choice(poss_actions)
            return act
        else:
            return np.random.choice(poss_actions)

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed, options=options)

        del self.game, self.agent_color, self.history
        self.game = Game(self.num_players, self.game_mode)
        self.agent_color = Color(np.random.randint(0, self.num_players))
        self.history = []
        # self.agent_heaven = 0

        while self.self_play and self.game.turn != self.agent_color.value:
            act = self._predict_opp()
            self.game.play_card(act)
            self.obs = self._get_obs()

        if self.render_mode == "human":
            self.game.print()

        self.obs = self._get_obs()
        return self.obs, {}

    def step(self, action):
        assert self.action_space.contains(action)
        if self.self_play:
            assert self.game.turn == self.agent_color.value

        poss_actions = self.game.possible_actions()
        reward = 0
        if action not in poss_actions and self.game.turn == self.agent_color.value:
            reward = -0.1
            logger.debug(f"the agent selected an invalid action: {action}")
            action = np.random.choice(poss_actions)

        self.game.play_card(action)

        while self.self_play and self.game.turn != self.agent_color.value and not self.game.finished():
            act = self._predict_opp()
            self.game.play_card(act)
            self.obs = self._get_obs()

        # agent_heaven = len([c for c in self.game.heaven if c.color == self.agent_color])
        # if agent_heaven > self.agent_heaven:
        #     reward += agent_heaven - self.agent_heaven
        #     self.agent_heaven = agent_heaven

        # done = self.game.finished()
        if done := self.game.finished():
            winners = self.game.winners()
            if len(winners) == self.num_players or not winners:
                reward += 0
            elif self.agent_color in winners:
                reward += 1
            else:
                reward += -1

        self.obs = self._get_obs()

        if self.render_mode == "human":
            self.game.print()

        return self.obs, reward, done, False, {}
