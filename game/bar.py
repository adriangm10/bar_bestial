# mypy: disable-error-code="empty-body,union-attr"
import logging
import unittest
from collections.abc import Callable
from enum import Enum
from functools import reduce
from random import randint, sample
from typing import Annotated, Literal

from termcolor import colored

CARD_WIDTH = 15
QUEUE_LEN = 5
logger = logging.getLogger(__name__)

type TableCards = Annotated[list[Card | None], QUEUE_LEN]
type Actions = list[int]
type Hell = list[Card]
type CardFunction = Callable[[int, TableCards, Hell, Actions], tuple[TableCards, Hell]]


class Color(Enum):
    YELLOW = 0
    BLUE = 1
    RED = 2
    GREEN = 3


class CardType(Enum):
    LEON = 12
    HIPOPOTAMO = 11
    COCODRILO = 10
    SERPIENTE = 9
    JIRAFA = 8
    CEBRA = 7
    FOCA = 6
    CAMALEON = 5
    MONO = 4
    CANGURO = 3
    LORO = 2
    MOFETA = 1

    @classmethod
    def toList(cls):
        return [c for c in cls]

    @classmethod
    def basicList(cls):
        return [c for c in cls if c != cls.CAMALEON and c != cls.CANGURO and c != cls.LORO]

    @classmethod
    def leon_action(
        cls,
        card_pos: int,
        cards: TableCards,
        hell: Hell,
        actions: Actions,
    ) -> tuple[TableCards, Hell]:
        if cards[card_pos] is None:
            logger.error("leon action: the card in cards[card_pos] is none")
            raise ValueError("cards[card_pos] is None")

        card: Card = cards[card_pos]  # type: ignore[assignment]
        cards[card_pos] = None

        if any([c.card_type == cls.LEON for c in cards[:card_pos]]):
            hell.append(card)
            cards[card_pos] = None
            logger.info(f"{card} has gone directly to hell because there was another leon")
            return cards, hell

        for i in reversed(range(card_pos)):
            c: Card = cards[i]  # type: ignore[assignment]
            if c.card_type == CardType.MONO:
                hell.append(c)
                logger.info(f"the leon sends {c} to hell")
                for j in range(i, card_pos):
                    cards[j] = cards[j + 1]

        for i in reversed(range(card_pos)):
            cards[i + 1] = cards[i]

        cards[0] = card
        logger.info(f"the {card} puts itself at the first position")

        return cards, hell

    @classmethod
    def hipopotamo_action(
        cls,
        card_pos: int,
        cards: TableCards,
        hell: Hell,
        actions: Actions,
    ) -> tuple[TableCards, Hell]:
        if cards[card_pos] is None:
            logger.error("hipopotamo_action: the card in cards[card_pos] is none")
            raise ValueError("cards[card_pos] is None")

        card = cards[card_pos]
        cards[card_pos] = None

        i = 0
        for i, c in reversed(list(enumerate(cards[:card_pos]))):
            if c.card_type >= cls.HIPOPOTAMO or c.card_type == cls.CEBRA:
                i += 1
                break

        for j in reversed(range(i, card_pos)):
            cards[j + 1] = cards[j]

        cards[i] = card
        logger.info(f"hipopotamo_action: the {card} tackles the animals and gets itself to the {i + 1} position")  # fmt: skip
        return cards, hell

    @classmethod
    def cocodrilo_action(
        cls,
        card_pos: int,
        cards: TableCards,
        hell: Hell,
        actions: Actions,
    ) -> tuple[TableCards, Hell]:
        if cards[card_pos] is None:
            logger.error("cocodrilo_action: the card in cards[card_pos] is none")
            raise ValueError("cards[card_pos] is None")

        card = cards[card_pos]
        cards[card_pos] = None

        i = 0
        for i, c in reversed(list(enumerate(cards[:card_pos]))):
            if c.card_type >= cls.COCODRILO or c.card_type == cls.CEBRA:
                i += 1
                break
            else:
                hell.append(c)  # type: ignore[arg-type]
                cards[i] = None

        dist = card_pos - i
        if dist > 0:
            for j in range(i + 1, QUEUE_LEN - dist):
                cards[j] = cards[j + dist]
                cards[j + dist] = None

        cards[i] = card
        logger.info(f"cocodrilo_action: the {card} sends all the animals in the way to hell and puts itself at the {i + 1} position")  # fmt: skip
        return cards, hell

    @classmethod
    def serpiente_action(
        cls,
        card_pos: int,
        cards: TableCards,
        hell: Hell,
        actions: Actions,
    ) -> tuple[TableCards, Hell]:
        if cards[card_pos] is None:
            logger.error("serpiente_action: the card in cards[card_pos] is none")
            raise ValueError("cards[card_pos] is None")
        logger.info(f"serpiente_action: the {cards[card_pos]} orders the queue based on the force of each animal")
        return sorted(cards[: card_pos + 1], reverse=True) + [None] * (QUEUE_LEN - card_pos - 1), hell  # type: ignore[type-var]

    @classmethod
    def jirafa_action(
        cls,
        card_pos: int,
        cards: TableCards,
        hell: Hell,
        actions: Actions,
    ) -> tuple[TableCards, Hell]:
        if cards[card_pos] is None:
            logger.error("jirafa_action: the card in cards[card_pos] is none")
            raise ValueError("cards[card_pos] is None")
        if card_pos > 0 and cards[card_pos - 1] < cards[card_pos]:  # type: ignore[operator]
            logger.info(f"jirafa_action: the {cards[card_pos]} passes above the {cards[card_pos - 1]}")
            cards[card_pos - 1], cards[card_pos] = cards[card_pos], cards[card_pos - 1]

        return cards, hell

    @classmethod
    def foca_action(
        cls,
        card_pos: int,
        cards: TableCards,
        hell: Hell,
        actions: Actions,
    ) -> tuple[TableCards, Hell]:
        if cards[card_pos] is None:
            logger.error("foca_action: the card in cards[card_pos] is none")
            raise ValueError("cards[card_pos] is None")
        # instead of changing the gates, the queue is inverted
        logger.info("foca_action: the foca inverts the queue")
        return cards[card_pos::-1] + [None] * (QUEUE_LEN - card_pos - 1), hell

    @classmethod
    def camaleon_action(
        cls,
        card_pos: int,
        cards: TableCards,
        hell: Hell,
        actions: Actions,
    ) -> tuple[TableCards, Hell]:
        if all([c.card_type == cls.CAMALEON for c in cards[:card_pos]]):
            logger.info("camaleon_action: there is no card to copy in the queue so the camaleon does nothing")
            return cards, hell
        if cards[card_pos] is None:
            logger.error("camaleon_action: the card in cards[card_pos] is none")
            raise ValueError("cards[card_pos] is None")
        if not actions or cards[actions[0]] is None or cards[actions[0]].card_type == cls.CAMALEON:
            logger.error(f"camaleon_action: invalid actions ({actions}) for a camaleon")
            raise ValueError(f"invalid action {actions} for a camaleon")

        f = cards[actions[0]].card_type.action()
        logger.info(f"camaleon_action: the {cards[card_pos]} copies {cards[actions[0]]} with actions {actions}")
        return f(card_pos, cards, hell, actions[1:])

    @classmethod
    def mono_action(
        cls,
        card_pos: int,
        cards: TableCards,
        hell: Hell,
        actions: Actions,
    ) -> tuple[TableCards, Hell]:
        if cards[card_pos] is None:
            logger.error("mono_action: the card in cards[card_pos] is none")
            raise ValueError("cards[card_pos] is None")

        if any([c.card_type == cls.MONO for c in cards[:card_pos]]):
            new_board = []
            logger.info("mono_action: there is another mono in the queue")
            for c in cards[:card_pos]:
                if c.card_type == cls.COCODRILO or c.card_type == cls.HIPOPOTAMO:
                    logger.info(f"mono_action: the monos send the {c} to hell")
                    hell.append(c)  # type: ignore[arg-type]
                else:
                    new_board.append(c)

            # fmt: off
            new_board = [cards[card_pos]] + [c for c in reversed(new_board) if c.card_type == cls.MONO] + list(
                filter(lambda c: c.card_type != cls.MONO, new_board)  # type: ignore[arg-type]
            )
            cards = new_board + [None] * (QUEUE_LEN - len(new_board))
            logger.info("mono_action: the monos put themselves on the beginning of the queue")

        return cards, hell

    @classmethod
    def canguro_action(
        cls,
        card_pos: int,
        cards: TableCards,
        hell: Hell,
        actions: Actions,
    ) -> tuple[TableCards, Hell]:
        if card_pos == 0:
            return cards, hell
        if cards[card_pos] is None:
            logger.error("canguro_action: the card in cards[card_pos] is none")
            raise ValueError("cards[card_pos] is None")
        if not actions or not 0 < actions[0] <= 2:
            logger.error(f"canguro_action: the action {actions} is invalid")
            raise ValueError("Invalid action for canguro")

        new_pos = max(card_pos - actions[0], 0)
        card = cards[card_pos]
        logger.info(f"canguro_action: the {card} jumps {actions[0]} positions")
        for i in reversed(range(card_pos - actions[0], card_pos)):
            cards[i + 1] = cards[i]
        cards[new_pos] = card

        return cards, hell

    @classmethod
    def loro_action(
        cls,
        card_pos: int,
        cards: TableCards,
        hell: Hell,
        actions: Actions,
    ) -> tuple[TableCards, Hell]:
        if card_pos == 0:
            return cards, hell
        if cards[card_pos] is None:
            logger.error("loro_action: the card in cards[card_pos] is none")
            raise ValueError("cards[card_pos] is None")
        if not actions or cards[actions[0]] is None:
            logger.error(f"loro_action: the action {actions} is invalid")
            raise ValueError(f"invalid action {actions} for a loro")

        logger.info(f"loro_action: the {cards[card_pos]} sends {cards[actions[0]]} to hell")
        hell.append(cards[actions[0]])  # type: ignore[arg-type]
        for i in range(actions[0], card_pos):
            cards[i] = cards[i + 1]

        cards[card_pos] = None
        return cards, hell

    @classmethod
    def mofeta_action(
        cls,
        card_pos: int,
        cards: TableCards,
        hell: Hell,
        actions: Actions,
    ) -> tuple[TableCards, Hell]:
        if cards[card_pos] is None:
            logger.error("mofeta_action: the card in cards[card_pos] is none")
            raise ValueError("cards[card_pos] is None")

        top1, top2 = 0, 0
        for c in cards[:card_pos]:
            if c.value > top1:
                top1, top2 = c.value, top1
            elif top1 > c.value > top2:
                top2 = c.value

        if top1 <= 1 and top2 <= 1:
            return cards, hell
        elif top1 > 1 and top2 == 0:
            top2 = top1

        logger.info(f"mofeta_action: the {cards[card_pos]} sends {cls(top1).name}s and {cls(top2).name}s to hell")
        filtered_cards = [c for c in cards[:card_pos] if c.value < top2 or c.value == CardType.MOFETA.value]
        hell_cards = [c for c in cards[:card_pos] if c.value >= top2 and c.value != CardType.MOFETA.value]
        cards = filtered_cards + [cards[card_pos]] + [None] * (QUEUE_LEN - len(filtered_cards) - 1)

        return cards, hell + hell_cards  # type: ignore[operator]

    def action(self) -> CardFunction:
        match self:
            case CardType.LEON:
                return CardType.leon_action
            case CardType.HIPOPOTAMO:
                return CardType.hipopotamo_action
            case CardType.COCODRILO:
                return CardType.cocodrilo_action
            case CardType.SERPIENTE:
                return CardType.serpiente_action
            case CardType.JIRAFA:
                return CardType.jirafa_action
            case CardType.CEBRA:
                return lambda cpos, cs, hell, acts: (cs, hell)
            case CardType.FOCA:
                return CardType.foca_action
            case CardType.CAMALEON:
                return CardType.camaleon_action
            case CardType.MONO:
                return CardType.mono_action
            case CardType.CANGURO:
                return CardType.canguro_action
            case CardType.LORO:
                return CardType.loro_action
            case CardType.MOFETA:
                return CardType.mofeta_action

    def is_recursive(self) -> bool:
        match self:
            case CardType.HIPOPOTAMO | CardType.COCODRILO | CardType.JIRAFA | CardType.CEBRA:
                return True
            case _:
                return False

    def action_msg(self) -> str | None:
        match self:
            case CardType.CAMALEON:
                return "Select a card to copy[0-]"
            case CardType.CANGURO:
                return "Select jump length[1|2]"
            case CardType.LORO:
                return "Select a card to scare away[0-]"
            case _:
                return None

    def has_options(self) -> bool:
        match self:
            case CardType.CAMALEON | CardType.CANGURO | CardType.LORO:
                return True
            case _:
                return False

    def possible_options(self, queue: TableCards) -> list[int]:
        match self:
            case CardType.LORO:
                return [i for i, c in enumerate(queue) if c]
            case CardType.CAMALEON:
                return [i for i, c in enumerate(queue) if c and c.card_type != CardType.CAMALEON]
            case CardType.CANGURO:
                return [1, 2]
            case _:
                return []

    def __ge__(self, o) -> bool:
        if self.__class__ == o.__class__:
            return self.value >= o.value
        raise ValueError("Error comparing " + str(self.__class__) + " with " + str(o.__class__))

    def __lt__(self, o) -> bool:
        if self.__class__ == o.__class__:
            return self.value < o.value
        raise ValueError("Error comparing " + str(self.__class__) + " with " + str(o.__class__))


class Card:
    def __init__(self, ctype: CardType, color: Color):
        self.card_type = ctype
        self.name = ctype.name
        self.value = ctype.value
        self.recursive = ctype.is_recursive()
        self.color = color

    def format(self) -> list[str]:
        recursive_symbol = "⟲"
        top_border = "╔{}╗".format("═" * (CARD_WIDTH - 2))
        bot_border = "╚{}╝".format("═" * (CARD_WIDTH - 2))
        header = "║{:02}{}{:02}║".format(
            self.value,
            " " * (CARD_WIDTH - 6),
            self.value,
        )
        body1 = "║{}{}{} ║".format(
            recursive_symbol if self.recursive else " ",
            " " * (CARD_WIDTH - 5),
            recursive_symbol if self.recursive else " ",
        )
        body2 = ("║{:^" + str(CARD_WIDTH - 2) + "}║").format(self.name)

        color = self.color.name.lower()

        return [
            colored(top_border, color),  # type: ignore[arg-type]
            colored(header, color),  # type: ignore[arg-type]
            colored(body1, color),  # type: ignore[arg-type]
            colored(body2, color),  # type: ignore[arg-type]
            colored(bot_border, color),  # type: ignore[arg-type]
        ]

    def action(self) -> CardFunction:
        return self.card_type.action()

    def action_msg(self) -> str | None:
        return self.card_type.action_msg()

    def has_options(self) -> bool:
        return self.card_type.has_options()

    def __eq__(self, o) -> bool:
        if self.__class__ == o.__class__:
            return self.card_type == o.card_type and self.color == o.color
        raise ValueError("Error comparing " + str(self.__class__) + " with " + str(o.__class__))

    def __lt__(self, o):
        if self.__class__ == o.__class__:
            return self.card_type < o.card_type
        raise ValueError("Error comparing " + str(self.__class__) + " with " + str(o.__class__))

    def __str__(self):
        return "<" + self.card_type.name + ", " + self.color.name + ">"

    def __repr__(self):
        return "<" + self.card_type.name + ", " + self.color.name + ">"


def blank_card() -> list[str]:
    top_border = "╔{}╗".format("═" * (CARD_WIDTH - 2))
    bot_border = "╚{}╝".format("═" * (CARD_WIDTH - 2))
    body = "║" + " " * (CARD_WIDTH - 2) + "║"

    return [top_border, body, body, body, bot_border]


def format_cards(hand: list[Card | None]) -> list[str]:
    handr = map(lambda c: c.format() if c else blank_card(), hand)
    return reduce(
        lambda acc, card: (card if not acc else list(map(lambda x: x[1] + card[x[0]], enumerate(acc)))),
        handr,
        [],
    )


class Game:
    def __init__(
        self,
        num_players: Literal[2, 3, 4] = 2,
        game_mode: Literal["basic", "medium", "full"] = "full",
    ):
        if not 2 <= num_players <= 4:
            raise ValueError("There must be between 2 and 4 players")
        assert game_mode in ["basic", "medium", "full"]

        self.num_players = num_players
        self.table_cards: TableCards = [None] * QUEUE_LEN
        self.hell: list[Card] = []
        self.heaven: list[Card] = []

        self.game_mode = game_mode
        cardType_list = CardType.basicList() if game_mode == "basic" else CardType.toList()
        cardType_count = len(cardType_list)
        self.decks = [[Card(ct, Color(c)) for ct in sample(cardType_list, cardType_count)] for c in range(num_players)]

        self.hands = [sorted(d[:4]) for d in self.decks]
        self.decks = [d[4:] for d in self.decks]

        self.turn = randint(0, num_players - 1)

        self.chosen_card: Card | None = None
        self.transformed_cardt: CardType | None = None
        self.actions: list[int] = []

    def print(self):
        hand_rep = format_cards(self.hands[self.turn])
        table_rep = format_cards(self.table_cards)

        print(f"{"\n".join(table_rep)}\n\n")
        print(f"{"\n".join(hand_rep)}\n\n")

    def possible_actions(self) -> list[int]:
        if self.chosen_card:
            cardt = self.transformed_cardt if self.transformed_cardt else self.chosen_card.card_type
            return cardt.possible_options(self.table_cards)
        else:
            return list(range(len(self.hands[self.turn])))

    def action_msg(self) -> str | None:
        if self.chosen_card:
            return self.transformed_cardt.action_msg() if self.transformed_cardt else self.chosen_card.action_msg()
        else:
            return f"Select a card [0-{len(self.hands[self.turn]) - 1}]"

    def play_card(self, action: int):
        if self.chosen_card is None:
            self.chosen_card = self.hands[self.turn].pop(action)
            logger.info(f"[GAME]: {Color(self.turn).name} player plays {self.chosen_card}")

            if self.chosen_card.has_options() and self.possible_actions() and self.table_cards[0] is not None:
                logger.info(f"[GAME]: {Color(self.turn).name} player has to input card action")
                return
        else:
            self.actions.append(action)
            if (
                self.chosen_card.card_type == CardType.CAMALEON
                and not self.transformed_cardt
                and self.table_cards[action].has_options()
            ):
                self.transformed_cardt = self.table_cards[action].card_type
                logger.info(
                    f"[GAME]: the {self.chosen_card} transforms to {self.transformed_cardt} and the player has to select its action"
                )
                return

        for i, c in enumerate(self.table_cards):
            if c is None:
                self.table_cards[i] = self.chosen_card
                card_pos = i
                break

        # execute card
        act = self.chosen_card.action()
        try:
            self.table_cards, self.hell = act(card_pos, self.table_cards, self.hell, self.actions)
        except ValueError as e:
            self.hands[self.turn].append(self.chosen_card)
            self.hands[self.turn].sort()
            self.table_cards[card_pos] = None
            self.chosen_card = None
            self.transformed_cardt = None
            self.actions = []
            raise e

        # execute recurrent cards
        if self.game_mode == "full":
            for i, c in enumerate(self.table_cards):
                if c and self.chosen_card != c and c.recursive:
                    self.table_cards, self.hell = c.action()(i, self.table_cards, self.hell, [])

        # open heaven and hell doors
        if all([c is not None for c in self.table_cards]):
            logger.info(f"[GAME]: {self.table_cards[0]} and {self.table_cards[1]} enter in heaven")
            logger.info(f"[GAME]: {self.table_cards[-1]} goes to hell")
            self.heaven.extend([self.table_cards[0], self.table_cards[1]])  # type: ignore[arg-type]
            self.hell.append(self.table_cards[4])  # type: ignore[arg-type]
            self.table_cards[0], self.table_cards[1] = self.table_cards[2], self.table_cards[3]
            self.table_cards[2], self.table_cards[3], self.table_cards[4] = None, None, None

        # draw a card
        if self.decks[self.turn]:
            self.hands[self.turn].append(self.decks[self.turn].pop(0))
            self.hands[self.turn].sort()

        # next turn
        self.turn = (self.turn + 1) % self.num_players

        self.transformed_cardt = None
        self.chosen_card = None
        self.actions = []

    def finished(self) -> bool:
        return all([not d and not h for d, h in zip(self.decks, self.hands)])

    def winners(self) -> list[Color]:
        players_heaven = [[c for c in self.heaven if c.color.value == col] for col in range(self.num_players)]
        players_heaven = [cs for cs in players_heaven if len(cs) > 0]

        max = 0
        winners_heaven = []
        for cards in players_heaven:
            if len(cards) > max:
                max = len(cards)
                winners_heaven = [cards]
            elif len(cards) == max:
                winners_heaven.append(cards)

        if len(winners_heaven) == 1:
            return [winners_heaven[0][0].color]

        min = 999
        winners = []
        for w in winners_heaven:
            if (s := sum([c.value for c in w])) < min:
                min = s
                winners = [w[0].color]
            elif (s := sum([c.value for c in w])) == min:
                winners.append(w[0].color)

        return winners

    def hand_card(self, pos: int) -> Card:
        return self.hands[self.turn][pos]


class TestGame(unittest.TestCase):
    def setUp(self):
        self.game = Game()

    def test_play_card(self):
        turn = self.game.turn
        card = Card(CardType.MONO, Color(0))
        deck = self.game.decks[turn].copy()
        self.game.hands[turn][0] = card
        self.game.play_card(0)

        self.assertEqual(self.game.table_cards[0], card)
        self.assertEqual(len(self.game.hands[turn]), 4)
        self.assertEqual(deck[1:], self.game.decks[turn])

    def test_play_card_empty_deck(self):
        turn = self.game.turn
        card = Card(CardType.MONO, Color(0))
        self.game.hands[turn][0] = card
        self.game.decks[turn] = []
        self.game.play_card(0)

        self.assertEqual(self.game.table_cards[0], card)
        self.assertEqual(len(self.game.hands[turn]), 3)

    def test_play_card_open_doors(self):
        table_cards = [
            Card(CardType.LEON, Color.GREEN),
            Card(CardType.HIPOPOTAMO, Color.YELLOW),
            Card(CardType.COCODRILO, Color.GREEN),
            Card(CardType.COCODRILO, Color.YELLOW),
        ]
        self.game.table_cards[:4] = table_cards
        hell_card = Card(CardType.MONO, Color.GREEN)
        self.game.hands[self.game.turn][0] = hell_card
        self.game.play_card(0)

        self.assertEqual(self.game.heaven, table_cards[:2])
        self.assertEqual(self.game.hell, [hell_card])
        self.assertEqual(self.game.table_cards, table_cards[2:] + [None, None, None])

    def test_game_finished(self):
        self.assertFalse(self.game.finished())
        self.game.decks = [[] for _ in self.game.decks]
        self.game.hands = [[] for _ in self.game.hands]
        self.assertTrue(self.game.finished())

    def test_game_winners_one_winner(self):
        self.game.heaven = [
            Card(CardType.LEON, Color(1)),
            Card(CardType.COCODRILO, Color(1)),
            Card(CardType.MOFETA, Color(1)),
            Card(CardType.JIRAFA, Color(1)),
            Card(CardType.LEON, Color(0)),
            Card(CardType.HIPOPOTAMO, Color(0)),
        ]
        self.assertEqual(self.game.winners(), [Color(1)])

    def test_game_winners_draw(self):
        self.game.heaven = [
            Card(CardType.LEON, Color(1)),
            Card(CardType.COCODRILO, Color(1)),
            Card(CardType.MOFETA, Color(1)),
            Card(CardType.JIRAFA, Color(1)),
            Card(CardType.LEON, Color(0)),
            Card(CardType.COCODRILO, Color(0)),
            Card(CardType.MOFETA, Color(0)),
            Card(CardType.JIRAFA, Color(0)),
        ]
        self.assertEqual(self.game.winners(), [Color(0), Color(1)])

    def test_game_winners_draw_in_num_of_cards(self):
        self.game.heaven = [
            Card(CardType.LEON, Color(1)),
            Card(CardType.COCODRILO, Color(1)),
            Card(CardType.MOFETA, Color(1)),
            Card(CardType.JIRAFA, Color(1)),
            Card(CardType.LEON, Color(0)),
            Card(CardType.MONO, Color(0)),
            Card(CardType.MOFETA, Color(0)),
            Card(CardType.JIRAFA, Color(0)),
        ]
        self.assertEqual(self.game.winners(), [Color(0)])


class TestCardActions(unittest.TestCase):
    def setUp(self):
        self.hell = []
        self.heaven = []

    def test_lion_action_no_monkeys(self):
        table_cards = [
            Card(CardType.HIPOPOTAMO, Color.YELLOW),
            Card(CardType.COCODRILO, Color.GREEN),
            Card(CardType.COCODRILO, Color.YELLOW),
            None,
            None,
        ]
        table_cards[3] = Card(CardType.LEON, Color.GREEN)
        cards, hell = CardType.leon_action(3, table_cards.copy(), self.hell, [])

        self.assertEqual(cards, [Card(CardType.LEON, Color.GREEN)] + table_cards[:3] + [None])

    def test_lion_action_monkeys(self):
        table_cards = [
            Card(CardType.HIPOPOTAMO, Color.YELLOW),
            Card(CardType.MONO, Color.GREEN),
            Card(CardType.COCODRILO, Color.YELLOW),
            Card(CardType.MONO, Color.YELLOW),
            None,
        ]
        table_cards[4] = Card(CardType.LEON, Color.GREEN)
        cards, hell = CardType.leon_action(4, table_cards.copy(), self.hell, [])

        # fmt: off
        self.assertEqual(cards, [
            table_cards[4],
            table_cards[0],
            table_cards[2],
            None,
            None,
        ])  # leon hipopotamo cocodrilo
        self.assertEqual(hell, [table_cards[3], table_cards[1]])

    def test_lion_action_monkeys2(self):
        table_cards = [
            Card(CardType.HIPOPOTAMO, Color.YELLOW),
            Card(CardType.MONO, Color.GREEN),
            Card(CardType.MONO, Color.YELLOW),
            Card(CardType.COCODRILO, Color.YELLOW),
            Card(CardType.LEON, Color.GREEN),
        ]
        cards, hell = CardType.leon_action(4, table_cards.copy(), self.hell, [])

        # fmt: off
        self.assertEqual(cards, [
            table_cards[4],
            table_cards[0],
            table_cards[3],
            None,
            None,
        ])  # leon hipopotamo cocodrilo
        self.assertEqual(hell, [table_cards[2], table_cards[1]])

    def test_lion_action_other_lion(self):
        table_cards = [
            Card(CardType.LEON, Color.GREEN),
            Card(CardType.HIPOPOTAMO, Color.YELLOW),
            Card(CardType.COCODRILO, Color.YELLOW),
            Card(CardType.MONO, Color.GREEN),
            None,
        ]
        table_cards[3] = Card(CardType.LEON, Color.YELLOW)
        cards, hell = CardType.leon_action(3, table_cards.copy(), self.hell, [])

        self.assertEqual(
            cards,
            table_cards[:3] + [None, None],
        )
        self.assertEqual(hell, [table_cards[3]])

    def test_hipopotamo_action(self):
        table_cards = [
            Card(CardType.LEON, Color.GREEN),
            Card(CardType.COCODRILO, Color.YELLOW),
            Card(CardType.MONO, Color.GREEN),
            None,
            None,
        ]
        table_cards[3] = Card(CardType.HIPOPOTAMO, Color.YELLOW)
        cards, hell = CardType.hipopotamo_action(3, table_cards.copy(), self.hell, [])
        table_cards[3] = None

        self.assertEqual(
            cards,
            [table_cards[0]] + [Card(CardType.HIPOPOTAMO, Color.YELLOW)] + table_cards[1:-1],
        )

    def test_hipopotamo_middle(self):
        table_cards = [
            Card(CardType.COCODRILO, Color.GREEN),
            Card(CardType.JIRAFA, Color.GREEN),
            Card(CardType.HIPOPOTAMO, Color.YELLOW),
            Card(CardType.COCODRILO, Color.YELLOW),
            None,
        ]
        cards, hell = CardType.hipopotamo_action(2, table_cards.copy(), self.hell, [])

        # fmt: off
        self.assertEqual(cards, [
            Card(CardType.HIPOPOTAMO, Color.YELLOW),
            Card(CardType.COCODRILO, Color.GREEN),
            Card(CardType.JIRAFA, Color.GREEN),
            Card(CardType.COCODRILO, Color.YELLOW),
            None,
        ])

    def test_cocodrilo_action(self):
        table_cards = [
            Card(CardType.LEON, Color.GREEN),
            Card(CardType.CAMALEON, Color.YELLOW),
            Card(CardType.MONO, Color.GREEN),
            None,
            None,
        ]
        table_cards[3] = Card(CardType.COCODRILO, Color.YELLOW)
        cards, hell = CardType.cocodrilo_action(3, table_cards.copy(), self.hell, [])
        table_cards[3] = None

        self.assertEqual(cards, [table_cards[0], Card(CardType.COCODRILO, Color.YELLOW)] + [None] * 3)
        self.assertEqual(
            hell,
            [Card(CardType.MONO, Color.GREEN), Card(CardType.CAMALEON, Color.YELLOW)],
        )

    def test_cocodrilo_middle(self):
        table_cards = [
            Card(CardType.CAMALEON, Color.GREEN),
            Card(CardType.JIRAFA, Color.GREEN),
            Card(CardType.COCODRILO, Color.YELLOW),
            Card(CardType.MONO, Color.YELLOW),
            None,
        ]
        cards, hell = CardType.cocodrilo_action(2, table_cards.copy(), self.hell, [])

        # fmt: off
        self.assertEqual(cards, [
            Card(CardType.COCODRILO, Color.YELLOW),
            Card(CardType.MONO, Color.YELLOW),
        ] + [None] * 3)
        self.assertEqual(hell, [
            Card(CardType.JIRAFA, Color.GREEN),
            Card(CardType.CAMALEON, Color.GREEN),
        ])

    def test_serpiente(self):
        table_cards = [
            Card(CardType.CAMALEON, Color.GREEN),
            Card(CardType.JIRAFA, Color.GREEN),
            Card(CardType.COCODRILO, Color.YELLOW),
            Card(CardType.SERPIENTE, Color.YELLOW),
            None,
        ]
        cards, hell = CardType.serpiente_action(3, table_cards.copy(), self.hell, [])

        self.assertEqual(
            cards,
            [table_cards[2], table_cards[3], table_cards[1], table_cards[0], None],
        )

    def test_jirafa(self):
        table_cards = [
            Card(CardType.CAMALEON, Color.GREEN),
            Card(CardType.MONO, Color.YELLOW),
            Card(CardType.JIRAFA, Color.GREEN),
            Card(CardType.SERPIENTE, Color.YELLOW),
            None,
        ]
        cards, hell = CardType.jirafa_action(2, table_cards.copy(), self.hell, [])

        # fmt: off
        self.assertEqual(cards, [
            Card(CardType.CAMALEON, Color.GREEN),
            Card(CardType.JIRAFA, Color.GREEN),
            Card(CardType.MONO, Color.YELLOW),
            Card(CardType.SERPIENTE, Color.YELLOW),
            None,
        ])

    def test_foca(self):
        table_cards = [
            Card(CardType.CAMALEON, Color.GREEN),
            Card(CardType.MONO, Color.YELLOW),
            Card(CardType.JIRAFA, Color.GREEN),
            Card(CardType.FOCA, Color.YELLOW),
            None,
        ]
        cards, hell = CardType.foca_action(3, table_cards.copy(), self.hell, [])

        # fmt: off
        self.assertEqual(cards, [
            Card(CardType.FOCA, Color.YELLOW),
            Card(CardType.JIRAFA, Color.GREEN),
            Card(CardType.MONO, Color.YELLOW),
            Card(CardType.CAMALEON, Color.GREEN),
            None,
        ])

    def test_camaleon(self):
        table_cards = [
            Card(CardType.HIPOPOTAMO, Color.YELLOW),
            Card(CardType.JIRAFA, Color.GREEN),
            Card(CardType.FOCA, Color.YELLOW),
            Card(CardType.CAMALEON, Color.GREEN),
            None,
        ]
        cards, hell = CardType.camaleon_action(3, table_cards.copy(), self.hell, [0])

        # fmt: off
        self.assertEqual(cards, [
            Card(CardType.HIPOPOTAMO, Color.YELLOW),
            Card(CardType.CAMALEON, Color.GREEN),
            Card(CardType.JIRAFA, Color.GREEN),
            Card(CardType.FOCA, Color.YELLOW),
            None
        ])

    def test_mono_nothing(self):
        table_cards = [
            Card(CardType.HIPOPOTAMO, Color.YELLOW),
            Card(CardType.JIRAFA, Color.GREEN),
            Card(CardType.FOCA, Color.YELLOW),
            Card(CardType.MONO, Color.GREEN),
            None,
        ]

        cards, hell = CardType.mono_action(3, table_cards.copy(), self.hell, [])

        self.assertEqual(cards, table_cards)

    def test_mono_2(self):
        table_cards = [
            Card(CardType.HIPOPOTAMO, Color.YELLOW),
            Card(CardType.JIRAFA, Color.GREEN),
            Card(CardType.MONO, Color.YELLOW),
            Card(CardType.MONO, Color.GREEN),
            None,
        ]

        cards, hell = CardType.mono_action(3, table_cards.copy(), self.hell, [])

        # fmt: off
        self.assertEqual(cards, [
            Card(CardType.MONO, Color.GREEN),
            Card(CardType.MONO, Color.YELLOW),
            Card(CardType.JIRAFA, Color.GREEN),
            None,
            None,
        ])
        self.assertEqual(hell, [Card(CardType.HIPOPOTAMO, Color.YELLOW)])

    def test_canguro(self):
        table_cards = [
            Card(CardType.HIPOPOTAMO, Color.YELLOW),
            Card(CardType.JIRAFA, Color.GREEN),
            Card(CardType.MONO, Color.YELLOW),
            Card(CardType.CANGURO, Color.GREEN),
            None,
        ]

        cards, hell = CardType.canguro_action(3, table_cards.copy(), self.hell, [1])

        # fmt: off
        self.assertEqual(cards, [
            Card(CardType.HIPOPOTAMO, Color.YELLOW),
            Card(CardType.JIRAFA, Color.GREEN),
            Card(CardType.CANGURO, Color.GREEN),
            Card(CardType.MONO, Color.YELLOW),
            None,
        ])

    def test_canguro2(self):
        table_cards = [
            Card(CardType.HIPOPOTAMO, Color.YELLOW),
            Card(CardType.JIRAFA, Color.GREEN),
            Card(CardType.MONO, Color.YELLOW),
            Card(CardType.CANGURO, Color.GREEN),
            None,
        ]

        cards, hell = CardType.canguro_action(3, table_cards.copy(), self.hell, [2])

        # fmt: off
        self.assertEqual(cards, [
            Card(CardType.HIPOPOTAMO, Color.YELLOW),
            Card(CardType.CANGURO, Color.GREEN),
            Card(CardType.JIRAFA, Color.GREEN),
            Card(CardType.MONO, Color.YELLOW),
            None,
        ])

    def test_loro(self):
        table_cards = [
            Card(CardType.HIPOPOTAMO, Color.YELLOW),
            Card(CardType.JIRAFA, Color.GREEN),
            Card(CardType.MONO, Color.YELLOW),
            Card(CardType.LORO, Color.GREEN),
            None,
        ]

        cards, hell = CardType.loro_action(3, table_cards.copy(), self.hell, [1])

        # fmt: off
        self.assertEqual(cards, [
            Card(CardType.HIPOPOTAMO, Color.YELLOW),
            Card(CardType.MONO, Color.YELLOW),
            Card(CardType.LORO, Color.GREEN),
            None,
            None
        ])
        self.assertEqual(hell, [Card(CardType.JIRAFA, Color.GREEN)])

    def test_mofeta(self):
        table_cards = [
            Card(CardType.COCODRILO, Color.YELLOW),
            Card(CardType.COCODRILO, Color.BLUE),
            Card(CardType.CEBRA, Color.RED),
            Card(CardType.HIPOPOTAMO, Color.GREEN),
            Card(CardType.MOFETA, Color.YELLOW),
        ]

        cards, hell = CardType.mofeta_action(4, table_cards.copy(), self.hell, [])

        # fmt: off
        self.assertEqual(cards, [
            Card(CardType.CEBRA, Color.RED),
            Card(CardType.MOFETA, Color.YELLOW),
            None, None, None,
        ])
        self.assertEqual(hell, [
            Card(CardType.COCODRILO, Color.YELLOW),
            Card(CardType.COCODRILO, Color.BLUE),
            Card(CardType.HIPOPOTAMO, Color.GREEN),
        ])
