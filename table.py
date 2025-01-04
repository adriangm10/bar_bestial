# mypy: disable-error-code="empty-body,union-attr"
import unittest
from collections.abc import Callable, Sequence
from enum import Enum
from functools import reduce
from random import sample
from typing import Annotated, Literal, Self

from termcolor import colored

type TableCards = Annotated[list[Card | None], BOARD_SIZE]
type Actions = list[int]
type Hell = list[Card]
type Heaven = list[Card]
type CardFunction = Callable[
    [int, TableCards, Hell, Heaven, Actions], tuple[TableCards, Hell, Heaven]
]

CARD_WIDTH = 15
BOARD_SIZE = 5


class Color(Enum):
    YELLOW = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    WHITE = 4


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

    CIELO = 0
    INFIERNO = 99

    @classmethod
    def playable_list(cls) -> list[Self]:
        return [c for c in cls if c != cls.CIELO and c != cls.INFIERNO]

    @classmethod
    def leon_action(
        cls,
        card_pos: int,
        cards: TableCards,
        hell: Hell,
        heaven: Heaven,
        actions: Actions,
    ) -> tuple[TableCards, Hell, Heaven]:
        if cards[card_pos] is None:
            raise ValueError("cards[card_pos] is None")

        card: Card = cards[card_pos]  # type: ignore[assignment]
        cards[card_pos] = None

        if any([c and c.card_type == cls.LEON for c in cards[:card_pos]]):
            hell.append(card)
            cards[card_pos] = None
            return cards, hell, heaven

        for i in reversed(range(card_pos)):
            c: Card = cards[i]  # type: ignore[assignment]
            if c and c.card_type == CardType.MONO:
                hell.append(c)
                cards[i] = None
                if i < 4 and cards[i + 1]:
                    cards[i] = cards[i + 1]
                    cards[i + 1] = None

        for i in reversed(range(card_pos)):
            cards[i + 1] = cards[i]
        cards[0] = card

        return cards, hell, heaven

    @classmethod
    def hipopotamo_action(
        cls,
        card_pos: int,
        cards: TableCards,
        hell: Hell,
        heaven: Heaven,
        actions: Actions,
    ) -> tuple[TableCards, Hell, Heaven]:
        if cards[card_pos] is None:
            raise ValueError("cards[card_pos] is None")

        card = cards[card_pos]
        cards[card_pos] = None

        for i, c in reversed(list(enumerate(cards[:card_pos]))):
            if (
                c.card_type != cls.CIELO
                and c.card_type != cls.INFIERNO
                and (c.card_type >= cls.HIPOPOTAMO or c.card_type == cls.CEBRA)
            ):
                i += 1
                break

        for j in reversed(range(i, card_pos)):
            cards[j + 1] = cards[j]

        cards[i] = card
        return cards, hell, heaven

    @classmethod
    def cocodrilo_action(
        cls,
        card_pos: int,
        cards: TableCards,
        hell: Hell,
        heaven: Heaven,
        actions: Actions,
    ) -> tuple[TableCards, Hell, Heaven]:
        if cards[card_pos] is None:
            raise ValueError("cards[card_pos] is None")

        card = cards[card_pos]
        cards[card_pos] = None

        for i, c in reversed(list(enumerate(cards[:card_pos]))):
            if (
                c.card_type != cls.CIELO
                and c.card_type != cls.INFIERNO
                and (c.card_type >= cls.COCODRILO or c.card_type == cls.CEBRA)
            ):
                i += 1
                break
            else:
                hell.append(c)  # type: ignore[arg-type]
                cards[i] = None

        dist = card_pos - i
        for j in range(i + 1, BOARD_SIZE - dist - 1):
            cards[j] = cards[j + dist]
            cards[j + dist] = None

        cards[i] = card
        return cards, hell, heaven

    @classmethod
    def serpiente_action(
        cls,
        card_pos: int,
        cards: TableCards,
        hell: Hell,
        heaven: Heaven,
        actions: Actions,
    ) -> tuple[TableCards, Hell, Heaven]:
        if cards[card_pos] is None:
            raise ValueError("cards[card_pos] is None")
        return sorted(cards[: card_pos + 1], reverse=True) + [None] * (BOARD_SIZE - card_pos - 1), hell, heaven  # type: ignore[type-var]

    @classmethod
    def jirafa_action(
        cls,
        card_pos: int,
        cards: TableCards,
        hell: Hell,
        heaven: Heaven,
        actions: Actions,
    ) -> tuple[TableCards, Hell, Heaven]:
        if cards[card_pos] is None:
            raise ValueError("cards[card_pos] is None")
        if card_pos > 0 and cards[card_pos - 1] < cards[card_pos]:  # type: ignore[operator]
            cards[card_pos - 1], cards[card_pos] = cards[card_pos], cards[card_pos - 1]

        return cards, hell, heaven

    @classmethod
    def foca_action(
        cls,
        card_pos: int,
        cards: TableCards,
        hell: Hell,
        heaven: Heaven,
        actions: Actions,
    ) -> tuple[TableCards, Hell, Heaven]:
        if cards[card_pos] is None:
            raise ValueError("cards[card_pos] is None")
        # instead of changing the gates, the queue is inverted
        return cards[card_pos::-1] + [None] * (BOARD_SIZE - card_pos - 1), hell, heaven

    @classmethod
    def camaleon_action(
        cls,
        card_pos: int,
        cards: TableCards,
        hell: Hell,
        heaven: Heaven,
        actions: Actions,
    ) -> tuple[TableCards, Hell, Heaven]:
        if cards[card_pos] is None:
            raise ValueError("cards[card_pos] is None")
        if not actions or cards[actions[0]] is None:
            raise ValueError("invalid action for a camaleon")

        f = cards[actions[0]].card_type.action()
        return f(card_pos, cards, hell, heaven, actions[1:])

    @classmethod
    def mono_action(
        cls,
        card_pos: int,
        cards: TableCards,
        hell: Hell,
        heaven: Heaven,
        actions: Actions,
    ) -> tuple[TableCards, Hell, Heaven]:
        if cards[card_pos] is None:
            raise ValueError("cards[card_pos] is None")

        if any([c.card_type == cls.MONO for c in cards[:card_pos]]):
            new_board = []
            for c in cards[: card_pos + 1]:
                if c.card_type == cls.COCODRILO or c.card_type == cls.HIPOPOTAMO:
                    hell.append(c)  # type: ignore[arg-type]
                else:
                    new_board.append(c)

            # fmt: off
            new_board = [c for c in reversed(new_board) if c.card_type == cls.MONO] + list(
                filter(lambda c: c.card_type != cls.MONO, new_board)  # type: ignore[arg-type]
            )
            cards = new_board + [None] * (BOARD_SIZE - len(new_board))

        return cards, hell, heaven

    @classmethod
    def canguro_action(
        cls,
        card_pos: int,
        cards: TableCards,
        hell: Hell,
        heaven: Heaven,
        actions: Actions,
    ) -> tuple[TableCards, Hell, Heaven]:
        if cards[card_pos] is None:
            raise ValueError("cards[card_pos] is None")
        if not actions or not 0 < actions[0] <= 2:
            raise ValueError("Invalid action for canguro")

        if card_pos - actions[0] >= 0:
            card = cards[card_pos]
            for i in reversed(range(card_pos - actions[0], card_pos)):
                cards[i + 1] = cards[i]
            cards[card_pos - actions[0]] = card

        return cards, hell, heaven

    @classmethod
    def loro_action(
        cls,
        card_pos: int,
        cards: TableCards,
        hell: Hell,
        heaven: Heaven,
        actions: Actions,
    ) -> tuple[TableCards, Hell, Heaven]:
        if cards[card_pos] is None:
            raise ValueError("cards[card_pos] is None")
        if not actions or cards[actions[0]] is None or actions[0] == card_pos:
            raise ValueError("invalid action for a loro")

        hell.append(cards[actions[0]])  # type: ignore[arg-type]
        for i in range(actions[0], card_pos):
            cards[i] = cards[i + 1]

        cards[card_pos] = None
        return cards, hell, heaven

    @classmethod
    def mofeta_action(
        cls,
        card_pos: int,
        cards: TableCards,
        hell: Hell,
        heaven: Heaven,
        actions: Actions,
    ) -> tuple[TableCards, Hell, Heaven]:
        if cards[card_pos] is None:
            raise ValueError("cards[card_pos] is None")

        top1, top2 = 0, 0
        for c in cards:
            if c.value > top1:
                top1, top2 = c.value, top1
            elif c.value > top2:
                top2 = c.value

        filtered_cards = [c for c in cards[:card_pos] if c.value < top2 or c.value == CardType.MOFETA.value]
        hell_cards = [c for c in cards[:card_pos] if c.value >= top2 and c.value != CardType.MOFETA.value]
        cards = filtered_cards + [cards[card_pos]] + [None] * (BOARD_SIZE - len(filtered_cards) - 1)

        return cards, hell + hell_cards, heaven  # type: ignore[operator]

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
                return lambda cpos, cs, hell, hevn, acts: (cs, hell, hevn)
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
            case _:
                raise ValueError("infierno and cielo have no action")

    def is_recursive(self) -> bool:
        match self:
            case (
                CardType.HIPOPOTAMO
                | CardType.COCODRILO
                | CardType.JIRAFA
                | CardType.CEBRA
            ):
                return True
            case _:
                return False

    def __ge__(self, o) -> bool:
        if self.__class__ == o.__class__:
            return self.value >= o.value
        raise ValueError(
            "Error comparing " + str(self.__class__) + " with " + str(o.__class__)
        )

    def __lt__(self, o) -> bool:
        if self.__class__ == o.__class__:
            return self.value < o.value
        raise ValueError(
            "Error comparing " + str(self.__class__) + " with " + str(o.__class__)
        )


class Card:
    def __init__(self, ctype: CardType, color: Color):
        self.card_type = ctype
        self.name = ctype.name
        self.value = ctype.value
        self.recursive = ctype.is_recursive()
        self.color = color

    def format(self) -> list[str]:
        recursive_symbol = ""
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

    def function(self) -> CardFunction:
        match self.card_type:
            case CardType.LEON:
                return CardType.leon_action
            case _:
                raise NotImplementedError

    def __eq__(self, o) -> bool:
        if self.__class__ == o.__class__:
            return self.card_type == o.card_type and self.color == o.color
        raise ValueError(
            "Error comparing " + str(self.__class__) + " with " + str(o.__class__)
        )

    def __lt__(self, o):
        if self.__class__ == o.__class__:
            return self.card_type < o.card_type
        raise ValueError(
            "Error comparing " + str(self.__class__) + " with " + str(o.__class__)
        )

    def __str__(self):
        return "<" + self.card_type.name + ", " + self.color.name + ">"

    def __repr__(self):
        return "<" + self.card_type.name + ", " + self.color.name + ">"


def blank_card() -> list[str]:
    top_border = "╔{}╗".format("═" * (CARD_WIDTH - 2))
    bot_border = "╚{}╝".format("═" * (CARD_WIDTH - 2))
    body = "║" + " " * (CARD_WIDTH - 2) + "║"

    return [top_border, body, body, body, bot_border]


def format_cards(hand: list[Card]) -> list[str]:
    handr = map(lambda c: c.format() if c else blank_card(), hand)
    return reduce(
        lambda acc, card: (
            card if not acc else list(map(lambda x: x[1] + card[x[0]], enumerate(acc)))
        ),
        handr,
        [],
    )


class Table:
    def __init__(self, num_players: Literal[2, 3, 4] = 2):
        if not 2 <= num_players <= 4:
            raise ValueError("There must be between 2 and 4 players")

        self.num_players = num_players
        self.table_cards: TableCards = [None] * BOARD_SIZE
        # self.table[0] = Card(CardType.CIELO, Color.WHITE)
        # self.table[-1] = Card(CardType.INFIERNO, Color.WHITE)
        self.hell: list[Card] = []
        self.heaven: list[Card] = []

        cardType_list = CardType.playable_list()
        cardType_count = len(cardType_list)
        self.decks = [
            [Card(ct, Color(c)) for ct in sample(cardType_list, cardType_count)]
            for c in range(num_players)
        ]

        self.hands = [d[:4] for d in self.decks]
        self.decks = [d[4:] for d in self.decks]

        self.turn = 0

    def print(self):
        hand_rep = format_cards(self.hands[self.turn])
        table_rep = format_cards(self.table_cards)

        print(f"{"\n".join(table_rep)}\n\n")
        print(f"{"\n".join(hand_rep)}\n\n")

    def play_card(self, card_idx: int, actions: Actions):
        card = self.hands[self.turn].pop(card_idx)

        for i, c in enumerate(self.table_cards):
            if c is None:
                self.table_cards[i] = card
                break

        # execute card
        # f = card.function()
        # self.table, self.hell, self.heaven = f(
        #     i, self.table, self.hell, self.heaven, actions
        # )
        # ======

        # execute recurrent cards
        for i, c in enumerate(self.table_cards):
            if c and card != c and c.recursive:
                pass

        # open heaven and hell doors
        if all([c is not None for c in self.table_cards]):
            self.heaven.append(self.table_cards[0])  # type: ignore[arg-type]
            self.heaven.append(self.table_cards[1])  # type: ignore[arg-type]
            self.hell.append(self.table_cards[4])  # type: ignore[arg-type]
            self.table_cards[4] = None
            self.table_cards[0] = self.table_cards[2]
            self.table_cards[1] = self.table_cards[3]
            self.table_cards[2] = None
            self.table_cards[3] = None

        # draw a card
        if self.decks[self.turn]:
            self.hands[self.turn].append(self.decks[self.turn].pop(0))

        # next turn
        self.turn = (self.turn + 1) % self.num_players


class TestTable(unittest.TestCase):
    def test_play_card(self):
        table = Table()
        turn = table.turn
        card = table.hands[turn][0]
        deck = table.decks[turn].copy()
        table.play_card(0, [])

        self.assertEqual(table.table_cards[0], card)
        self.assertEqual(len(table.hands[turn]), 4)
        self.assertEqual(deck[1:], table.decks[turn])

    def test_play_card_empty_deck(self):
        table = Table()
        turn = table.turn
        card = table.hands[table.turn][0]
        table.decks[turn] = []
        table.play_card(0, [])

        self.assertEqual(table.table_cards[0], card)
        self.assertEqual(len(table.hands[turn]), 3)

    def test_play_card_open_doors(self):
        table = Table()
        table_cards = [
            Card(CardType.LEON, Color.GREEN),
            Card(CardType.HIPOPOTAMO, Color.YELLOW),
            Card(CardType.COCODRILO, Color.GREEN),
            Card(CardType.COCODRILO, Color.YELLOW),
        ]
        table.table_cards[:4] = table_cards
        hell_card = table.hands[table.turn][0]
        table.play_card(0, [])

        self.assertEqual(table.heaven, table_cards[:2])
        self.assertEqual(table.hell, [hell_card])
        self.assertEqual(table.table_cards, table_cards[2:] + [None, None, None])


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
        cards, hell, heaven = CardType.leon_action(
            3, table_cards.copy(), self.hell, self.heaven, []
        )

        self.assertEqual(
            cards, [Card(CardType.LEON, Color.GREEN)] + table_cards[:3] + [None]
        )

    def test_lion_action_monkeys(self):
        table_cards = [
            Card(CardType.HIPOPOTAMO, Color.YELLOW),
            Card(CardType.MONO, Color.GREEN),
            Card(CardType.COCODRILO, Color.YELLOW),
            Card(CardType.MONO, Color.YELLOW),
            None,
        ]
        table_cards[4] = Card(CardType.LEON, Color.GREEN)
        cards, hell, heaven = CardType.leon_action(
            4, table_cards.copy(), self.hell, self.heaven, []
        )

        # fmt: off
        self.assertEqual(cards, [
            table_cards[4],
            table_cards[0],
            table_cards[2],
            None,
            None,
        ])  # leon hipopotamo cocodrilo
        self.assertEqual(hell, [table_cards[3], table_cards[1]])

    def test_lion_action_other_lion(self):
        table_cards = [
            Card(CardType.LEON, Color.GREEN),
            Card(CardType.HIPOPOTAMO, Color.YELLOW),
            Card(CardType.COCODRILO, Color.YELLOW),
            Card(CardType.MONO, Color.GREEN),
            None,
        ]
        table_cards[3] = Card(CardType.LEON, Color.YELLOW)
        cards, hell, heaven = CardType.leon_action(
            3, table_cards.copy(), self.hell, self.heaven, []
        )

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
        cards, hell, heaven = CardType.hipopotamo_action(
            3, table_cards.copy(), self.hell, self.heaven, []
        )
        table_cards[3] = None

        self.assertEqual(
            cards,
            [table_cards[0]]
            + [Card(CardType.HIPOPOTAMO, Color.YELLOW)]
            + table_cards[1:-1],
        )

    def test_hipopotamo_middle(self):
        table_cards = [
            Card(CardType.COCODRILO, Color.GREEN),
            Card(CardType.JIRAFA, Color.GREEN),
            Card(CardType.HIPOPOTAMO, Color.YELLOW),
            Card(CardType.COCODRILO, Color.YELLOW),
            None,
        ]
        cards, hell, heaven = CardType.hipopotamo_action(
            2, table_cards.copy(), self.hell, self.heaven, []
        )

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
        cards, hell, heaven = CardType.cocodrilo_action(
            3, table_cards.copy(), self.hell, self.heaven, []
        )
        table_cards[3] = None

        self.assertEqual(
            cards, [table_cards[0], Card(CardType.COCODRILO, Color.YELLOW)] + [None] * 3
        )
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
        cards, hell, heaven = CardType.cocodrilo_action(
            2, table_cards.copy(), self.hell, self.heaven, []
        )

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
        cards, hell, heaven = CardType.serpiente_action(
            3, table_cards.copy(), self.hell, self.heaven, []
        )

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
        cards, hell, heaven = CardType.jirafa_action(
            2, table_cards.copy(), self.hell, self.heaven, []
        )

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
        cards, hell, heaven = CardType.foca_action(
            3, table_cards.copy(), self.hell, self.heaven, []
        )

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
        cards, hell, heaven = CardType.camaleon_action(
            3, table_cards.copy(), self.hell, self.heaven, [0]
        )

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

        cards, hell, heaven = CardType.mono_action(
            3, table_cards.copy(), self.hell, self.heaven, []
        )

        self.assertEqual(cards, table_cards)

    def test_mono_2(self):
        table_cards = [
            Card(CardType.HIPOPOTAMO, Color.YELLOW),
            Card(CardType.JIRAFA, Color.GREEN),
            Card(CardType.MONO, Color.YELLOW),
            Card(CardType.MONO, Color.GREEN),
            None,
        ]

        cards, hell, heaven = CardType.mono_action(
            3, table_cards.copy(), self.hell, self.heaven, []
        )

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

        cards, hell, heaven = CardType.canguro_action(
            3, table_cards.copy(), self.hell, self.heaven, [1]
        )

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

        cards, hell, heaven = CardType.canguro_action(
            3, table_cards.copy(), self.hell, self.heaven, [2]
        )

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

        cards, hell, heaven = CardType.loro_action(
            3, table_cards.copy(), self.hell, self.heaven, [1]
        )

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

        cards, hell, heaven = CardType.mofeta_action(
            4, table_cards.copy(), self.hell, self.heaven, []
        )

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
