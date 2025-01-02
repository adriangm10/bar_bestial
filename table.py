from collections.abc import Callable, Sequence
from copy import deepcopy
from enum import Enum
from functools import reduce
from random import sample
from typing import Annotated, Literal

from termcolor import colored

type Hand = Annotated[list[Card], 4]
type TableCards = Annotated[list[Card | None], 7]
type Actions = list[int]
type Hell = list[Card]
type Heaven = list[Card]
type CardFunction = Callable[
    [int, TableCards, Hell, Heaven, Actions], tuple[TableCards, Hell, Heaven]
]

CARD_WIDTH = 15


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
    def to_playable_list(cls) -> list[int]:
        return [c.value for c in cls if c != cls.CIELO and c != cls.INFIERNO]

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

    def __lt__(self, o) -> bool:
        if self.__class__ == o.__class__:
            return self.value < o.value
        raise ValueError("o is of different class")


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

    # def leon_function(
    #     self, cards: TableCards, hell: Hell, heaven: Heaven, actions: Actions
    # ) -> tuple[TableCards, Hell, Heaven]:
    #     # change the way to search other lions
    #     if self in cards:
    #         hell.append(self)
    #         return cards, hell, heaven

    #     for i in reversed(range(5)):
    #         card = cards[i]
    #         if card and card.card_type == CardType.MONO:
    #             hell.append(card)
    #             cards[i] = None
    #             if i < 4 and cards[i + 1]:
    #                 cards[i] = cards[i + 1]
    #                 cards[i + 1] = None

    #     for i in reversed(range(4)):
    #         cards[i + 1] = cards[i]
    #     cards[0] = self

    #     return cards, hell, heaven

    # def act(
    #     self, cards: TableCards, hell: Hell, heaven: Heaven, actions: Actions
    # ) -> tuple[TableCards, Hell, Heaven]:
    #     match self.card_type:
    #         case CardType.LEON:
    #             return self.leon_function(cards, hell, heaven, actions)
    #         case _:
    #             raise NotImplementedError

    def __eq__(self, o):
        return self.value == o.value and self.color == o.color


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
        self.num_players = num_players
        self.table: TableCards = [None] * 7
        self.table[0] = Card(CardType.CIELO, Color.WHITE)
        self.table[-1] = Card(CardType.INFIERNO, Color.WHITE)
        self.hell: list[Card] = []
        self.heaven: list[Card] = []

        cardType_list = CardType.to_playable_list()
        cardType_count = len(cardType_list)
        self.decks = [
            [
                Card(CardType(ct), Color(c))
                for ct in sample(cardType_list, cardType_count)
            ]
            for c in range(num_players)
        ]

        self.hands: list[Hand] = [deepcopy(d[:4]) for d in self.decks]
        self.decks = [d[4:] for d in self.decks]

        self.turn = 0

    def print(self):
        hand_rep = format_cards(self.hands[self.turn])
        table_rep = format_cards(self.table)

        print(f"{"\n".join(table_rep)}\n\n")
        print(f"{"\n".join(hand_rep)}\n\n")

    def play_card(self, card_idx: int, actions: Actions):
        card = self.hands[self.turn].pop(card_idx)

        if self.table[0].card_type == CardType.CIELO:  # type: ignore[union-attr]
            for i, c in enumerate(self.table):
                if c is None:
                    self.table[i] = card
                    break
        else:
            for i, c in reversed(list(enumerate(self.table))):
                if c is None:
                    self.table[i] = card
                    break

        # execute card
        # ======

        # execute recurrent cards
        for i, c in enumerate(self.table):
            if c and card != c and c.recursive:
                pass

        # open heaven and hell doors
        if all(map(lambda c: c is not None, self.table[1:-1])):
            if self.table[0].card_type == CardType.CIELO:  # type: ignore[union-attr]
                self.heaven.append(self.table[1])  # type: ignore[arg-type]
                self.heaven.append(self.table[2])  # type: ignore[arg-type]
                self.hell.append(self.table[-2])  # type: ignore[arg-type]
                self.table[-2] = None
                for i in range(1, 5):
                    self.table[i] = self.table[i + 1]
            else:
                self.heaven.append(self.table[-2])  # type: ignore[arg-type]
                self.heaven.append(self.table[-3])  # type: ignore[arg-type]
                self.hell.append(self.table[1])  # type: ignore[arg-type]
                self.table[1] = None
                for i in reversed(range(2, 6)):
                    self.table[i] = self.table[i - 1]

        # draw a card
        if self.decks[self.turn]:
            self.hands[self.turn].append(self.decks[self.turn].pop(0))

        # next turn
        self.turn = (self.turn + 1) % self.num_players
