from collections.abc import Callable
from enum import Enum
from functools import reduce
from typing import Annotated

from termcolor import colored

type Hand = Annotated[list[Card | None], 4]
type TableCards = Annotated[list[Card | None], 5]
type Actions = list[int]
type Hell = list[Card]
type Heaven = list[Card]
type CardFunction = Callable[
    [Card, TableCards, Hell, Heaven, Actions], tuple[TableCards, Hell, Heaven]
]


class Color(Enum):
    WHITE = 0
    YELLOW = 1
    BLUE = 2
    RED = 3
    GREEN = 4


class CardName(Enum):
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

    def is_recursive(self) -> bool:
        match self:
            case (
                CardName.HIPOPOTAMO
                | CardName.COCODRILO
                | CardName.JIRAFA
                | CardName.CEBRA
            ):
                return True
            case _:
                return False


class Card:
    """
    Two cards are equal if they have the same value (it ignores color).
    """

    def __init__(self, name: CardName, color: Color):
        self.card_type = name
        self.name = name.name
        self.value = name.value
        self.recursive = name.is_recursive()
        self.color = color

    def format(self) -> list[str]:
        width = 15
        recursive_symbol = ""
        top_border = "╔{}╗".format("═" * (width - 2))
        bot_border = "╚{}╝".format("═" * (width - 2))
        header = "║{:02}{}{:02}║".format(
            self.value,
            " " * (width - 6),
            self.value,
        )
        body1 = "║{}{}{} ║".format(
            recursive_symbol if self.recursive else " ",
            " " * (width - 5),
            recursive_symbol if self.recursive else " ",
        )
        body2 = ("║{:^" + str(width - 2) + "}║").format(self.name)

        color = self.color.name.lower()

        return [
            colored(top_border, color),  # type: ignore[arg-type]
            colored(header, color),  # type: ignore[arg-type]
            colored(body1, color),  # type: ignore[arg-type]
            colored(body2, color),  # type: ignore[arg-type]
            colored(bot_border, color),  # type: ignore[arg-type]
        ]

    def leon_function(
        self, cards: TableCards, hell: Hell, heaven: Heaven, actions: Actions
    ) -> tuple[TableCards, Hell, Heaven]:
        if self in cards:
            hell.append(self)
            return cards, hell, heaven

        for i in reversed(range(5)):
            card = cards[i]
            if card is not None and card.card_type == CardName.MONO:
                hell.append(card)
                cards[i] = None
                if i < 4 and cards[i + 1] is not None:
                    cards[i] = cards[i + 1]
                    cards[i + 1] = None

        for i in reversed(range(4)):
            cards[i + 1] = cards[i]
        cards[0] = self

        return cards, hell, heaven

    def act(
        self, cards: TableCards, hell: Hell, heaven: Heaven, actions: Actions
    ) -> tuple[TableCards, Hell, Heaven]:
        match self.card_type:
            case CardName.LEON:
                return self.leon_function(cards, hell, heaven, actions)
            case _:
                raise NotImplementedError

    def __eq__(self, o):
        return self.value == o.value


def format_cards(hand: list[Card]) -> list[str]:
    handr = map(lambda c: c.format(), hand)
    return reduce(
        lambda acc, card: (
            card if not acc else list(map(lambda x: x[1] + card[x[0]], enumerate(acc)))
        ),
        handr,
        [],
    )


class Table:
    def __init__(self, num_players=2):
        self.hands = [[None] * 4] * num_players
        self.num_players = num_players
        self.table = [None] * 5
        self.hell = []
        self.heaven = []

        self.hands = [
            [
                Card(CardName.LEON, Color.BLUE),
                Card(CardName.HIPOPOTAMO, Color.BLUE),
                Card(CardName.JIRAFA, Color.BLUE),
            ],
            [
                Card(CardName.LEON, Color.GREEN),
                Card(CardName.HIPOPOTAMO, Color.GREEN),
                Card(CardName.JIRAFA, Color.GREEN),
            ],
        ]

        self.table = [
            Card(CardName.CAMALEON, Color.BLUE),
            Card(CardName.MONO, Color.GREEN),
            Card(CardName.COCODRILO, Color.GREEN),
            Card(CardName.MONO, Color.BLUE),
        ]

    def print(self):
        hands_rep = [format_cards(hand) for hand in self.hands]
        table = list(filter(lambda card: card is not None, self.table))
        table_rep = format_cards(table)

        print(f"{"\n".join(hands_rep[0])}\n\n")
        print(f"{"\n".join(table_rep)}\n\n")
        print(f"{"\n".join(hands_rep[1])}")
