from bar import CardType, Game

if __name__ == "__main__":
    game = Game()

    while not game.finished():
        game.print()

        pos = int(input("Select a card to play[0-3]: "))
        actions: list[int] = []
        msg = game.hands[game.turn][pos].action_msg()
        if msg:
            actions.append(int(input(msg + ": ")))
            if game.hands[game.turn][pos].card_type == CardType.CAMALEON and (
                msg := game.table_cards[actions[0]].action_msg()  # type: ignore[union-attr]
            ):
                actions.append(int(input(msg + ": ")))

        game.play_card(pos, actions)

    print(f"The winners are: {", ".join([c.name for c in game.winners()])}")
