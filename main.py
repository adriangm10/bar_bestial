from bar import Game

if __name__ == "__main__":
    game = Game()

    while not game.finished():
        game.print()

        pos = int(input("Select a card to play[0-3]: "))
        actions: list[int] = []
        while True:
            try:
                game.play_card(pos, actions)
                break
            except ValueError as e:
                msg = (
                    game.hands[game.turn][pos].action_msg()
                    if not actions
                    else game.table_cards[actions[0]].action_msg()  # type: ignore[union-attr]
                )
                if msg is None:
                    raise e
                action = int(input(msg + ": "))
                actions.append(action)

    print(f"The winners are: {", ".join([c.name for c in game.winners()])}")
