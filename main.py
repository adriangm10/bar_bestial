from table import Game

if __name__ == "__main__":
    game = Game()

    while not game.finished():
        game.print()

        # TODO: change the way to input card actions
        actions = list(map(int, input(f"Make your action (separated by \",\"): ").split(",")))
        game.play_card(actions[0], actions[1:])


    print(f"Winners colors are: {game.winners()}")
