from table import Game

if __name__ == "__main__":
    table = Game()
    table.print()

    print("═" * 110, end="\n\n")
    table.play_card(0, [])
    table.print()

    print("═" * 110, end="\n\n")
    table.play_card(1, [])
    table.print()
