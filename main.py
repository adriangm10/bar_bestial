from table import Table

if __name__ == "__main__":
    table = Table()
    table.print()

    print("═" * 110, end="\n\n")
    table.play_card(0, [])
    table.print()

    print("═" * 110, end="\n\n")
    table.play_card(1, [])
    table.print()
