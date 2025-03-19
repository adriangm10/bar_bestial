# Bar Bestial

This is a terminal-based implementation of the classical board game [Bar Bestial](https://quejuegosdemesa.com/wp-content/uploads/2019/07/Bar-Bestial.pdf) where beasts battle for entry into the parties of the bar.

## Gameplay

In **Bar Bestial**, players each control a hand of 4 animal-themed cards with unique powers and attempt to be the first to get their beasts into the bar.
The queue outside the bar is tight, only 5 spaces are available, each card in the queue influences the other — some sneak ahead, some block others...
When the queue is filled the first 2 will enter the bar and the last one will go to hell.

### State representation in the terminal

Each card is represented by a square, the square will have the colour of the card, the name (in spanish), force, and its recursive symbol ↻ (if it is).
The queue is represented with 5 squares, the heaven (the bar) is always at the left and the hell at the right.

## Features

- Player vs Player (up to 4 players)
- Player vs AI (just 1 vs 1 for the moment)
- AI vs AI (just 1 vs 1 for the moment)
- Script to train AIs
- Different game options accesible with `python main.py -h`

### Game modes

- **Basic:** All cards with actions are removed and the recursive actions are only executed when the card is played.
- **Medium:** All the cards are in the game but the recursive actions are still executed only when the card is played.
- **Full:** The game follows the [rules](https://quejuegosdemesa.com/wp-content/uploads/2019/07/Bar-Bestial.pdf) of the original **Bar Bestial** game.

## How to play

You can acces the rules in this link <https://quejuegosdemesa.com/wp-content/uploads/2019/07/Bar-Bestial.pdf>

- To execute the game: `python3 main.py`
- To play against an AI: `python3 main.py --agent [path to the model]`

## Training a new AI

A wrapper class is implemented using the [gymnasium](https://gymnasium.farama.org/) API converting the game into a gymnasium. To train an AI the stable_baselines3 library is used, but anything that works with the API can be used,
the script used to train the AIs is in the `agent_train.py` file.

## Trained AIs

Some AIs have been trained using the stable_baselines3 library, they can be found in the `models/` folder, t(d) is the number of turns represented in the state (1 is just the current and is best) and
p(d) is the number of players in the game.
