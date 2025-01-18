# This file contains code that can run our model against the SOTA MoHex model

import string
from typing import List

import numpy as np
import torch

from config import Config
from model.hex3 import softmax_global, sample_gumbel
from utils.mohex import MoHex

noise_scale = 0.5

# NB!: In this file the Hex board is filled in the following way:
#   1. If a cell has the value 0, it means that the space is empty;
#   2. If a cell has the value 1, it means that UL IMCS has taken the cell;
#   3. If a cell has the value 2, it means that MoHex has taken the cell.


def play_against_sota(model, train_step, writer, starting_player):
    # starting_player is the index of the starting (black) player. Index 1 means that
    #   our (UL IMCS) model is the starting player, index 2 means that MoHex is the
    #   starting player
    board_size = model.board_size
    #   # Each row is created independently
    board = [[0 for _ in range(board_size)] for _ in range(board_size)]

    print_board(board, starting_player)

    print(f"Playing a game: {'UL IMCS' if starting_player == 1 else 'MoHex'} vs. {'MoHex' if starting_player == 1 else 'UL IMCS'}...")
    print("UL IMCS" if starting_player == 1 else "MoHex", "is starting (black) player...")
    print("MoHex" if starting_player == 1 else "UL IMCS", "is following (white) player...")

    # Move count is useful, as, for example, if you lose in 10 moves, you probably
    #   played horrible moves, however, if you lose in 40 moves, you had at least some
    #   kind of battle. That is, if we lose, we want to lose in many moves, and, if we
    #   win, we want to win in very little moves, but this metric isn't completely
    #   reliable
    # For a 9x9 board the fastest possibly victory is in 17 and 18 moves for black and
    #   white pieces respectively
    moves = 0

    our_solver = model.player_model
    # All hard coded 1 below specify the batch size, as it's a single game, it's 1

    if starting_player == 1:
        state = our_solver.initial_state(1)
    else:
        state = torch.transpose(our_solver.initial_state(1), 2, 3)
    # Moves made by UL IMCS
    moves_by_ul_imcs = torch.zeros([1, 1, board_size, board_size], device=Config.device)
    # Moves made by MoHex
    moves_by_mohex = torch.zeros([1, 1, board_size, board_size], device=Config.device)
    # Last move made by MoHex
    last_mohex_move = torch.zeros([1, 1, board_size, board_size], device=Config.device)

    # Open the solver
    mohex_solver = MoHex(board_size=board_size, move_time_limit=0.1, is_starting_player=(starting_player == 2))

    current_player = starting_player
    last_player = 0  # 0 – no last player yet
    while not has_game_ended(
        board=board,
        last_player_id=last_player,
        starting_player_id=starting_player
    ):
        # Start of turn code
        moves += 1

        print(f"[Move {moves}] {'UL IMCS' if current_player == 1 else 'MoHex'} ({'black' if current_player == starting_player else 'white'}) has to move now...")

        def char_to_index(char):  # For example, "a" is 0, "b" is 1, ..., used to access columns
            return ord(char) - ord("a")

        def assert_legal_move(the_board, the_move):
            # Assumes that the move is not out of range, only checks if the move isn't
            #   contradicting another previous move by one of the players
            assert the_board[int(the_move[1:]) - 1][char_to_index(the_move[0])] == 0

        if current_player == 1:  # (1) If UL IMCS needs to move
            # To get ideas on how to implement the UL IMCS move you need to go to
            #   hex3.py, HexGame class and see forward function, which gets called by
            #   train_step and predict_step
            # 1 - moves_by_player_one - moves_by_player_two are the empty spaces
            model_inputs = torch.cat(
                [
                    moves_by_ul_imcs,
                    moves_by_mohex,
                    1 - moves_by_ul_imcs - moves_by_mohex,
                    last_mohex_move,
                    torch.zeros_like(moves_by_ul_imcs) #todo swap_allowed
                ],
                dim=1
            )
            if starting_player == 2:  # If UL IMCS is the second player, we need to transpose
                model_inputs = torch.transpose(model_inputs, 2, 3)
            move_logits, state, _ = our_solver(
                model_inputs,
                state
            )
            if starting_player == 2:  # If UL IMCS is the second player, we need to transpose
                move_logits = torch.transpose(move_logits, 2, 3).contiguous()

            move_logits = move_logits + sample_gumbel(move_logits.shape) * noise_scale
            move = softmax_global(move_logits, 1 - moves_by_ul_imcs - moves_by_mohex)
            move, moves_by_ul_imcs = model.move(move, moves_by_ul_imcs, moves_by_mohex)
            # Flatten the move tensor and find the index of the maximum value
            flat_index = torch.argmax(move)
            # Convert the flat index back to the original shape
            max_index = torch.unravel_index(flat_index, move.shape)
            # Get the actual, useful values
            row = int(max_index[2])
            column = int(max_index[3])

            # MoHex uses 1-based numbering, so we need to add one
            mohex_row = row + 1
            # MoHex requires lowercase letters for column indexing
            mohex_column = chr(ord("a") + column)

            move = f"{mohex_column}{mohex_row}"
            print(f"UL IMCS move: {move}")

            assert_legal_move(board, move)

            # Update the local board
            board[row][column] = current_player
            # Update the MoHex board
            mohex_solver.set_opponent_move(move)
        else:  # elif current_player == 2:  # (2) If MoHex needs to move
            move = mohex_solver.generate_move()
            print(f"MoHex move was: {move}")
            assert_legal_move(board, move)

            letter = move[0]
            nmbr = int(move[1:])

            board[nmbr - 1][char_to_index(letter)] = current_player

            last_mohex_move.zero_()
            last_mohex_move[0][0][nmbr - 1][char_to_index(letter)] = 1
            moves_by_mohex[0][0][nmbr - 1][char_to_index(letter)] = 1

        print_board(board, starting_player)
        # Let's print the board to TensorBoard
        board_group_name = f"sota_battle_{'we_start' if starting_player == 1 else 'mohex_starts'}"
        # Green moves are us, the good guys, red moves are MoHex, the bad guys
        model.show_image(moves_by_ul_imcs, moves_by_mohex, f"{board_group_name}/{moves}", None, train_step)

        # End of turn code
        last_player = current_player
        # If current player is 1, it's going to be 2, if it's 2, it's going to be 1
        current_player = 3 - current_player
    mohex_solver.close()
    print("Game ended!")
    print(f"{'UL IMCS' if last_player == 1 else 'MoHex'} ({'black' if last_player == starting_player else 'white'}) won against {'MoHex' if last_player == 1 else 'UL IMCS'} ({'white' if last_player == starting_player else 'black'}) in {moves} moves...")

    # Let's add winning or losing to TensorBoard
    score_group_name = f"testloss/{'we_start' if starting_player == 1 else 'mohex_starts'}_we_win"
    writer.add_scalar(
        score_group_name,
        np.mean(1.0 if last_player == 1 else 0.0),
        train_step
    )


def evaluate_against_sota(model, train_step, writer):
    print("Comparing our model against the SOTA model...")

    # The current implementation plays 1 game, where UL IMCS begins and 1 game, where
    #   MoHex begins, but later we can create more games, for example, force UL IMCS and
    #   MoHex to start in all possible cells

    # Play Game 1: UL IMCS starts
    print("Game 1 (UL IMCS starts) is starting...")
    play_against_sota(model, train_step, writer, starting_player=1)

    # Play Game 2: MoHex starts
    print("Game 2 (MoHex starts) is starting...")
    play_against_sota(model, train_step, writer, starting_player=2)


def print_board(board, starting_player_id):
    """
              BLACK
           A B C D E
        1  [][][][][]
         2  [][][][][]
    WHITE 3  [][][][][]
    """
    board_size = len(board)
    print(f"  {' ' * int((board_size / 2) * 3)}B")
    print("  ", end="")
    for i in range(board_size):
        print(f" {string.ascii_uppercase[i]} ", end="")
    print()
    for row_id, row in enumerate(board):
        starting_spaces = row_id * 2 if row_id + 1 < 10 else row_id * 2 - 1
        if row_id + 1 == int(board_size / 2) + 1:
            print(f"{' ' * (starting_spaces - 1 - 1) if board_size != 1 else ' '}W {row_id + 1}", end=" ")
        else:
            print(f"{' ' * starting_spaces}{row_id + 1}", end=" ")
        for cell in row:
            if cell == 0:
                cell_text = " "
            elif cell == starting_player_id:
                cell_text = "B"
            else:
                cell_text = "W"
            print(f"[{cell_text}]", end="")
        print(f" {row_id + 1}{' W' if row_id + 1 == int(board_size / 2) + 1 else ''}")
    print("  ", end="")
    print(f"{' ' * (2 * (board_size - 1) - 1)}", end=" ")
    for i in range(board_size):
        print(f" {string.ascii_uppercase[i]} ", end="")
    print()
    print(f"  {' ' * (board_size - 1) * 2}{' ' * int((board_size / 2) * 3)}B")


def has_game_ended(
    board: List[List[int]],
    last_player_id: int,
    starting_player_id: int
) -> bool:
    """ Given a Hex board, this function finds out whether the game has ended.

    Before the last player made his move the game obviously was still ongoing, that
        means we only have to check if last player's move made the game end with his
        pieces.

    Args:
        board: The Hex board to check.
        last_player_id: Index of the last player. Index 0 means that the game just
            started, that is, there isn't a last player yet. Index 1 means that UL IMCS
            was the last player. Index 2 means that MoHex was the last player.
        starting_player_id: Index of the starting player. For index descriptions see
            last_player_id docstring.

    Returns:
        game_has_ended: Whether the game has ended.
    """
    # If last_player_id is 0, that means that the game just started, so obviously it
    #   hasn't ended
    if last_player_id == 0:
        return False

    # We need to check if any of the players have filled the board in a way that a full
    #   line connects. Effective way to do this is to do DFS (what we will do) or BFS,
    #   or make the board into a graph, as is done in the model code, and then check
    #   with nx.has_path if there is a path between all the points that could make a
    #   winning line

    # Keep in mind that if both players play very horribly, for example, black players
    #   connects the board's white sides with black pieces (only possible if white
    #   player plays very intentionally badly), then the white player can't win anymore,
    #   but the black player will win (by the theorem that there is always a winner in
    #   Hex)

    # If last_player_id is starting_player_id, we will try to see if he has connected
    #   the top and bottom with last_player_id written in cells, if the last_player_id
    #   is not the same as starting_player_id, we will check if he has connected left
    #   and right side with last_player_id written in cells
    connect_top = last_player_id == starting_player_id
    connecting_value = last_player_id

    board_size = len(board)
    # Let's keep visited cells set, to avoid redundant checks during DFS
    visited_cells = set()

    # Let's define directions for the neighboring cells in the hex grid:
    # If X is the current cell, it has 6 neighbors (from 1 to 6):
    # 1. neighbor – one row higher (x--), same column
    # 2. neighbor – one row higher (x--), next column (y++)
    # 3. neighbor – same row, previous column (y--)
    # 4. neighbor – same row, next column (y++)
    # 5. neighbor – one row lower (x++), previous column (y--)
    # 6. neighbor – one row lower (x++), same column
    # [ ][1][2]
    #   [3][X][4]
    #     [5][6][ ]
    directions = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

    def is_valid_unexplored_player_cell(x, y):
        # The cell is in Hex board limits
        # The cell isn't visited
        # The cell belongs to the correct player
        return 0 <= x < board_size and 0 <= y < board_size and (x, y) not in visited_cells and board[x][y] == connecting_value

    # This function is initially called from a point on the top side or a point on the
    #  left side, then it recursively explores each neighboring cell and stops if it
    #  reaches the opposite side or runs out of cells
    def dfs(x, y):
        # For the player that needs to connect top with the bottom, if x (row) index has
        #   reached the end, we can say that there is a connected line
        if connect_top and x == board_size - 1:
            return True
        # For the player that needs to connect the left side with the right side, if y
        #   (column) index has reached the end, we can say that there is a connected
        #   line
        elif (not connect_top) and y == board_size - 1:
            return True

        # We have visited the cell
        visited_cells.add((x, y))

        # If no connected line, has been found yet, explore all other neighboring cells
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if is_valid_unexplored_player_cell(nx, ny) and dfs(nx, ny):
                # Return True, if DFS found truth as well
                return True

        return False

    # If there is a connecting line, there is at least one point for at the top that
    #   somehow goes to the bottom fully connected or there is at least one point on the
    #   left side that somehow goes to the right side fully connected.
    # To find out if such a line exists, we will check for all starting points if they
    #   connect a line
    # Starting points for connect top player are all the points at the top, that is the
    #   first row
    if connect_top:
        for j in range(board_size):
            if board[0][j] == connecting_value and dfs(0, j):
                return True
    # Starting points for connect sides player are all the points on the left side, that
    #   is the first column
    else:  # not connect_top:
        for i in range(board_size):
            if board[i][0] == connecting_value and dfs(i, 0):
                return True

    # If a connecting line hasn't been found, the game is still on!
    return False
