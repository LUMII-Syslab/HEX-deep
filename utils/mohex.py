# This file contains code that can run the MoHex solver and play a game against him
import subprocess


class MoHex:
    def __init__(
        self,
        board_size: int = 13,
        move_time_limit: float = 10.0,
        is_starting_player: bool = False
    ):
        print("Initializing the MoHex solver...")
        print(f"MoHex will play on a board with size {board_size}x{board_size}")
        print(f"Mohex move time limit will be {move_time_limit}")
        self.board_size = board_size
        self.move_time_limit = move_time_limit
        self.is_starting_player = is_starting_player
        # Create the process that will manage MoHex solver
        self.process = subprocess.Popen(
            ["/tmp/benzene-vanilla-cmake/build/src/mohex/mohex"],  # Path to the MoHex executable
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # Line-buffered output
        )
        # Initialize the board
        # Although this command will limit the time, it doesn't do a hard limit, that
        #   is, for example, if you set it to 0.1, it still might take a second or two
        # Note: If you run the program directly, this time resets to default every time
        #   you close the program
        self.send_command_and_get_answer(f"param_mohex max_time {self.move_time_limit}")
        self.send_command_and_get_answer(f"boardsize {self.board_size}")

    def send_command_and_get_answer(self, command):
        # Send the command
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
        # Read the output
        output = []
        while True:
            line = self.process.stdout.readline().strip()
            if line == "":
                break
            output.append(line)
        return output

    def set_opponent_move(self, move: str):
        self.send_command_and_get_answer(f"play {'black' if not self.is_starting_player else 'white'} {move}")

    def generate_move(self):
        move = self.send_command_and_get_answer(f"genmove {'black' if self.is_starting_player else 'white'}")
        if len(move) == 0:
            raise Exception("Something went wrong while MoHex was generating the move...")
        return move[0][2:]

    def close(self):
        # Closes the process forcefully (if this leaves temporary files lying around, we
        #   should later change it, so it sends a "quit" command
        if self.process:  # If the process still exists
            self.process.terminate()  # Forcefully kill the subprocess
            self.process.wait()  # Wait for the process to exit to avoid zombie processes
