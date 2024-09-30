import unittest
import chess
from chess_net_server import minimax, ai_color, simple_evaluate

class TestMinimax(unittest.TestCase):
    def setUp(self):
        self.depth = 3
        self.alpha = float('-inf')
        self.beta = float('inf')

    def debug_print_board(self, board, message):
        print(f"\n{message}")
        self.print_board_with_enumeration(board)
        print(f"FEN: {board.fen()}")

    # Function to print the board with enumeration
    def print_board_with_enumeration(self, board):
        board_str = str(board)
        rows = board_str.split('\n')
    
        # Add column labels (a-h)
        col_labels = '   a b c d e f g h'
        print(col_labels)
    
        # Add row numbers and board contents
        for i, row in enumerate(rows):
            print(f"{8-i}  {row}  {8-i}")
    
        # Repeat column labels at the bottom
        print(col_labels)

    def test_checkmate_in_one(self):
        # Fool's mate position
        board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2")
        self.debug_print_board(board, "Checkmate in one before minimax:")
        score, move = minimax(board, self.depth, self.alpha, self.beta, True)
        board.push(move)
        self.debug_print_board(board, "Position after minimax move: " + str(move))
        self.assertEqual(move, chess.Move.from_uci("d8h4"))  # Qh4#

    def test_avoid_checkmate(self):
        # Position where black is threatening checkmate
        board = chess.Board("3r1rk1/ppp2ppp/8/4N3/8/1P6/PB3PPP/3R2K1 w - - 0 1")
        self.debug_print_board(board, "Avoid checkmate position before minimax:")
        score, move = minimax(board, self.depth, self.alpha, self.beta, True)
        board.push(move)
        self.debug_print_board(board, "Position after minimax move: " + str(move))
        self.assertNotEqual(move, chess.Move.from_uci("e5f7"))  # Nf7 loses to Rd1#

    def test_promotion(self):
        # Position with a pawn about to promote
        board = chess.Board("8/4P3/8/8/8/8/5k2/4K3 w - - 0 1")
        self.debug_print_board(board, "Promotion position before minimax:")
        score, move = minimax(board, self.depth, self.alpha, self.beta, True)
        board.push(move)
        self.debug_print_board(board, "Position after minimax move: " + str(move))
        self.assertEqual(move, chess.Move.from_uci("e7e8q"))  # e8=Q

    def test_nimzo_penalize_bad_move(self):
        # The knight is pinned and cannot move and should be captured.
        board = chess.Board("rnbqk2r/pppp1ppp/4pn2/8/1bPP4/1PN5/P3PPPP/R1BQKBNR b KQkq - 0 4")
        self.debug_print_board(board, "Nimzo-Penalized position before minimax:")
        score, move = minimax(board, self.depth, self.alpha, self.beta, True)
        board.push(move)
        self.debug_print_board(board, "Position after minimax move: " + str(move))
        self.assertEqual(move, chess.Move.from_uci("b4c3"))  # Bxc3


if __name__ == '__main__':
    unittest.main()