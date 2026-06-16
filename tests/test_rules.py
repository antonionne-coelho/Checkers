import unittest

from core.board import Board
from core.rules import get_valid_moves


class TestRules(unittest.TestCase):
    def test_mandatory_capture(self):
        board = Board()
        board.grid = [[None for _ in range(8)] for _ in range(8)]
        board.grid[2][3] = "BLACK"
        board.grid[3][4] = "WHITE"
        board.current_player = "BLACK"

        moves = get_valid_moves(board, "BLACK")
        self.assertEqual(1, len(moves))
        move = moves[0]
        self.assertEqual((4, 5), move.end_pos)
        self.assertEqual([(3, 4)], move.captured_positions)
