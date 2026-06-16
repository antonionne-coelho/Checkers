import unittest

from core.board import Board


class TestBoard(unittest.TestCase):
    def test_initial_piece_setup(self):
        board = Board()
        black_pieces = sum(1 for row in board.grid for piece in row if piece and piece.startswith("BLACK"))
        white_pieces = sum(1 for row in board.grid for piece in row if piece and piece.startswith("WHITE"))

        self.assertEqual(12, black_pieces)
        self.assertEqual(12, white_pieces)

    def test_initial_moves_not_empty(self):
        board = Board()
        self.assertTrue(len(board.get_valid_moves("BLACK")) > 0)
        self.assertTrue(len(board.get_valid_moves("WHITE")) > 0)
