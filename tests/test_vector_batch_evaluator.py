import unittest

from ai.board_encoder import encode_board_flat, encode_boards_flat
from ai.evaluator import BoardEvaluator
from ai.vector_batch_evaluator import VectorizedCPUBatchEvaluator
from core.board import Board


class TestVectorBatchEvaluator(unittest.TestCase):
    def test_encode_board_flat_uses_64_numeric_values(self):
        board = Board()
        encoded = encode_board_flat(board)

        self.assertEqual(64, len(encoded))
        self.assertTrue(all(isinstance(value, int) for value in encoded))
        self.assertIn(1, encoded)
        self.assertIn(-1, encoded)

    def test_encode_boards_flat_preserves_batch_size(self):
        boards = [Board(), Board()]

        encoded = encode_boards_flat(boards)

        self.assertEqual(2, len(encoded))
        self.assertTrue(all(len(row) == 64 for row in encoded))

    def test_vector_scores_match_board_evaluator(self):
        board = Board()
        move = board.get_valid_moves("BLACK")[0]
        next_board = board.clone()
        next_board.apply_move(move)

        reference = BoardEvaluator()
        vector_evaluator = VectorizedCPUBatchEvaluator()

        self.assertEqual(
            [
                reference.evaluate(board, "BLACK"),
                reference.evaluate(next_board, "BLACK"),
            ],
            vector_evaluator.evaluate_batch([board, next_board], "BLACK"),
        )

    def test_evaluate_moves_matches_reference_scores(self):
        board = Board()
        moves = board.get_valid_moves("BLACK")
        reference = BoardEvaluator()
        vector_evaluator = VectorizedCPUBatchEvaluator()

        expected_scores = []
        for move in moves:
            next_board = board.clone()
            next_board.apply_move(move)
            expected_scores.append(reference.evaluate(next_board, "BLACK"))

        self.assertEqual(
            expected_scores,
            vector_evaluator.evaluate_moves(board, moves, "BLACK"),
        )


if __name__ == "__main__":
    unittest.main()
