import unittest

from agents.mcts_agent import MCTSAgent
from main import create_agent
from ai.batch_evaluator import PythonBatchEvaluator
from ai.evaluator import BoardEvaluator
from core.board import Board


class SpyBatchEvaluator(PythonBatchEvaluator):
    def __init__(self):
        super().__init__()
        self.batch_calls = 0

    def evaluate_batch(self, boards, color):
        self.batch_calls += 1
        return super().evaluate_batch(boards, color)


class TestBatchEvaluator(unittest.TestCase):
    def test_evaluate_batch_matches_board_evaluator(self):
        board = Board()
        move = board.get_valid_moves("BLACK")[0]
        next_board = board.clone()
        next_board.apply_move(move)

        evaluator = BoardEvaluator()
        batch_evaluator = PythonBatchEvaluator(evaluator)

        self.assertEqual(
            [
                evaluator.evaluate(board, "BLACK"),
                evaluator.evaluate(next_board, "BLACK"),
            ],
            batch_evaluator.evaluate_batch([board, next_board], "BLACK"),
        )

    def test_evaluate_moves_matches_manual_next_boards(self):
        board = Board()
        moves = board.get_valid_moves("BLACK")
        evaluator = BoardEvaluator()
        batch_evaluator = PythonBatchEvaluator(evaluator)

        expected_scores = []
        for move in moves:
            next_board = board.clone()
            next_board.apply_move(move)
            expected_scores.append(evaluator.evaluate(next_board, "BLACK"))

        self.assertEqual(
            expected_scores,
            batch_evaluator.evaluate_moves(board, moves, "BLACK"),
        )

    def test_mcts_uses_injected_batch_evaluator(self):
        board = Board()
        batch_evaluator = SpyBatchEvaluator()
        agent = MCTSAgent(
            color="BLACK",
            iterations=5,
            batch_evaluator=batch_evaluator,
        )

        move = agent.get_best_move(board.clone())

        self.assertIsNotNone(move)
        self.assertGreater(batch_evaluator.batch_calls, 0)

    def test_create_agent_passes_batch_evaluator_to_mcts(self):
        batch_evaluator = SpyBatchEvaluator()
        agent = create_agent(
            "mcts",
            "BLACK",
            depth=2,
            iterations=5,
            batch_evaluator=batch_evaluator,
        )

        agent.get_best_move(Board())

        self.assertGreater(batch_evaluator.batch_calls, 0)


if __name__ == "__main__":
    unittest.main()
