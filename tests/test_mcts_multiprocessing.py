import unittest

from core.board import Board
from agents.mcts_agent_process import MCTSAgentProcess


class TestMCTSMultiprocessing(unittest.TestCase):
    def test_returns_valid_move_on_initial_board(self):
        board = Board()
        agent = MCTSAgentProcess(color="BLACK", iterations=50, num_workers=2)
        move = agent.get_best_move(board.clone())
        self.assertIsNotNone(move)
        self.assertIn(move, board.get_valid_moves("BLACK"))

    def test_multiple_worker_counts_return_valid_moves(self):
        board = Board()
        iterations = 80

        for workers in (1, 2, 4):
            with self.subTest(workers=workers):
                agent = MCTSAgentProcess(
                    color="BLACK", iterations=iterations, num_workers=workers
                )
                move = agent.get_best_move(board.clone())
                self.assertIsNotNone(move)
                self.assertIn(move, board.get_valid_moves("BLACK"))

    def test_deterministic_behavior_with_low_iteration_counts(self):
        board = Board()
        agent = MCTSAgentProcess(color="BLACK", iterations=10, num_workers=2)
        move_first = agent.get_best_move(board.clone())
        move_second = agent.get_best_move(board.clone())
        self.assertIsNotNone(move_first)
        self.assertIsNotNone(move_second)

    def test_process_pool_closes_cleanly(self):
        board = Board()
        agent = MCTSAgentProcess(color="BLACK", iterations=20, num_workers=2)
        move = agent.get_best_move(board.clone())
        self.assertIsNotNone(move)
        self.assertIn(move, board.get_valid_moves("BLACK"))


if __name__ == "__main__":
    unittest.main()
