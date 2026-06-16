import unittest

from agents.minimax_agent import MinimaxAgent
from agents.minimax_agent_parallel import MinimaxAgentParallel
from core.board import Board
from main import create_agent


class TestMinimaxParallel(unittest.TestCase):
    def test_returns_valid_move_on_initial_board(self):
        board = Board()
        agent = MinimaxAgentParallel(color="BLACK", depth=2, num_workers=2)
        move = agent.get_best_move(board.clone())
        self.assertIsNotNone(move)
        self.assertIn(move, board.get_valid_moves("BLACK"))

    def test_matches_sequential_minimax_tie_breaking(self):
        board = Board()
        sequential = MinimaxAgent(color="BLACK", depth=2)
        parallel = MinimaxAgentParallel(color="BLACK", depth=2, num_workers=2)

        self.assertEqual(
            parallel.get_best_move(board.clone()),
            sequential.get_best_move(board.clone()),
        )

    def test_multiple_worker_counts_return_valid_moves(self):
        board = Board()
        for workers in (1, 2, 4):
            with self.subTest(workers=workers):
                agent = MinimaxAgentParallel(
                    color="BLACK", depth=2, num_workers=workers
                )
                move = agent.get_best_move(board.clone())
                self.assertIsNotNone(move)
                self.assertIn(move, board.get_valid_moves("BLACK"))

    def test_create_agent_accepts_parallel_minimax_tokens(self):
        for token in ("minimax-parallel", "minimax_parallel"):
            with self.subTest(token=token):
                agent = create_agent(token, "BLACK", depth=2, iterations=10, workers=2)
                self.assertIsInstance(agent, MinimaxAgentParallel)
                self.assertEqual(agent.num_workers, 2)


if __name__ == "__main__":
    unittest.main()
