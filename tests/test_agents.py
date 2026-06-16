import unittest

from core.board import Board
from agents.minimax_agent import MinimaxAgent
from agents.mcts_agent import MCTSAgent


class TestAgents(unittest.TestCase):
    def test_minimax_returns_move_on_initial_position(self):
        board = Board()
        agent = MinimaxAgent(color="BLACK", depth=2)
        move = agent.get_best_move(board.clone())
        self.assertIsNotNone(move)

    def test_mcts_returns_move_on_initial_position(self):
        board = Board()
        agent = MCTSAgent(color="BLACK", iterations=50)
        move = agent.get_best_move(board.clone())
        self.assertIsNotNone(move)
