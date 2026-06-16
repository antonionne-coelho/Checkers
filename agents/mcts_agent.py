import random

from agents.base_agent import Agent
from ai.batch_evaluator import PythonBatchEvaluator
from ai.evaluator import BoardEvaluator
from ai.mcts_node import MCTSNode


class MCTSAgent(Agent):
    """Agent that uses Monte Carlo Tree Search (MCTS) with UCT selection."""

    def __init__(
        self,
        color: str,
        iterations: int = 1000,
        exploration_constant: float = 1.41,
        batch_evaluator=None,
    ):
        """Initialize the MCTS agent.

        Args:
            color: Agent color, either 'BLACK' or 'WHITE'.
            iterations: Number of MCTS iterations to run per turn.
            exploration_constant: Exploration weight used in UCT selection.
        """
        super().__init__("MCTS UCT", color)
        self.iterations = iterations
        self.exploration_constant = exploration_constant
        self.states_analyzed = 0
        self.evaluator = BoardEvaluator()
        self.batch_evaluator = batch_evaluator or PythonBatchEvaluator(self.evaluator)

    def get_best_move(self, board):
        moves = board.get_valid_moves(board.current_player)
        if not moves:
            return None

        self.states_analyzed = 0
        root = MCTSNode(board.clone())

        for _ in range(self.iterations):
            node = self.selection(root)
            if not node.board.is_terminal():
                node = self.expansion(node)
            result = self.simulation(node.board)
            self.backpropagation(node, result)

        best_child = max(root.children, key=lambda child: child.visits, default=None)

        # Coletar informações de decisão dos filhos raiz
        children_stats = []
        for child in sorted(root.children, key=lambda c: c.visits, reverse=True):
            winrate = child.wins / child.visits if child.visits > 0 else 0.0
            children_stats.append({
                "start": child.move.start_pos if child.move else None,
                "end": child.move.end_pos if child.move else None,
                "visits": child.visits,
                "winrate": round(winrate, 3),
                "captures": len(child.move.captured_positions) if child.move and child.move.captured_positions else 0,
            })

        best_winrate = best_child.wins / best_child.visits if best_child and best_child.visits > 0 else 0.0
        self.last_decision_info = {
            "algorithm": "MCTS UCT",
            "iterations": self.iterations,
            "best_visits": best_child.visits if best_child else 0,
            "best_winrate": round(best_winrate, 3),
            "states_analyzed": self.states_analyzed,
            "total_candidates": len(root.children),
            "candidates": children_stats[:5],
        }

        return best_child.move if best_child else random.choice(moves)

    def selection(self, node):
        while not node.board.is_terminal() and node.is_fully_expanded():
            next_node = node.best_child(self.exploration_constant)
            if next_node is None:
                break
            node = next_node
            self.states_analyzed += 1
        return node

    def expansion(self, node):
        if node.is_fully_expanded() or node.board.is_terminal():
            return node

        move = random.choice(node.untried_moves)
        board_copy = node.board.clone()
        board_copy.apply_move(move)
        prior = self.compute_prior(node.board, move)
        child = node.add_child(move, board_copy, prior=prior)
        self.states_analyzed += 1
        return child

    def simulation(self, board):
        current_board = board.clone()
        move_depth = 0
        while not current_board.is_terminal() and move_depth < 10:
            moves = current_board.get_valid_moves(current_board.current_player)
            if not moves:
                break
            move = self.guided_rollout_policy(current_board, moves)
            current_board.apply_move(move)
            self.states_analyzed += 1
            move_depth += 1

        if current_board.is_terminal():
            winner = current_board.get_winner()
            return 1 if winner == self.color else 0

        return self.value_estimate(current_board)

    def guided_rollout_policy(self, board, moves):
        capture_moves = [move for move in moves if move.captured_positions]
        if capture_moves:
            capture_moves.sort(key=lambda move: len(move.captured_positions), reverse=True)
            best_capture_moves = [
                move for move in capture_moves
                if len(move.captured_positions) == len(capture_moves[0].captured_positions)
            ]
            if len(best_capture_moves) == 1:
                return best_capture_moves[0]
            moves = best_capture_moves

        scores = self.batch_evaluator.evaluate_moves(board, moves, board.current_player)
        best_score = float("-inf")
        best_move = None
        for move, score in zip(moves, scores):
            if score > best_score:
                best_score = score
                best_move = move

        return best_move if best_move is not None else random.choice(moves)

    def compute_prior(self, board, move):
        score = self.batch_evaluator.evaluate_move(board, move, board.current_player)
        prior = 1.0 + score / 20.0
        return max(0.01, prior)

    def value_estimate(self, board):
        raw_score = self.batch_evaluator.evaluate_batch([board], self.color)[0]
        value = 0.5 + raw_score / 2000.0
        return max(0.0, min(1.0, value))

    def backpropagation(self, node, result):
        while node is not None:
            node.visits += 1
            node.wins += result
            node = node.parent
