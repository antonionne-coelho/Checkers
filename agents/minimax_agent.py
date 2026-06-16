from math import inf

from agents.base_agent import Agent
from ai.evaluator import BoardEvaluator


class MinimaxAgent(Agent):
    """Agent that uses Minimax search with alpha-beta pruning."""

    def __init__(self, color: str, depth: int = 4):
        """Initialize the Minimax agent.

        Args:
            color: Agent color, either 'BLACK' or 'WHITE'.
            depth: Search depth for the Minimax algorithm.
        """
        super().__init__("Minimax Alpha-Beta", color)
        self.depth = depth
        self.states_analyzed = 0
        self.evaluator = BoardEvaluator()

    def get_best_move(self, board):
        moves = board.get_valid_moves(board.current_player)
        if not moves:
            return None

        self.states_analyzed = 0
        best_move = None
        best_score = -inf
        maximizing = board.current_player == self.color

        move_scores = []

        for move in moves:
            next_board = board.clone()
            next_board.apply_move(move)
            score = self.minimax(
                next_board,
                self.depth - 1,
                -inf,
                inf,
                not maximizing,
            )
            move_scores.append((move, score))
            if score > best_score or best_move is None:
                best_score = score
                best_move = move

        # Ordenar candidatos por score (melhor primeiro)
        move_scores.sort(key=lambda x: x[1], reverse=True)
        candidates = [
            {
                "start": m.start_pos,
                "end": m.end_pos,
                "score": round(s, 2),
                "captures": len(m.captured_positions) if m.captured_positions else 0,
            }
            for m, s in move_scores[:5]
        ]

        self.last_decision_info = {
            "algorithm": "Minimax α-β",
            "depth": self.depth,
            "best_score": round(best_score, 2),
            "states_analyzed": self.states_analyzed,
            "total_candidates": len(moves),
            "candidates": candidates,
        }

        return best_move

    def minimax(self, board, depth, alpha, beta, maximizing_player):
        self.states_analyzed += 1
        if depth == 0 or board.is_terminal():
            return self.evaluate_position(board)

        valid_moves = board.get_valid_moves(board.current_player)
        if not valid_moves:
            return self.evaluate_position(board)

        if maximizing_player:
            value = -inf
            for move in valid_moves:
                next_board = board.clone()
                next_board.apply_move(move)
                value = max(
                    value,
                    self.minimax(next_board, depth - 1, alpha, beta, False),
                )
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value

        value = inf
        for move in valid_moves:
            next_board = board.clone()
            next_board.apply_move(move)
            value = min(
                value,
                self.minimax(next_board, depth - 1, alpha, beta, True),
            )
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value

    def evaluate_position(self, board):
        return self.evaluator.evaluate(board, self.color)
