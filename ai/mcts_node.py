from __future__ import annotations

import math
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.board import Board
    from core.move import Move


class MCTSNode:
    """Represents a node in the Monte Carlo Tree Search tree."""

    def __init__(
        self,
        board: Board,
        parent: MCTSNode | None = None,
        move: Move | None = None,
        prior: float = 1.0,
    ):
        self.board = board
        self.parent = parent
        self.move = move
        self.children: list[MCTSNode] = []
        self.visits = 0
        self.wins = 0.0
        self.prior = prior
        self.untried_moves = board.get_valid_moves(board.current_player)

        # Phase 2 Optimization: Thread-safety locks
        self.stats_lock = threading.RLock()  # Protege wins/visits (reentrant)
        self.children_lock = threading.Lock()  # Protege children e untried_moves

    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

    def best_child(self, exploration_constant: float) -> MCTSNode | None:
        best_score = float("-inf")
        best_node = None

        for child in self.children:
            q_value = child.wins / child.visits if child.visits > 0 else 0.0
            u_value = (
                exploration_constant
                * child.prior
                * math.sqrt(max(1.0, self.visits))
                / (1 + child.visits)
            )
            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_node = child

        return best_node

    def add_child(self, move: Move, board: Board, prior: float = 1.0) -> MCTSNode:
        child = MCTSNode(board, parent=self, move=move, prior=prior)
        self.children.append(child)
        self.untried_moves = [m for m in self.untried_moves if m != move]
        return child

    def get_board_clone(self) -> Board:
        """Return a deep clone of the board for safe simulation."""
        return self.board.clone()

    def get_move(self) -> Move | None:
        return self.move

    def get_visits(self) -> int:
        return self.visits

    def get_wins(self) -> float:
        return self.wins

    def is_terminal(self) -> bool:
        return self.board.is_terminal()

    def get_current_player(self) -> str:
        return self.board.current_player

    # ============================================================================
    # Phase 2 Optimization: Thread-Safe Methods for Parallel MCTS
    # ============================================================================

    def update_stats_safe(self, wins_delta: float, visits_delta: int) -> None:
        """Thread-safe update of wins and visits statistics.

        Args:
            wins_delta: Value to add to wins
            visits_delta: Increment visits count (usually 1)
        """
        with self.stats_lock:
            self.wins += wins_delta
            self.visits += visits_delta

    def add_child_safe(self, move: Move, board: Board, prior: float = 1.0) -> MCTSNode:
        """Thread-safe child addition.

        Args:
            move: Move that leads to this child
            board: Board state at this child
            prior: Prior probability for this move

        Returns:
            The newly created child node
        """
        with self.children_lock:
            child = self.add_child(move, board, prior)
        return child

    def best_child_safe(self, exploration_constant: float) -> MCTSNode | None:
        """Thread-safe best child selection using UCT formula.

        Args:
            exploration_constant: C parameter for exploration

        Returns:
            Best child node according to UCT, or None if no children
        """
        with self.children_lock:
            best = self.best_child(exploration_constant)
        return best

    def is_fully_expanded_safe(self) -> bool:
        """Thread-safe check if node is fully expanded."""
        with self.children_lock:
            return self.is_fully_expanded()

    def get_untried_move_safe(self) -> Move | None:
        """Thread-safe selection of an untried move.

        Returns:
            A random untried move, or None if no untried moves
        """
        import random

        with self.children_lock:
            if not self.untried_moves:
                return None
            move = random.choice(self.untried_moves)
            # Note: Don't remove here - removal happens in add_child
        return move

    def get_best_move_safe(self) -> Move | None:
        """Get the best move by visit count (for final move selection).

        Returns:
            The move with highest visit count among children
        """
        with self.children_lock:
            if not self.children:
                return None
            best_child = max(self.children, key=lambda child: child.visits)
        return best_child.move if best_child else None

    def get_stats_safe(self) -> tuple:
        """Thread-safe read of statistics.

        Returns:
            Tuple of (visits, wins)
        """
        with self.stats_lock:
            return (self.visits, self.wins)
