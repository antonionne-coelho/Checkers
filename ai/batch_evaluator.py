from abc import ABC, abstractmethod
from typing import Iterable

from ai.evaluator import BoardEvaluator


class BatchEvaluator(ABC):
    """Interface for board evaluators that can score positions in batches."""

    name = "batch-evaluator"

    @abstractmethod
    def evaluate_batch(self, boards: Iterable, color: str) -> list[float]:
        """Return one score per board for the requested color."""

    def evaluate_move(self, board, move, color: str) -> float:
        return self.evaluate_moves(board, [move], color)[0]

    def evaluate_moves(self, board, moves: Iterable, color: str) -> list[float]:
        next_boards = []
        for move in moves:
            next_board = board.clone()
            next_board.apply_move(move)
            next_boards.append(next_board)
        return self.evaluate_batch(next_boards, color)


class PythonBatchEvaluator(BatchEvaluator):
    """Pure-Python reference batch evaluator.

    This keeps today's heuristic behavior while giving search agents a stable
    batch API that future GPU backends can implement.
    """

    name = "python"

    def __init__(self, evaluator: BoardEvaluator | None = None):
        self.evaluator = evaluator or BoardEvaluator()

    def evaluate_batch(self, boards: Iterable, color: str) -> list[float]:
        return [self.evaluator.evaluate(board, color) for board in boards]
