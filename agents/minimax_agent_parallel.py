import logging
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import inf

from agents.minimax_agent import MinimaxAgent
from ai.minimax_worker import evaluate_minimax_move

logger = logging.getLogger(__name__)


class MinimaxAgentParallel(MinimaxAgent):
    """Minimax agent that evaluates root moves in parallel processes."""

    def __init__(
        self,
        color: str,
        depth: int = 4,
        num_workers: int = 4,
        use_parallel: bool = True,
    ):
        super().__init__(color=color, depth=depth)
        self.name = "Minimax Alpha-Beta (Parallel)"
        self.num_workers = max(1, min(num_workers, 16))
        self.use_parallel = use_parallel
        self.parallel_stats = {
            "total_time": 0.0,
            "num_workers": self.num_workers,
            "root_moves": 0,
        }

    def get_best_move(self, board):
        moves = board.get_valid_moves(board.current_player)
        if not moves:
            return None

        self.states_analyzed = 0
        self.parallel_stats["root_moves"] = len(moves)

        if not self.use_parallel or len(moves) == 1:
            return super().get_best_move(board)

        start_time = time.perf_counter()
        maximizing = board.current_player == self.color
        root_state = board.clone()
        results = {}

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(
                    evaluate_minimax_move,
                    root_state,
                    move,
                    self.color,
                    self.depth,
                    maximizing,
                ): index
                for index, move in enumerate(moves)
            }

            for future in as_completed(futures, timeout=120):
                index = futures[future]
                try:
                    move, score, states_analyzed = future.result()
                except Exception:
                    logger.error("Minimax worker process failed", exc_info=True)
                    raise

                results[index] = (move, score)
                self.states_analyzed += states_analyzed

        self.parallel_stats["total_time"] = time.perf_counter() - start_time

        if not results:
            return random.choice(moves)

        best_move = None
        best_score = -inf
        for index in sorted(results):
            move, score = results[index]
            if score > best_score or best_move is None:
                best_score = score
                best_move = move

        return best_move
