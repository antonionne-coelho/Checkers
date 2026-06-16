import logging
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from agents.base_agent import Agent
from ai.mcts_worker import run_mcts_simulations
from core.move import Move

logger = logging.getLogger(__name__)


def _move_key(move: Move):
    return (
        move.start_pos,
        move.end_pos,
        tuple(move.captured_positions),
        tuple(move.captured_pieces),
        move.piece_moved,
        move.is_promotion,
    )


class MCTSAgentProcess(Agent):
    """MCTS Agent using multiprocessing with a shared tree root."""

    def __init__(
        self,
        color: str,
        iterations: int = 1000,
        exploration_constant: float = 1.41,
        num_workers: int = 4,
    ):
        super().__init__("MCTS UCT (Process)", color)
        self.iterations = max(1, iterations)
        self.exploration_constant = exploration_constant
        self.num_workers = max(1, min(num_workers, 16))
        self.process_stats = {
            "total_time": 0.0,
            "num_workers": self.num_workers,
            "iterations_per_worker": 0,
        }

    def get_best_move(self, board):
        moves = board.get_valid_moves(board.current_player)
        if not moves:
            return None

        # Use root-parallel workers that run independent MCTS simulations and
        # aggregate root-children visit counts.
        self.process_stats["iterations_per_worker"] = (
            self.iterations // self.num_workers
        )
        worker_iterations = self.iterations // self.num_workers
        remaining = self.iterations % self.num_workers

        start_time = time.perf_counter()
        visit_counts = {}
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for worker_id in range(self.num_workers):
                iterations_for_worker = worker_iterations + (
                    1 if worker_id < remaining else 0
                )
                if iterations_for_worker == 0:
                    continue
                futures.append(
                    executor.submit(
                        run_mcts_simulations,
                        board.clone(),
                        iterations_for_worker,
                        self.exploration_constant,
                        random.randrange(2**32),
                    )
                )

            for future in as_completed(futures, timeout=120):
                try:
                    result = future.result()
                except Exception:
                    logger.error("MCTS worker process failed", exc_info=True)
                    raise

                for move, visits in result:
                    key = _move_key(move)
                    visit_counts[key] = visit_counts.get(key, 0) + visits

        self.process_stats["total_time"] = time.perf_counter() - start_time

        if not visit_counts:
            return random.choice(moves)

        # find move object for max visits
        best_key = max(visit_counts.items(), key=lambda kv: kv[1])[0]
        for m in moves:
            if _move_key(m) == best_key:
                return m
        return random.choice(moves)
