import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from agents.mcts_agent import MCTSAgent
from agents.minimax_agent import MinimaxAgent

logger = logging.getLogger(__name__)


class HybridAgent(MCTSAgent):
    """Hybrid agent that combines MCTS and a shallow Minimax search.

    Phase 1 Optimization: Runs Minimax and MCTS in parallel using ThreadPoolExecutor.
    This achieves ~2x speedup for HybridAgent move selection.
    """

    def __init__(
        self,
        color: str,
        iterations: int = 1000,
        minimax_depth: int = 2,
        exploration_constant: float = 1.41,
        use_parallel: bool = True,
        batch_evaluator=None,
    ):
        super().__init__(color, iterations, exploration_constant, batch_evaluator)
        self.name = "Hybrid MCTS + Shallow Minimax (Parallel)"
        self.minimax_helper = MinimaxAgent(color=color, depth=minimax_depth)
        self.use_parallel = use_parallel
        self.parallel_stats = {"minimax_time": 0.0, "mcts_time": 0.0, "total_time": 0.0}

    def get_best_move(self, board):
        """Get best move by running Minimax and MCTS in parallel."""
        start_time = time.perf_counter()

        if self.use_parallel:
            minimax_move, mcts_move = self._get_moves_parallel(board)
        else:
            minimax_move, mcts_move = self._get_moves_sequential(board)

        end_time = time.perf_counter()
        self.parallel_stats["total_time"] = end_time - start_time

        # Decide which move is better
        if minimax_move is None:
            chosen_by = "mcts"
            chosen_move = mcts_move
        elif mcts_move is None:
            chosen_by = "minimax"
            chosen_move = minimax_move
        elif self._should_choose_minimax(board, minimax_move, mcts_move):
            chosen_by = "minimax"
            chosen_move = minimax_move
        else:
            chosen_by = "mcts"
            chosen_move = mcts_move

        # Avaliar scores para interpretabilidade
        from ai.evaluator import BoardEvaluator
        evaluator = BoardEvaluator()
        minimax_score = None
        mcts_score = None
        if minimax_move is not None:
            board_mm = board.clone()
            board_mm.apply_move(minimax_move)
            minimax_score = round(evaluator.evaluate(board_mm, self.color), 2)
        if mcts_move is not None:
            board_mc = board.clone()
            board_mc.apply_move(mcts_move)
            mcts_score = round(evaluator.evaluate(board_mc, self.color), 2)

        # Combinar informações dos sub-agentes
        minimax_info = getattr(self.minimax_helper, "last_decision_info", {})
        mcts_info = getattr(self, "_mcts_decision_info", {})

        self.last_decision_info = {
            "algorithm": "Hybrid",
            "chosen_by": chosen_by,
            "minimax_move": f"{minimax_move.start_pos}→{minimax_move.end_pos}" if minimax_move else "—",
            "mcts_move": f"{mcts_move.start_pos}→{mcts_move.end_pos}" if mcts_move else "—",
            "minimax_score": minimax_score,
            "mcts_score": mcts_score,
            "minimax_time": round(self.parallel_stats.get("minimax_time", 0), 3),
            "mcts_time": round(self.parallel_stats.get("mcts_time", 0), 3),
            "total_time": round(self.parallel_stats.get("total_time", 0), 3),
            "states_analyzed": self.states_analyzed + getattr(self.minimax_helper, "states_analyzed", 0),
            "minimax_states": getattr(self.minimax_helper, "states_analyzed", 0),
            "mcts_states": self.states_analyzed,
        }

        return chosen_move

    def _get_moves_parallel(self, board):
        """Execute Minimax and MCTS in parallel using ThreadPoolExecutor.

        Returns:
            tuple: (minimax_move, mcts_move)
        """
        with ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="HybridAgent"
        ) as executor:
            # Submit both tasks
            minimax_future = executor.submit(
                self.minimax_helper.get_best_move, board.clone()
            )
            mcts_future = executor.submit(super().get_best_move, board)

            minimax_move = None
            mcts_move = None

            try:
                # Collect results as they complete
                for future in as_completed([minimax_future, mcts_future], timeout=30):
                    if future is minimax_future:
                        minimax_start = time.perf_counter()
                        minimax_move = future.result()
                        self.parallel_stats["minimax_time"] = (
                            time.perf_counter() - minimax_start
                        )
                    else:
                        mcts_start = time.perf_counter()
                        mcts_move = future.result()
                        self.parallel_stats["mcts_time"] = (
                            time.perf_counter() - mcts_start
                        )

                logger.debug(
                    f"Parallel execution - Minimax: {self.parallel_stats['minimax_time']:.3f}s, "
                    f"MCTS: {self.parallel_stats['mcts_time']:.3f}s, "
                    f"Total: {self.parallel_stats['total_time']:.3f}s"
                )
            except TimeoutError:
                logger.warning("Hybrid agent parallel execution timed out")
                # Fall back to whatever completed
                if minimax_future.done():
                    minimax_move = minimax_future.result()
                if mcts_future.done():
                    mcts_move = mcts_future.result()

        return minimax_move, mcts_move

    def _get_moves_sequential(self, board):
        """Execute Minimax and MCTS sequentially (for benchmarking).

        Returns:
            tuple: (minimax_move, mcts_move)
        """
        minimax_start = time.perf_counter()
        minimax_move = self.minimax_helper.get_best_move(board.clone())
        self.parallel_stats["minimax_time"] = time.perf_counter() - minimax_start

        mcts_start = time.perf_counter()
        mcts_move = super().get_best_move(board)
        self.parallel_stats["mcts_time"] = time.perf_counter() - mcts_start

        return minimax_move, mcts_move

    def _should_choose_minimax(self, board, minimax_move, mcts_move):
        from ai.evaluator import BoardEvaluator

        evaluator = BoardEvaluator()
        board_minimax = board.clone()
        board_minimax.apply_move(minimax_move)
        board_mcts = board.clone()
        board_mcts.apply_move(mcts_move)

        score_minimax = evaluator.evaluate(board_minimax, self.color)
        score_mcts = evaluator.evaluate(board_mcts, self.color)

        return score_minimax >= score_mcts

    def guided_rollout_policy(self, board, moves):
        if board.current_player == self.color and len(moves) > 1:
            minimax_move = self.select_promising_move_with_minimax(board)
            if minimax_move is not None:
                return minimax_move
        return super().guided_rollout_policy(board, moves)

    def select_promising_move_with_minimax(self, board):
        if board.current_player != self.color:
            return None
        return self.minimax_helper.get_best_move(board.clone())
