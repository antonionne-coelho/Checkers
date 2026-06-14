import random
import time
from dataclasses import dataclass
from typing import Iterable

from ai.batch_evaluator import PythonBatchEvaluator
from ai.batch_evaluator_factory import create_batch_evaluator
from core.board import Board


@dataclass(frozen=True)
class BatchEvaluatorBenchmarkResult:
    backend: str
    positions: int
    evaluations: int
    elapsed_seconds: float
    average_ms_per_evaluation: float
    score_checksum: float
    max_delta_from_python: float
    mismatch_count: int


def generate_benchmark_boards(position_count: int, seed: int = 0) -> list[Board]:
    """Generate deterministic board positions through random legal play."""
    if position_count < 1:
        raise ValueError("position_count must be at least 1")

    rng = random.Random(seed)
    boards = []
    board = Board()

    while len(boards) < position_count:
        boards.append(board.clone())
        if len(boards) >= position_count:
            break

        moves = board.get_valid_moves(board.current_player)
        if board.is_terminal() or not moves:
            board = Board()
            continue

        board.apply_move(rng.choice(moves))

    return boards


def benchmark_batch_evaluators(
    backends: Iterable[str],
    position_count: int = 32,
    seed: int = 0,
    colors: tuple[str, ...] = ("BLACK", "WHITE"),
    tolerance: float = 1e-9,
) -> list[BatchEvaluatorBenchmarkResult]:
    boards = generate_benchmark_boards(position_count, seed)
    reference_scores = _evaluate_backend(PythonBatchEvaluator(), boards, colors)

    results = []
    for backend in backends:
        normalized_backend = backend.strip().lower().replace("_", "-")
        evaluator = create_batch_evaluator(
            backend=normalized_backend,
            gpu_enabled=False,
            strict_gpu=False,
        )

        start = time.perf_counter()
        scores = _evaluate_backend(evaluator, boards, colors)
        elapsed = time.perf_counter() - start

        deltas = [
            abs(score - reference)
            for score, reference in zip(scores, reference_scores)
        ]
        evaluations = len(scores)
        results.append(
            BatchEvaluatorBenchmarkResult(
                backend=normalized_backend,
                positions=len(boards),
                evaluations=evaluations,
                elapsed_seconds=elapsed,
                average_ms_per_evaluation=(
                    (elapsed / evaluations) * 1000.0 if evaluations else 0.0
                ),
                score_checksum=sum(scores),
                max_delta_from_python=max(deltas, default=0.0),
                mismatch_count=sum(1 for delta in deltas if delta > tolerance),
            )
        )

    return results


def format_benchmark_results(
    results: Iterable[BatchEvaluatorBenchmarkResult],
) -> str:
    rows = [
        "backend      positions  evals  total_s   ms/eval  max_delta  mismatches  checksum",
        "------------ ---------- ------ -------- -------- ---------- ----------- --------",
    ]
    for result in results:
        rows.append(
            f"{result.backend:<12} "
            f"{result.positions:>10} "
            f"{result.evaluations:>6} "
            f"{result.elapsed_seconds:>8.4f} "
            f"{result.average_ms_per_evaluation:>8.4f} "
            f"{result.max_delta_from_python:>10.6f} "
            f"{result.mismatch_count:>11} "
            f"{result.score_checksum:>8.2f}"
        )
    return "\n".join(rows)


def _evaluate_backend(evaluator, boards: list[Board], colors: tuple[str, ...]) -> list[float]:
    scores = []
    for color in colors:
        scores.extend(evaluator.evaluate_batch(boards, color))
    return scores
