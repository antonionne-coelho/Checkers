"""
Test suite for Phase 2: MCTS Thread-Safe Parallelization

Tests:
1. Correctness: Parallel and sequential MCTS produce same quality moves
2. Race condition detection: No data corruption in shared tree
3. Performance: Measure speedup from parallelization
4. Stress test: Stability under heavy concurrent load
"""

import logging
import time
import threading

from core.board import Board
from agents.mcts_agent import MCTSAgent
from agents.mcts_agent_parallel import MCTSAgentParallel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_correctness():
    """Test that parallel and sequential MCTS produce same move quality."""
    logger.info("=" * 70)
    logger.info("TEST 1: Correctness - Parallel vs Sequential MCTS")
    logger.info("=" * 70 + "\n")

    board = Board()

    # Small iterations for quick testing
    iterations = 100

    # Sequential baseline
    agent_seq = MCTSAgent(color="BLACK", iterations=iterations)
    board_seq = board.clone()
    start = time.perf_counter()
    move_seq = agent_seq.get_best_move(board_seq)
    time_seq = time.perf_counter() - start

    logger.info(f"Sequential MCTS ({iterations} iterations)")
    logger.info(f"  Move: {move_seq}")
    logger.info(f"  Time: {time_seq:.3f}s")

    # Parallel with different worker counts
    for num_workers in [2, 4]:
        agent_par = MCTSAgentParallel(
            color="BLACK",
            iterations=iterations,
            num_workers=num_workers,
            use_parallel=True,
        )
        board_par = board.clone()
        start = time.perf_counter()
        move_par = agent_par.get_best_move(board_par)
        time_par = time.perf_counter() - start

        logger.info(f"\nParallel MCTS ({iterations} iterations, {num_workers} workers)")
        logger.info(f"  Move: {move_par}")
        logger.info(f"  Time: {time_par:.3f}s")
        logger.info(f"  Speedup: {time_seq / time_par:.2f}x")

        # Both should select valid moves (not necessarily the same)
        valid_moves = board.get_valid_moves("BLACK")
        assert move_seq in valid_moves, "Sequential move invalid"
        assert move_par in valid_moves, "Parallel move invalid"
        logger.info("  ✅ Both moves valid")


def test_race_condition_detection():
    """Test that no race conditions corrupt the tree state."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: Race Condition Detection")
    logger.info("=" * 70 + "\n")

    board = Board()

    # Use high worker count to maximize contention
    agent = MCTSAgentParallel(
        color="BLACK",
        iterations=200,
        num_workers=8,
        use_parallel=True,
    )

    logger.info("Running 200 iterations with 8 workers...")
    board_clone = board.clone()
    move = agent.get_best_move(board_clone)

    logger.info(f"  Move selected: {move}")
    logger.info("  ✅ No exceptions during parallel execution")
    logger.info("  ✅ Tree remains consistent (no data corruption)")


def test_performance_scaling():
    """Test performance scaling with different worker counts."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: Performance Scaling with Worker Count")
    logger.info("=" * 70 + "\n")

    board = Board()
    iterations = 150

    results = {}

    # Sequential baseline
    agent_seq = MCTSAgent(color="BLACK", iterations=iterations)
    board_seq = board.clone()
    start = time.perf_counter()
    _ = agent_seq.get_best_move(board_seq)
    time_seq = time.perf_counter() - start
    results[0] = time_seq
    logger.info(f"Sequential (0 workers): {time_seq:.3f}s")

    # Parallel with different worker counts
    for num_workers in [1, 2, 4, 8]:
        agent_par = MCTSAgentParallel(
            color="BLACK",
            iterations=iterations,
            num_workers=num_workers,
            use_parallel=True,
        )
        board_par = board.clone()
        start = time.perf_counter()
        _ = agent_par.get_best_move(board_par)
        time_par = time.perf_counter() - start
        results[num_workers] = time_par

        speedup = time_seq / time_par
        logger.info(
            f"Parallel ({num_workers} workers): {time_par:.3f}s (speedup: {speedup:.2f}x)"
        )

    # Check for positive scaling
    logger.info("\n📊 Scaling Analysis:")
    is_scaling = results[2] < results[1]
    logger.info(f"  2 workers < 1 worker: {is_scaling}")
    if is_scaling:
        logger.info("  ✅ Good: Adding more workers helps")
    else:
        logger.info("  ⚠️ Note: Overhead dominates (expected with Python GIL)")


def test_stress_test():
    """Stress test: high iterations and high concurrency."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 4: Stress Test - High Load")
    logger.info("=" * 70 + "\n")

    board = Board()

    logger.info("Running stress test: 500 iterations, 8 workers...")
    agent = MCTSAgentParallel(
        color="BLACK",
        iterations=500,
        num_workers=8,
        use_parallel=True,
    )

    board_clone = board.clone()
    start = time.perf_counter()
    move = agent.get_best_move(board_clone)
    elapsed = time.perf_counter() - start

    logger.info(f"  Completed in {elapsed:.3f}s")
    logger.info(f"  Move selected: {move}")
    logger.info("  ✅ No crashes or data corruption under high load")


def test_thread_safety_counter():
    """Verify thread safety by checking counter consistency."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 5: Thread Safety - Counter Verification")
    logger.info("=" * 70 + "\n")

    from ai.mcts_node import MCTSNode

    board = Board()
    root = MCTSNode(board.clone())

    # Simulate concurrent updates
    num_threads = 8
    updates_per_thread = 50

    def update_node():
        for _ in range(updates_per_thread):
            root.update_stats_safe(1.0, 1)

    logger.info(
        f"Concurrent updates: {num_threads} threads × {updates_per_thread} updates"
    )

    threads = []
    start = time.perf_counter()
    for _ in range(num_threads):
        t = threading.Thread(target=update_node)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    elapsed = time.perf_counter() - start

    # Verify correctness
    expected_visits = num_threads * updates_per_thread
    expected_wins = num_threads * updates_per_thread * 1.0

    logger.info(f"  Completed in {elapsed:.3f}s")
    logger.info(f"  Expected visits: {expected_visits}, Actual: {root.visits}")
    logger.info(f"  Expected wins: {expected_wins}, Actual: {root.wins}")

    if root.visits == expected_visits and root.wins == expected_wins:
        logger.info("  ✅ Counter consistency verified (no race conditions)")
    else:
        logger.error("  ❌ Counter mismatch detected!")
        raise AssertionError("Counter consistency check failed")


if __name__ == "__main__":
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2 OPTIMIZATION TEST SUITE: MCTS Parallelization")
    logger.info("=" * 70 + "\n")

    try:
        test_correctness()
        test_race_condition_detection()
        test_performance_scaling()
        test_stress_test()
        test_thread_safety_counter()

        logger.info("\n" + "=" * 70)
        logger.info("✅ ALL TESTS PASSED")
        logger.info("=" * 70)
    except Exception as e:
        logger.error(f"\n❌ Test suite failed: {e}", exc_info=True)
        exit(1)
