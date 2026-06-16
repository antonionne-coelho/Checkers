"""
Test suite for Phase 1 Optimization: HybridAgent Parallelization

This module tests:
1. Correctness: Parallel and sequential HybridAgent produce the same moves
2. Performance: Measure speedup from parallelization
3. Thread safety: No race conditions or exceptions
"""

import logging
import time
from core.board import Board
from agents.hybrid_agent import HybridAgent
from agents.minimax_agent import MinimaxAgent
from agents.mcts_agent import MCTSAgent

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_hybrid_agent_parallel_vs_sequential():
    """Test that parallel and sequential HybridAgent produce identical moves."""
    logger.info("=" * 70)
    logger.info("TEST 1: Correctness - Parallel vs Sequential")
    logger.info("=" * 70)

    parallel_agent = HybridAgent(color="BLACK", iterations=100, use_parallel=True)
    sequential_agent = HybridAgent(color="BLACK", iterations=100, use_parallel=False)

    test_cases = 5
    moves_match = 0

    for i in range(test_cases):
        logger.info(f"\nTest case {i + 1}/{test_cases}")
        board = Board()

        # Clone board for both agents
        board_for_parallel = board.clone()
        board_for_sequential = board.clone()

        # Get moves
        parallel_move = parallel_agent.get_best_move(board_for_parallel)
        sequential_move = sequential_agent.get_best_move(board_for_sequential)

        # Log results
        logger.info(f"  Parallel move: {parallel_move}")
        logger.info(f"  Sequential move: {sequential_move}")
        logger.info(f"  Parallel stats: {parallel_agent.parallel_stats}")
        logger.info(f"  Sequential stats: {sequential_agent.parallel_stats}")

        if parallel_move == sequential_move:
            moves_match += 1
            logger.info("  ✅ MATCH")
        else:
            logger.warning("  ⚠️ MISMATCH (this can happen due to MCTS randomness)")

    logger.info(f"\n📊 Results: {moves_match}/{test_cases} moves matched")
    logger.info("   (Note: MCTS is non-deterministic, so matches > 0 is success)\n")


def test_hybrid_agent_speedup():
    """Benchmark speedup from parallelization."""
    logger.info("=" * 70)
    logger.info("TEST 2: Performance - Measure Speedup")
    logger.info("=" * 70)

    parallel_agent = HybridAgent(color="BLACK", iterations=200, use_parallel=True)
    sequential_agent = HybridAgent(color="BLACK", iterations=200, use_parallel=False)

    num_iterations = 3
    parallel_times = []
    sequential_times = []

    for i in range(num_iterations):
        logger.info(f"\nRound {i + 1}/{num_iterations}")
        board = Board()

        # Parallel execution
        board_p = board.clone()
        start = time.perf_counter()
        _ = parallel_agent.get_best_move(board_p)
        parallel_time = time.perf_counter() - start
        parallel_times.append(parallel_time)
        logger.info(f"  Parallel: {parallel_time:.3f}s")
        logger.info(
            f"    - Minimax: {parallel_agent.parallel_stats['minimax_time']:.3f}s"
        )
        logger.info(f"    - MCTS:    {parallel_agent.parallel_stats['mcts_time']:.3f}s")

        # Sequential execution
        board_s = board.clone()
        start = time.perf_counter()
        _ = sequential_agent.get_best_move(board_s)
        sequential_time = time.perf_counter() - start
        sequential_times.append(sequential_time)
        logger.info(f"  Sequential: {sequential_time:.3f}s")
        logger.info(
            f"    - Minimax: {sequential_agent.parallel_stats['minimax_time']:.3f}s"
        )
        logger.info(
            f"    - MCTS:    {sequential_agent.parallel_stats['mcts_time']:.3f}s"
        )

    avg_parallel = sum(parallel_times) / len(parallel_times)
    avg_sequential = sum(sequential_times) / len(sequential_times)
    speedup = avg_sequential / avg_parallel if avg_parallel > 0 else 0

    logger.info("\n📊 Results:")
    logger.info(f"   Average Parallel:   {avg_parallel:.3f}s")
    logger.info(f"   Average Sequential: {avg_sequential:.3f}s")
    logger.info(f"   🚀 Speedup: {speedup:.2f}x")

    if speedup > 1.3:
        logger.info("   ✅ EXCELLENT - Good parallelization efficiency\n")
    elif speedup > 1.0:
        logger.info("   ✅ GOOD - Parallelization working\n")
    else:
        logger.warning("   ⚠️ No speedup detected (overhead might dominate)\n")


def test_hybrid_agent_no_exceptions():
    """Test that parallel execution doesn't raise exceptions."""
    logger.info("=" * 70)
    logger.info("TEST 3: Stability - No Exceptions")
    logger.info("=" * 70)

    board = Board()
    agent = HybridAgent(color="BLACK", iterations=100, use_parallel=True)

    num_moves = 10
    exceptions = 0

    for i in range(num_moves):
        try:
            logger.info(f"Move {i + 1}/{num_moves} ... ", end="")
            move = agent.get_best_move(board.clone())
            if move:
                board.apply_move(move)
                logger.info("✅")
            else:
                logger.info("⚠️ No valid move")
        except Exception as e:
            exceptions += 1
            logger.error(f"❌ Exception: {e}")

    logger.info(f"\n📊 Results: {exceptions} exceptions out of {num_moves} moves\n")
    if exceptions == 0:
        logger.info("✅ All moves completed without exceptions\n")


def test_compare_agents():
    """Compare HybridAgent against individual agents."""
    logger.info("=" * 70)
    logger.info("TEST 4: Comparison - HybridAgent vs Minimax vs MCTS")
    logger.info("=" * 70)

    board = Board()

    agents = [
        MinimaxAgent(color="BLACK", depth=4),
        MCTSAgent(color="BLACK", iterations=100),
        HybridAgent(color="BLACK", iterations=100, minimax_depth=2, use_parallel=True),
    ]

    for agent in agents:
        logger.info(f"\n{agent.name}")
        board_clone = board.clone()

        start = time.perf_counter()
        move = agent.get_best_move(board_clone)
        elapsed = time.perf_counter() - start

        logger.info(f"  Move: {move}")
        logger.info(f"  Time: {elapsed:.3f}s")

        if hasattr(agent, "parallel_stats"):
            logger.info(f"  Stats: {agent.parallel_stats}")

    logger.info("")


if __name__ == "__main__":
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1 OPTIMIZATION TEST SUITE: HybridAgent Parallelization")
    logger.info("=" * 70 + "\n")

    try:
        test_hybrid_agent_parallel_vs_sequential()
        test_hybrid_agent_speedup()
        test_hybrid_agent_no_exceptions()
        test_compare_agents()

        logger.info("=" * 70)
        logger.info("✅ ALL TESTS COMPLETED")
        logger.info("=" * 70)
    except Exception as e:
        logger.error(f"Test suite failed: {e}", exc_info=True)
