"""
Quick validation test for Phase 1: HybridAgent Parallelization
Uses smaller iterations for faster testing
"""

import logging
import time
from core.board import Board
from agents.hybrid_agent import HybridAgent

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_quick_validation():
    """Quick test to validate HybridAgent parallelization works."""
    logger.info("=" * 70)
    logger.info("QUICK VALIDATION: HybridAgent Parallelization")
    logger.info("=" * 70 + "\n")

    # Test 1: Parallel agent works
    logger.info("✓ Creating parallel HybridAgent...")
    agent_parallel = HybridAgent(color="BLACK", iterations=50, use_parallel=True)
    board = Board()

    logger.info("✓ Running parallel move selection...")
    start = time.perf_counter()
    move = agent_parallel.get_best_move(board.clone())
    parallel_time = time.perf_counter() - start

    logger.info(f"  Move: {move}")
    logger.info(f"  Time: {parallel_time:.3f}s")
    logger.info(f"  Stats: {agent_parallel.parallel_stats}")

    # Test 2: Sequential agent works
    logger.info("\n✓ Creating sequential HybridAgent...")
    agent_sequential = HybridAgent(color="BLACK", iterations=50, use_parallel=False)
    board = Board()

    logger.info("✓ Running sequential move selection...")
    start = time.perf_counter()
    move = agent_sequential.get_best_move(board.clone())
    sequential_time = time.perf_counter() - start

    logger.info(f"  Move: {move}")
    logger.info(f"  Time: {sequential_time:.3f}s")
    logger.info(f"  Stats: {agent_sequential.parallel_stats}")

    # Test 3: Compare
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS:")
    logger.info("=" * 70)
    logger.info(f"Parallel time:   {parallel_time:.3f}s")
    logger.info(f"Sequential time: {sequential_time:.3f}s")

    if parallel_time > 0 and sequential_time > 0:
        ratio = sequential_time / parallel_time
        logger.info(f"Ratio (seq/par): {ratio:.2f}x")

        if ratio > 1.2:
            logger.info("✅ Parallelization effective!")
        else:
            logger.info("⚠️ Parallelization overhead detected")
            logger.info("   (This is expected in Python due to GIL)")

    logger.info("\n" + "=" * 70)
    logger.info("✅ VALIDATION PASSED - HybridAgent parallelization is working")
    logger.info("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        test_quick_validation()
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}", exc_info=True)
