import csv
import tempfile
import unittest
from pathlib import Path

from core.board import Board
from metrics.match_report import MatchReport
from metrics.metrics_collector import MetricsCollector


class TestLearningDataset(unittest.TestCase):
    def test_exports_move_level_dataset_with_outcome(self):
        board = Board()
        board_before = board.clone()
        move = board.get_valid_moves("BLACK")[0]
        board.apply_move(move)

        metrics = MetricsCollector()
        metrics.start_match(
            black_agent=type("AgentStub", (), {"name": "MCTS UCT"})(),
            white_agent=type("AgentStub", (), {"name": "Minimax Alpha-Beta"})(),
        )
        metrics.record_move(
            "MCTS UCT",
            move,
            elapsed_time=0.1,
            states_analyzed=12,
            player="BLACK",
            board_before=board_before,
            board_after=board.clone(),
        )
        metrics.end_match("BLACK")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "dataset.csv"
            MatchReport(metrics).export_learning_dataset_csv(str(output_path))

            with output_path.open(newline="", encoding="utf-8") as csvfile:
                rows = list(csv.DictReader(csvfile))

        self.assertEqual(1, len(rows))
        self.assertEqual("BLACK", rows[0]["player"])
        self.assertEqual("BLACK", rows[0]["winner"])
        self.assertEqual("1.0", rows[0]["outcome_for_player"])
        self.assertTrue(rows[0]["board_before"].startswith("[["))
        self.assertTrue(rows[0]["board_after"].startswith("[["))


if __name__ == "__main__":
    unittest.main()
