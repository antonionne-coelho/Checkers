import csv
import json
import os
from typing import Any


class MatchReport:
    """Generates reports and exports metrics collected from matches."""

    def __init__(self, metrics_collector: Any) -> None:
        self.metrics = metrics_collector

    def generate_console_report(self) -> None:
        print("=== Match Report ===")
        print(f"Matches played: {len(self.metrics.matches)}")
        print(f"Win rates: {self.metrics.get_win_rate()}")
        print(f"Average time per move: {self.metrics.get_average_time_per_move():.4f}s")
        print(f"Average states analyzed: {self.metrics.get_average_states_analyzed():.2f}")

    def export_to_csv(self, filepath: str) -> None:
        with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["match_index", "agent", "start", "end", "captures", "promotion", "elapsed_time", "states_analyzed", "winner"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for index, match in enumerate(self.metrics.matches, start=1):
                for move in match["moves"]:
                    writer.writerow(
                        {
                            "match_index": index,
                            "agent": move["agent"],
                            "start": move["move"]["start"],
                            "end": move["move"]["end"],
                            "captures": move["move"]["captures"],
                            "promotion": move["move"]["promotion"],
                            "elapsed_time": move["elapsed_time"],
                            "states_analyzed": move["states_analyzed"],
                            "winner": match["winner"],
                        }
                    )

    def export_learning_dataset_csv(self, filepath: str) -> None:
        self._ensure_parent_dir(filepath)
        with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "match_index",
                "ply",
                "agent",
                "player",
                "board_before",
                "board_after",
                "move_start",
                "move_end",
                "captured_positions",
                "capture_count",
                "promotion",
                "elapsed_time",
                "states_analyzed",
                "winner",
                "outcome_for_player",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for match_index, match in enumerate(self.metrics.matches, start=1):
                winner = match["winner"]
                for ply, move in enumerate(match["moves"], start=1):
                    player = move.get("player")
                    writer.writerow(
                        {
                            "match_index": match_index,
                            "ply": ply,
                            "agent": move["agent"],
                            "player": player,
                            "board_before": self._to_json(move.get("board_before")),
                            "board_after": self._to_json(move.get("board_after")),
                            "move_start": self._to_json(move["move"]["start"]),
                            "move_end": self._to_json(move["move"]["end"]),
                            "captured_positions": self._to_json(move["move"]["captures"]),
                            "capture_count": len(move["move"]["captures"] or []),
                            "promotion": move["move"]["promotion"],
                            "elapsed_time": move["elapsed_time"],
                            "states_analyzed": move["states_analyzed"],
                            "winner": winner,
                            "outcome_for_player": self._outcome_for_player(
                                winner, player
                            ),
                        }
                    )

    def export_to_json(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as jsonfile:
            json.dump(self.metrics.matches, jsonfile, indent=2)

    def _to_json(self, value: Any) -> str:
        return json.dumps(value, ensure_ascii=False)

    def _outcome_for_player(self, winner: Any, player: Any) -> float:
        if winner is None or player is None:
            return 0.5
        return 1.0 if winner == player else 0.0

    def _ensure_parent_dir(self, filepath: str) -> None:
        parent_dir = os.path.dirname(filepath)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
