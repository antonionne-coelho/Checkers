import statistics
from typing import Any, Optional


class MetricsCollector:
    """Collects and stores statistics for played matches."""

    def __init__(self):
        self.matches: list[dict[str, Any]] = []
        self.current_match: Optional[dict[str, Any]] = None

    def start_match(self, black_agent: Any, white_agent: Any) -> None:
        self.current_match = {
            "black_agent": black_agent.name,
            "white_agent": white_agent.name,
            "moves": [],
            "winner": None,
        }

    def record_move(
        self,
        agent_name: str,
        move: Any,
        elapsed_time: float,
        states_analyzed: int,
        player: Optional[str] = None,
        board_before: Any = None,
        board_after: Any = None,
    ) -> None:
        if self.current_match is None:
            return

        self.current_match["moves"].append(
            {
                "agent": agent_name,
                "player": player,
                "board_before": self._serialize_board(board_before),
                "board_after": self._serialize_board(board_after),
                "move": {
                    "start": move.start_pos,
                    "end": move.end_pos,
                    "captures": move.captured_positions,
                    "promotion": move.is_promotion,
                },
                "elapsed_time": elapsed_time,
                "states_analyzed": states_analyzed,
            }
        )

    def _serialize_board(self, board: Any) -> Optional[list[list[str | None]]]:
        if board is None:
            return None
        return [row.copy() for row in board.grid]

    def end_match(self, winner: Optional[str]) -> None:
        if self.current_match is None:
            return
        self.current_match["winner"] = winner
        self.matches.append(self.current_match)
        self.current_match = None

    def get_win_rate(self) -> dict[str, float]:
        if not self.matches:
            return {}

        results: dict[str, int] = {}
        for match in self.matches:
            winner = match["winner"]
            if winner is None:
                continue
            results[winner] = results.get(winner, 0) + 1

        return {
            color: results.get(color, 0) / len(self.matches)
            for color in ["BLACK", "WHITE"]
        }

    def get_average_time_per_move(self) -> float:
        timings = [move["elapsed_time"] for match in self.matches for move in match["moves"]]
        return statistics.mean(timings) if timings else 0.0

    def get_average_states_analyzed(self) -> float:
        values = [move["states_analyzed"] for match in self.matches for move in match["moves"]]
        return statistics.mean(values) if values else 0.0
