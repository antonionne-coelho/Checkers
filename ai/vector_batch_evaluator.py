from ai.batch_evaluator import BatchEvaluator
from ai.board_encoder import encode_boards_flat
from ai.evaluator import (
    CAPTURE_WEIGHT,
    KING_VALUE,
    KING_WEIGHT,
    MAN_VALUE,
    MATERIAL_WEIGHT,
    MOBILITY_WEIGHT,
    POSITION_WEIGHT,
    PROMOTION_WEIGHT,
    king_positional_value,
    man_positional_value,
    promotion_potential_value,
)
from core.rules import is_king, piece_color


class VectorizedCPUBatchEvaluator(BatchEvaluator):
    """CPU evaluator backed by numeric board vectors.

    It mirrors `BoardEvaluator` scores while moving material, position and king
    features to a flat numeric representation. Mobility, capture potential and
    promotion potential still use rule generation because those features depend
    on legal moves.
    """

    name = "cpu-vector"

    def evaluate_batch(self, boards, color: str) -> list[float]:
        board_list = list(boards)
        encoded_boards = encode_boards_flat(board_list)
        return [
            self._evaluate_encoded(board, encoded, color)
            for board, encoded in zip(board_list, encoded_boards)
        ]

    def _evaluate_encoded(self, board, encoded: list[int], color: str) -> float:
        winner = board.get_winner()
        if winner == color:
            return 1000.0
        if winner is not None and winner != color:
            return -1000.0

        opponent_color = "WHITE" if color == "BLACK" else "BLACK"
        material = self._material_score(encoded, color) - self._material_score(
            encoded, opponent_color
        )
        mobility = self._mobility_score(board, color) - self._mobility_score(
            board, opponent_color
        )
        position = self._positional_score(encoded, color) - self._positional_score(
            encoded, opponent_color
        )
        king_value = self._king_score(encoded, color) - self._king_score(
            encoded, opponent_color
        )
        capture_potential = self._capture_potential_score(
            board, color
        ) - self._capture_potential_score(board, opponent_color)
        promotion_potential = self._promotion_potential_score(
            board, color
        ) - self._promotion_potential_score(board, opponent_color)

        return (
            material * MATERIAL_WEIGHT
            + mobility * MOBILITY_WEIGHT
            + position * POSITION_WEIGHT
            + king_value * KING_WEIGHT
            + capture_potential * CAPTURE_WEIGHT
            + promotion_potential * PROMOTION_WEIGHT
        )

    def _material_score(self, encoded: list[int], color: str) -> float:
        man_value, king_value = self._piece_values(color)
        score = 0.0
        for value in encoded:
            if value == man_value:
                score += MAN_VALUE
            elif value == king_value:
                score += KING_VALUE
        return score

    def _positional_score(self, encoded: list[int], color: str) -> float:
        man_value, king_value = self._piece_values(color)
        score = 0.0
        for index, value in enumerate(encoded):
            if value == man_value:
                row_idx = index // 8
                col_idx = index % 8
                score += man_positional_value(row_idx, col_idx, color)
            elif value == king_value:
                row_idx = index // 8
                col_idx = index % 8
                score += king_positional_value(row_idx, col_idx)
        return score

    def _king_score(self, encoded: list[int], color: str) -> float:
        _, king_value = self._piece_values(color)
        return float(sum(1 for value in encoded if value == king_value)) * KING_VALUE

    def _mobility_score(self, board, color: str) -> float:
        moves = board.get_valid_moves(color)
        capture_moves = sum(1 for move in moves if move.captured_positions)
        return float(len(moves) + capture_moves * 0.5)

    def _capture_potential_score(self, board, color: str) -> float:
        score = 0.0
        for move in board.get_valid_moves(color):
            captures = len(move.captured_positions) if move.captured_positions else 0
            if captures:
                score += captures * 2.0
                if captures > 1:
                    score += (captures - 1) * 1.25
        return score

    def _promotion_potential_score(self, board, color: str) -> float:
        score = 0.0
        for row_idx, row in enumerate(board.grid):
            for piece in row:
                if piece is None:
                    continue
                if isinstance(piece, int):
                    if self._matches_encoded_color(piece, color) and abs(piece) == 1:
                        score += promotion_potential_value(row_idx, color)
                else:
                    if piece_color(piece) == color and not is_king(piece):
                        score += promotion_potential_value(row_idx, color)
        return score

    def _matches_encoded_color(self, piece: int, color: str) -> bool:
        return (piece > 0 and color == "BLACK") or (piece < 0 and color == "WHITE")

    def _piece_values(self, color: str) -> tuple[int, int]:
        if color == "BLACK":
            return 1, 2
        return -1, -2
