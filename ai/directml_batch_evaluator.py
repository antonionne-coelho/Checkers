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
    man_positional_value,
    promotion_potential_value,
)
from ai.gpu.backends.directml import SUPPORTED_TORCH_DIRECTML_TAGS
from ai.python_compat import current_python_tag, is_current_python_tag_supported


class DirectMLBatchEvaluator(BatchEvaluator):
    """Torch-DirectML evaluator for numeric board batches.

    DirectML acceleration is optional. This class only imports torch packages
    when instantiated, so the rest of the project runs without GPU dependencies.
    """

    name = "directml"

    @staticmethod
    def is_available() -> bool:
        if not is_current_python_tag_supported(SUPPORTED_TORCH_DIRECTML_TAGS):
            return False

        try:
            import torch  # noqa: F401
            import torch_directml

            torch_directml.device()
            return True
        except Exception:
            return False

    def __init__(self):
        if not is_current_python_tag_supported(SUPPORTED_TORCH_DIRECTML_TAGS):
            raise RuntimeError(
                "torch-directml is not available for "
                f"{current_python_tag()}. Use Python 3.12 for DirectML."
            )

        import torch
        import torch_directml

        self.torch = torch
        self.device = torch_directml.device()
        self.black_position_weights = self._build_position_weights("BLACK")
        self.white_position_weights = self._build_position_weights("WHITE")

    def evaluate_batch(self, boards, color: str) -> list[float]:
        board_list = list(boards)
        if not board_list:
            return []

        encoded = encode_boards_flat(board_list)
        tensor = self.torch.tensor(
            encoded,
            dtype=self.torch.float32,
        ).to(self.device)

        scores = self._evaluate_tensor_batch(tensor, board_list, color)
        return scores.cpu().tolist()

    def _evaluate_tensor_batch(self, tensor, boards, color: str):
        opponent_color = "WHITE" if color == "BLACK" else "BLACK"
        material = self._material_score(tensor, color) - self._material_score(
            tensor, opponent_color
        )
        position = self._positional_score(tensor, color) - self._positional_score(
            tensor, opponent_color
        )
        king_value = self._king_score(tensor, color) - self._king_score(
            tensor, opponent_color
        )
        mobility = self._cpu_feature_tensor(
            boards,
            lambda board: (
                self._mobility_score(board, color)
                - self._mobility_score(board, opponent_color)
            ),
        )
        capture_potential = self._cpu_feature_tensor(
            boards,
            lambda board: (
                self._capture_potential_score(board, color)
                - self._capture_potential_score(board, opponent_color)
            ),
        )
        promotion_potential = self._cpu_feature_tensor(
            boards,
            lambda board: (
                self._promotion_potential_score(board, color)
                - self._promotion_potential_score(board, opponent_color)
            ),
        )

        scores = (
            material * MATERIAL_WEIGHT
            + mobility * MOBILITY_WEIGHT
            + position * POSITION_WEIGHT
            + king_value * KING_WEIGHT
            + capture_potential * CAPTURE_WEIGHT
            + promotion_potential * PROMOTION_WEIGHT
        )
        return self._apply_terminal_overrides(scores, boards, color)

    def _material_score(self, tensor, color: str):
        man_value, king_value = self._piece_values(color)
        return (tensor == man_value).sum(dim=1).to(self.torch.float32) * MAN_VALUE + (
            tensor == king_value
        ).sum(dim=1).to(self.torch.float32) * KING_VALUE

    def _positional_score(self, tensor, color: str):
        man_value, king_value = self._piece_values(color)
        weights = (
            self.black_position_weights
            if color == "BLACK"
            else self.white_position_weights
        )
        man_score = ((tensor == man_value).to(self.torch.float32) * weights).sum(dim=1)
        king_score = ((tensor == king_value).to(self.torch.float32) * weights).sum(
            dim=1
        )
        return man_score + king_score

    def _king_score(self, tensor, color: str):
        _, king_value = self._piece_values(color)
        return (tensor == king_value).sum(dim=1).to(self.torch.float32) * KING_VALUE

    def _cpu_feature_tensor(self, boards, feature_fn):
        values = [feature_fn(board) for board in boards]
        return self.torch.tensor(values, dtype=self.torch.float32).to(self.device)

    def _apply_terminal_overrides(self, scores, boards, color: str):
        overrides = []
        for index, board in enumerate(boards):
            winner = board.get_winner()
            if winner == color:
                overrides.append((index, 1000.0))
            elif winner is not None:
                overrides.append((index, -1000.0))

        if not overrides:
            return scores

        adjusted = scores.clone()
        for index, value in overrides:
            adjusted[index] = value
        return adjusted

    def _build_position_weights(self, color: str):
        weights = []
        for row_idx in range(8):
            for col_idx in range(8):
                if color == "BLACK":
                    weights.append(man_positional_value(row_idx, col_idx, color))
                else:
                    weights.append(man_positional_value(row_idx, col_idx, color))
        return self.torch.tensor(weights, dtype=self.torch.float32).to(self.device)

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

    def _piece_values(self, color: str) -> tuple[int, int]:
        if color == "BLACK":
            return 1, 2
        return -1, -2

    def _matches_encoded_color(self, piece, color: str) -> bool:
        return (piece > 0 and color == "BLACK") or (piece < 0 and color == "WHITE")
