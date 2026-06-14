"""Heurísticas de avaliação de tabuleiro para o jogo de damas.

Versão 2.0 — heurística balanceada que elimina viés do primeiro jogador.
Correções aplicadas:
- back_row_bonus corrigido: agora representa a linha de ORIGEM de cada cor
  (a linha traseira que protege contra capturas, não a linha de promoção)
- Adicionado tempo_bonus: pequena vantagem para quem tem o turno
- Adicionado safety_score: penaliza peças expostas nas bordas sem suporte
- Posição pesada de 0.5 → 1.5 para reduzir dominância pura do material
"""

from core.rules import is_king, piece_color, get_valid_moves


MAN_VALUE = 1.0
KING_VALUE = 2.5

MATERIAL_WEIGHT = 10.0
MOBILITY_WEIGHT = 1.5
POSITION_WEIGHT = 1.5
KING_WEIGHT = 5.0
CAPTURE_WEIGHT = 3.5
PROMOTION_WEIGHT = 2.0
SAFETY_WEIGHT = 1.0
TEMPO_BONUS = 0.3


def is_center_column(col_idx: int) -> bool:
    return 2 <= col_idx <= 5


def man_positional_value(row_idx: int, col_idx: int, color: str) -> float:
    """Valor posicional de um peão normal.

    BLACK avança aumentando row (0→7), WHITE avança diminuindo row (7→0).
    back_row_bonus recompensa manter peças na linha de ORIGEM (proteção),
    não confundir com a linha de promoção (que fica no lado oposto).
    """
    if color == "BLACK":
        advance_score = row_idx / 7.0          # 0.0 no topo, 1.0 embaixo
        back_row_bonus = 0.15 if row_idx == 0 else 0.0   # linha de origem de BLACK
    else:
        advance_score = (7 - row_idx) / 7.0   # 0.0 embaixo, 1.0 no topo
        back_row_bonus = 0.15 if row_idx == 7 else 0.0   # linha de origem de WHITE

    center_bonus = 0.25 if is_center_column(col_idx) else 0.0
    edge_penalty = -0.1 if col_idx in {0, 7} else 0.0    # borda é ruim (menos mobilidade)
    return advance_score * 0.4 + center_bonus + edge_penalty + back_row_bonus


def king_positional_value(row_idx: int, col_idx: int) -> float:
    """Valor posicional de um rei — prefere centro e meio do tabuleiro."""
    center_bonus = 0.25 if is_center_column(col_idx) else 0.0
    row_bonus = 0.15 if 2 <= row_idx <= 5 else 0.0
    return center_bonus + row_bonus


def promotion_potential_value(row_idx: int, color: str) -> float:
    """Quão perto o peão está de promover a rei."""
    distance = 7 - row_idx if color == "BLACK" else row_idx
    if distance <= 4:
        return max(0.0, (5 - distance) * 0.3)
    return 0.0


def _count_supporters(grid, row: int, col: int, color: str) -> int:
    """Conta quantas peças aliadas estão atrás/diagonal de (row, col)."""
    if color == "BLACK":
        back_rows = [row - 1]
    else:
        back_rows = [row + 1]

    count = 0
    for br in back_rows:
        for bc in [col - 1, col + 1]:
            if 0 <= br < 8 and 0 <= bc < 8:
                p = grid[br][bc]
                if p is not None and piece_color(p) == color:
                    count += 1
    return count


class BoardEvaluator:
    """Avalia posições do tabuleiro usando heurísticas múltiplas balanceadas."""

    def evaluate(self, board, color: str) -> float:
        winner = board.get_winner()
        if winner == color:
            return 1000.0
        if winner is not None and winner != color:
            return -1000.0

        opp = "WHITE" if color == "BLACK" else "BLACK"

        material      = self.material_score(board, color) - self.material_score(board, opp)
        mobility      = self.mobility_score(board, color) - self.mobility_score(board, opp)
        position      = self.positional_score(board, color) - self.positional_score(board, opp)
        king_val      = self.king_score(board, color) - self.king_score(board, opp)
        capture_pot   = self.capture_potential_score(board, color) - self.capture_potential_score(board, opp)
        promo_pot     = self.promotion_potential_score(board, color) - self.promotion_potential_score(board, opp)
        safety        = self.safety_score(board, color) - self.safety_score(board, opp)

        # Pequeno bônus para quem tem o turno (tempo)
        tempo = TEMPO_BONUS if board.current_player == color else -TEMPO_BONUS

        return (
            material    * MATERIAL_WEIGHT
            + mobility  * MOBILITY_WEIGHT
            + position  * POSITION_WEIGHT
            + king_val  * KING_WEIGHT
            + capture_pot * CAPTURE_WEIGHT
            + promo_pot * PROMOTION_WEIGHT
            + safety    * SAFETY_WEIGHT
            + tempo
        )

    def material_score(self, board, color: str) -> float:
        score = 0.0
        for row in board.grid:
            for piece in row:
                if piece and piece_color(piece) == color:
                    score += KING_VALUE if is_king(piece) else MAN_VALUE
        return score

    def positional_score(self, board, color: str) -> float:
        score = 0.0
        for row_idx, row in enumerate(board.grid):
            for col_idx, piece in enumerate(row):
                if piece and piece_color(piece) == color:
                    if is_king(piece):
                        score += king_positional_value(row_idx, col_idx)
                    else:
                        score += man_positional_value(row_idx, col_idx, color)
        return score

    def mobility_score(self, board, color: str) -> float:
        moves = board.get_valid_moves(color)
        capture_moves = sum(1 for m in moves if m.captured_positions)
        return float(len(moves) + capture_moves * 0.5)

    def king_score(self, board, color: str) -> float:
        return sum(
            KING_VALUE
            for row in board.grid
            for piece in row
            if piece and is_king(piece) and piece_color(piece) == color
        )

    def capture_potential_score(self, board, color: str) -> float:
        moves = board.get_valid_moves(color)
        score = 0.0
        for move in moves:
            captures = len(move.captured_positions) if move.captured_positions else 0
            if captures:
                score += captures * 2.0
                if captures > 1:
                    score += (captures - 1) * 1.0
        return score

    def promotion_potential_score(self, board, color: str) -> float:
        score = 0.0
        for row_idx, row in enumerate(board.grid):
            for col_idx, piece in enumerate(row):
                if piece and piece_color(piece) == color and not is_king(piece):
                    score += promotion_potential_value(row_idx, color)
        return score

    def safety_score(self, board, color: str) -> float:
        """Recompensa peças com suporte aliado atrás delas."""
        score = 0.0
        for row_idx, row in enumerate(board.grid):
            for col_idx, piece in enumerate(row):
                if piece and piece_color(piece) == color:
                    supporters = _count_supporters(board.grid, row_idx, col_idx, color)
                    score += supporters * 0.3
        return score
