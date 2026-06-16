from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class Move:
    """Representa um movimento de damas.

    Exemplos de uso:
        >>> Move((2,1), (3,2), [], [], 'BLACK')
        Move(start_pos=(2, 1), end_pos=(3, 2), ...)

    Attributes:
        start_pos: Tupla `(row, col)` com a posição inicial.
        end_pos: Tupla `(row, col)` com a posição final.
        captured_positions: Lista de posições capturadas (padrão: lista vazia).
        captured_pieces: Lista das peças capturadas correspondentes (padrão: lista vazia).
        piece_moved: Código da peça movida (ex.: 'BLACK', 'WHITE_KING').
        is_promotion: True se o movimento promoveu a peça a rei.
    """
    start_pos: Tuple[int, int]
    end_pos: Tuple[int, int]
    captured_positions: List[Tuple[int, int]] = field(default_factory=list)
    captured_pieces: List[str] = field(default_factory=list)
    piece_moved: Optional[str] = None
    is_promotion: bool = False
