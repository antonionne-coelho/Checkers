"""Regras do jogo de damas e utilitários para geração de movimentos.

Este módulo fornece funções para determinar movimentos válidos no tabuleiro de
damas, incluindo geração de movimentos simples, sequências de captura (multi-
captures), promoção a rei e utilitários de inspeção de peças.

As funções públicas mais importantes:
- `get_valid_moves(board, color)` — retorna lista de `Move` válidos para a cor.
- `opponent(color)` — retorna a cor adversária.

Observações de implementação:
- A geração de capturas percorre recursivamente sequências possíveis para
  suportar múltiplas capturas em um único turno.
- A função aplica a regra de captura obrigatória: se qualquer captura existir,
  apenas sequências de captura são retornadas.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from core.move import Move

if TYPE_CHECKING:
    from core.board import Board


def opponent(color: str) -> str:
    """Retorna a cor oposta a `color`.

    Args:
        color: Valor esperado 'BLACK' ou 'WHITE'.

    Returns:
        A cor adversária como string.

    Exemplo:
        >>> opponent('BLACK')
        'WHITE'
    """
    return "WHITE" if color == "BLACK" else "BLACK"


def is_king(piece: str | None) -> bool:
    """Verifica se uma `piece` é um rei.

    Retorna ``True`` se `piece` não for ``None`` e estiver codificada como
    'COLOR_KING'. Valores esperados para peças são strings como 'BLACK',
    'WHITE', 'BLACK_KING'.
    """
    return piece is not None and piece.endswith("_KING")


def piece_color(piece: str | None) -> str | None:
    """Retorna a cor base de uma peça codificada.

    Converte 'WHITE_KING' -> 'WHITE' e 'BLACK' -> 'BLACK'. Se `piece` for
    ``None``, retorna ``None``.

    Args:
        piece: Código da peça ou ``None``.

    Returns:
        String com a cor ou ``None``.
    """
    if piece is None:
        return None
    return piece.split("_")[0]


def get_piece_directions(piece: str) -> list[tuple[int, int]]:
    """Retorna as direções de movimento válidas para a `piece`.

    Para um `piece` normal (não rei), as direções são direcionais dependendo da
    cor: peças 'BLACK' movem-se para baixo (aumentando o índice de linha),
    'WHITE' movem-se para cima. Para reis, todas as diagonais são permitidas.

    Args:
        piece: Código da peça (ex.: 'BLACK', 'WHITE_KING').

    Returns:
        Lista de tuplas (dr, dc) indicando deslocamentos de linha/coluna.
    """
    if piece is None:
        return []

    if is_king(piece):
        return [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    color = piece_color(piece)
    if color == "BLACK":
        return [(1, -1), (1, 1)]
    return [(-1, -1), (-1, 1)]


def position_in_bounds(position: tuple[int, int]) -> bool:
    """Verifica se uma posição está dentro do tabuleiro 8x8.

    Args:
        position: Tupla `(row, col)`.

    Returns:
        ``True`` quando 0 <= row,col < 8.
    """
    row, col = position
    return 0 <= row < 8 and 0 <= col < 8


def _clone_grid(grid: list[list[str | None]]) -> list[list[str | None]]:
    """Cria uma cópia rasa do grid do tabuleiro.

    Esta função preserva a forma do grid e copia referências de linha para
    evitar mutações no grid original durante simulações de movimentos.
    """
    return [row.copy() for row in grid]


def _can_capture(piece: str, target: str) -> bool:
    """Determina se `piece` pode capturar `target`.

    A captura é possível quando `target` existe e pertence à cor adversária.

    Args:
        piece: Código da peça atacante.
        target: Código da peça alvo.

    Returns:
        ``True`` se `target` for de cor oposta, ``False`` caso contrário.
    """
    return target is not None and piece_color(target) == opponent(piece_color(piece))


def _generate_simple_moves(grid: list[list[str | None]], start_pos: tuple[int, int], piece: str) -> list[Move]:
    """Gera movimentos não-captura para a peça na posição `start_pos`.

    Para cada direção válida, verifica se a casa adjacente está vazia e cria
    um objeto `Move`. Marca `is_promotion` quando o destino alcança a última
    linha (0 para `WHITE`, 7 para `BLACK`).

    Args:
        grid: Matrizes 8x8 com códigos de peça ou ``None``.
        start_pos: Posição inicial da peça.
        piece: Código da peça movida.

    Returns:
        Lista de objetos `Move` representando movimentos simples.
    """
    moves: list[Move] = []
    directions = get_piece_directions(piece)
    row, col = start_pos

    for dr, dc in directions:
        new_pos = (row + dr, col + dc)
        if position_in_bounds(new_pos) and grid[new_pos[0]][new_pos[1]] is None:
            is_promotion = (
                (piece == "BLACK" and new_pos[0] == 7)
                or (piece == "WHITE" and new_pos[0] == 0)
            )
            moves.append(
                Move(
                    start_pos=start_pos,
                    end_pos=new_pos,
                    captured_positions=[],
                    captured_pieces=[],
                    piece_moved=piece,
                    is_promotion=is_promotion,
                )
            )
    return moves


def _generate_capture_sequences(
    grid: list[list[str | None]],
    start_pos: tuple[int, int],
    piece: str,
    current_pos: tuple[int, int],
    captured_positions: list[tuple[int, int]],
    captured_pieces: list[str],
) -> list[Move]:
    """Explora recursivamente todas as sequências de captura possíveis para uma peça.

    A função realiza os passos abaixo:
    1. Para cada direção válida, verifica se existe uma peça inimiga na posição
       'meio' e uma casa vazia imediatamente após (posição de pouso).
    2. Simula a captura atual removendo a peça capturada e movendo a peça ao
       local de pouso (considera promoção quando atinge a última fileira).
    3. Chama-se recursivamente para detectar captura adicional a partir da nova
       posição, acumulando as posições e peças capturadas.
    4. Se não houver capturas adicionais, finaliza a sequência atual como um
       movimento completo.

    Args:
        grid: Estado atual do tabuleiro (8x8).
        start_pos: Posição inicial da peça antes da primeira captura.
        piece: Código da peça no início da sequência (pode se tornar _KING).
        current_pos: Posição a partir da qual procuramos capturas nesta chamada.
        captured_positions: Lista acumulada de posições capturadas até agora.
        captured_pieces: Lista acumulada de códigos de peças capturadas.

    Returns:
        Lista de `Move` representando todas as sequências de captura encontradas.
    """
    row, col = current_pos
    directions = get_piece_directions(piece)
    sequences: list[Move] = []

    for dr, dc in directions:
        mid_pos = (row + dr, col + dc)
        land_pos = (row + 2 * dr, col + 2 * dc)
        if not position_in_bounds(mid_pos) or not position_in_bounds(land_pos):
            continue

        mid_piece = grid[mid_pos[0]][mid_pos[1]]
        landing_piece = grid[land_pos[0]][land_pos[1]]
        if mid_piece is None or landing_piece is not None:
            continue

        if not _can_capture(piece, mid_piece):
            continue

        new_grid = _clone_grid(grid)
        new_grid[row][col] = None
        new_grid[mid_pos[0]][mid_pos[1]] = None

        promoted = False
        next_piece = piece
        if not is_king(piece) and ((piece == "BLACK" and land_pos[0] == 7) or (piece == "WHITE" and land_pos[0] == 0)):
            next_piece = f"{piece_color(piece)}_KING"
            promoted = True

        new_grid[land_pos[0]][land_pos[1]] = next_piece
        next_captured_positions = captured_positions + [mid_pos]
        next_captured_pieces = captured_pieces + [mid_piece]

        if promoted and not is_king(piece):
            sequences.append(
                Move(
                    start_pos=start_pos,
                    end_pos=land_pos,
                    captured_positions=next_captured_positions,
                    captured_pieces=next_captured_pieces,
                    piece_moved=piece,
                    is_promotion=True,
                )
            )
            continue

        next_sequences = _generate_capture_sequences(
            new_grid,
            start_pos,
            next_piece,
            land_pos,
            next_captured_positions,
            next_captured_pieces,
        )
        if next_sequences:
            sequences.extend(next_sequences)
        else:
            sequences.append(
                Move(
                    start_pos=start_pos,
                    end_pos=land_pos,
                    captured_positions=next_captured_positions,
                    captured_pieces=next_captured_pieces,
                    piece_moved=piece,
                    is_promotion=promoted,
                )
            )

    return sequences


def get_valid_moves(board: "Board", color: str) -> list[Move]:
    """Gera a lista de movimentos válidos para `color` no `board`.

    Implementa a regra de captura obrigatória: se qualquer captura estiver
    disponível, somente sequências de captura serão retornadas.

    Args:
        board: Instância de `Board` com atributo `.grid` (8x8).
        color: 'WHITE' ou 'BLACK' indicando o jogador para o qual gerar
               movimentos.

    Returns:
        Lista de `Move` válidos para o jogador.
    """
    moves: list[Move] = []
    capture_moves: list[Move] = []

    for row in range(8):
        for col in range(8):
            piece = board.grid[row][col]
            if piece is None or piece_color(piece) != color:
                continue

            captures = _generate_capture_sequences(board.grid, (row, col), piece, (row, col), [], [])
            if captures:
                capture_moves.extend(captures)
                continue

            simple_moves = _generate_simple_moves(board.grid, (row, col), piece)
            moves.extend(simple_moves)

    return capture_moves if capture_moves else moves


def count_pieces(grid: list[list[str | None]], color: str) -> int:
    """Conta quantas peças de `color` existem no `grid`.

    Args:
        grid: Matriz 8x8 com códigos de peças ou ``None``.
        color: 'WHITE' ou 'BLACK'.

    Returns:
        Inteiro com a contagem de peças da cor.
    """
    return sum(1 for row in grid for piece in row if piece is not None and piece_color(piece) == color)


def has_pieces(grid: list[list[str | None]], color: str) -> bool:
    """Retorna True se `color` possuir ao menos uma peça no `grid`.

    Args:
        grid: Matriz do tabuleiro.
        color: 'WHITE' ou 'BLACK'.

    Returns:
        True quando houver pelo menos uma peça da cor.
    """
    return count_pieces(grid, color) > 0
