"""Utilities for search and move selection used by agents.

This module provides a few focused helpers used throughout search algorithms
and agents for move ranking, filtering and combination. They are intentionally
small and well-documented so they can be reused in tests and tutorials.
"""

from typing import Iterable, List, Tuple, Any, Callable


def flatten_moves(nested: Iterable[Iterable[Any]]) -> List[Any]:
    """Flatten an iterable of iterables into a single list.

    Args:
        nested: Qualquer iterable que produz iteráveis de movimentos.

    Returns:
        Lista simples contendo todos os elementos de forma sequencial.

    Example:
        >>> flatten_moves([[1,2],[3]])
        [1,2,3]
    """
    return [item for sub in nested for item in sub]


def moves_with_captures(moves: Iterable[Any]) -> List[Any]:
    """Retorna apenas os movimentos que realizam captura.

    A função assume que um objeto `move` expõe o atributo
    `captured_positions` (lista) quando uma captura ocorre.

    Args:
        moves: Iterável de objetos `Move`.

    Returns:
        Lista de movimentos que possuem pelo menos uma posição em
        `captured_positions`.
    """
    return [m for m in moves if getattr(m, "captured_positions", None)]


def top_n_moves_by_key(moves: Iterable[Any], key: Callable[[Any], float], n: int = 3) -> List[Any]:
    """Retorna os `n` melhores movimentos ordenados por `key(move)`.

    Args:
        moves: Iterável de movimentos.
        key: Função que recebe um `move` e retorna um valor numérico para ordenar.
        n: Quantidade máxima de movimentos a retornar.

    Returns:
        Lista com no máximo `n` movimentos, ordenada do melhor ao pior.
    """
    return sorted(list(moves), key=key, reverse=True)[:n]
