PIECE_TO_VALUE = {
    None: 0,
    "BLACK": 1,
    "BLACK_KING": 2,
    "WHITE": -1,
    "WHITE_KING": -2,
}


def encode_board_flat(board) -> list[int]:
    """Encode an 8x8 board as 64 numeric values."""
    return [PIECE_TO_VALUE[piece] for row in board.grid for piece in row]


def encode_boards_flat(boards) -> list[list[int]]:
    """Encode multiple boards using the same 64-value representation."""
    return [encode_board_flat(board) for board in boards]
