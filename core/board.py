from copy import deepcopy

from core.rules import get_valid_moves, opponent
from core.move import Move


class Board:
    """Represents the Draughts board, pieces and applied moves."""

    def __init__(self):
        self.grid = self.create_initial_board()
        self.current_player = "BLACK"

    def create_initial_board(self):
        board = [[None for _ in range(8)] for _ in range(8)]
        for row in range(8):
            for col in range(8):
                if (row + col) % 2 == 1:
                    if row < 3:
                        board[row][col] = "BLACK"
                    elif row > 4:
                        board[row][col] = "WHITE"
        return board

    def clone(self):
        copy_board = Board()
        copy_board.grid = deepcopy(self.grid)
        copy_board.current_player = self.current_player
        return copy_board

    def __str__(self):
        header = "   " + " ".join(str(i) for i in range(8))
        rows = [header]
        for row_idx, row in enumerate(self.grid):
            row_cells = []
            for piece in row:
                if piece is None:
                    row_cells.append(".")
                elif piece == "BLACK":
                    row_cells.append("b")
                elif piece == "WHITE":
                    row_cells.append("w")
                elif piece == "BLACK_KING":
                    row_cells.append("B")
                elif piece == "WHITE_KING":
                    row_cells.append("W")
                else:
                    row_cells.append("?")
            rows.append(f"{row_idx}  " + " ".join(row_cells))
        rows.append(f"Jogador atual: {self.current_player}")
        return "\n".join(rows)

    def get_valid_moves(self, color):
        return get_valid_moves(self, color)

    def apply_move(self, move: Move):
        if move is None:
            return

        start_row, start_col = move.start_pos
        end_row, end_col = move.end_pos
        piece = self.grid[start_row][start_col]
        if piece is None:
            raise ValueError("No piece at the starting position")

        move.piece_moved = piece
        self.grid[start_row][start_col] = None
        self.grid[end_row][end_col] = piece

        move.captured_positions = move.captured_positions or []
        move.captured_pieces = move.captured_pieces or []
        for idx, position in enumerate(move.captured_positions):
            r, c = position
            captured_piece = self.grid[r][c]
            move.captured_pieces.append(captured_piece) if captured_piece is not None else None
            self.grid[r][c] = None

        if move.is_promotion:
            self.grid[end_row][end_col] = f"{self.current_player}_KING"

        self.switch_player()

    def undo_move(self, move: Move):
        if move is None or move.piece_moved is None:
            return

        start_row, start_col = move.start_pos
        end_row, end_col = move.end_pos
        self.grid[end_row][end_col] = None

        restored_piece = move.piece_moved
        if move.is_promotion and restored_piece is not None and not restored_piece.endswith("_KING"):
            self.grid[start_row][start_col] = restored_piece
        else:
            self.grid[start_row][start_col] = restored_piece

        if move.captured_positions and move.captured_pieces:
            for position, captured_piece in zip(move.captured_positions, move.captured_pieces):
                self.grid[position[0]][position[1]] = captured_piece

        self.switch_player()

    def is_terminal(self):
        black_pieces = any(piece is not None and piece.startswith("BLACK") for row in self.grid for piece in row)
        white_pieces = any(piece is not None and piece.startswith("WHITE") for row in self.grid for piece in row)
        if not black_pieces or not white_pieces:
            return True

        if not self.get_valid_moves(self.current_player):
            return True

        return False

    def get_winner(self):
        black_moves = self.get_valid_moves("BLACK")
        white_moves = self.get_valid_moves("WHITE")
        black_pieces = any(piece is not None and piece.startswith("BLACK") for row in self.grid for piece in row)
        white_pieces = any(piece is not None and piece.startswith("WHITE") for row in self.grid for piece in row)

        if not black_pieces:
            return "WHITE"
        if not white_pieces:
            return "BLACK"
        if self.current_player == "BLACK" and not black_moves:
            return "WHITE"
        if self.current_player == "WHITE" and not white_moves:
            return "BLACK"
        return None

    def switch_player(self):
        self.current_player = opponent(self.current_player)
