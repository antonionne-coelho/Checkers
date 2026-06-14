from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.board import Board
    from core.move import Move


class PygameView:
    """Simple Pygame-based view for visualizing the board.

    This class provides lightweight hooks for initializing a Pygame window,
    drawing the board and pieces, handling user input and updating the frame.
    The implementation is intentionally minimal so it can be adapted to the
    user's preferred rendering details.

    Expectations:
    - The provided `board` object must expose `.grid` (8x8) with piece codes.
    - Methods are split to allow unit testing of drawing logic independent of
      the Pygame loop (e.g. call `draw_board()` + `draw_pieces()` then `update()`).
    """

    def __init__(self, board: "Board") -> None:
        """Create a view attached to a `board` object.

        Args:
            board: Instância de `Board` com atributo `.grid`.
        """
        self.board = board

    def initialize_window(self) -> None:
        """Inicializa a janela Pygame e recursos relacionados.

        Comportamento esperado (a implementar):
        - Chamar `pygame.init()`, criar `display.set_mode()` e carregar assets.
        - Criar superfícies para o tabuleiro e ícones de peça.

        Retorna:
            None
        """
        pass

    def draw_board(self) -> None:
        """Desenha o tabuleiro (grade, cores das casas, marcas).

        Deve se limitar a operações de desenho - não deve atualizar a janela.
        Isso facilita testes unitários ao inspecionar chamadas ou superfícies.
        """
        pass

    def draw_pieces(self) -> None:
        """Renderiza peças na superfície do tabuleiro.

        Itera sobre `self.board.grid` e desenha cada peça na célula correspondente.
        Peças como 'BLACK_KING' devem ser representadas de forma distinta.
        """
        pass

    def draw_valid_moves(self, moves: list["Move"]) -> None:
        """Destaca movimentos válidos para a peça selecionada.

        Args:
            moves: Lista de objetos `Move` com `end_pos` indicando destinos.
        """
        pass

    def handle_user_input(self) -> Any:
        """Processa eventos Pygame e traduz em ações de jogo.

        Deve retornar valores no formato esperado pelo `HumanAgent`, por exemplo
        coordenadas selecionadas ou uma ação de 'quit'. A implementação pode
        ser adaptada conforme a interface do agente humano.
        """
        pass

    def update(self) -> None:
        """Atualiza a exibição; deve chamar `pygame.display.flip()` ou similar.

        Separar `draw_*` de `update()` facilita animações e testes.
        """
        pass