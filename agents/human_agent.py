from agents.base_agent import Agent


class HumanAgent(Agent):
    """Agent that prompts a human player for moves in the terminal."""

    def __init__(self, color: str):
        """Initialize the human agent."""
        super().__init__("Human", color)

    def get_best_move(self, board):
        valid_moves = board.get_valid_moves(board.current_player)
        if not valid_moves:
            print("Nenhuma jogada válida disponível.")
            return None

        print(f"Jogador humano ({self.color}), é sua vez.")
        for index, move in enumerate(valid_moves, start=1):
            captures = len(move.captured_positions) if move.captured_positions else 0
            print(f"{index}: {move.start_pos} -> {move.end_pos} | Capturas: {captures} | Promoção: {move.is_promotion}")

        while True:
            choice = input("Escolha o número da jogada: ")
            if not choice.isdigit():
                print("Entrada inválida, digite um número.")
                continue
            choice_index = int(choice) - 1
            if 0 <= choice_index < len(valid_moves):
                return valid_moves[choice_index]
            print("Escolha fora do intervalo, tente novamente.")
