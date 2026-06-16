"""Loop do jogo com suporte a empate por inatividade."""

import time
from typing import Optional


MAX_MOVES_WITHOUT_CAPTURE = 120  # Regra de empate: 120 half-moves sem captura


class Game:
    """Controls the game loop, turn order and metrics collection."""

    def __init__(self, board, black_agent, white_agent, metrics_collector=None):
        self.board = board
        self.black_agent = black_agent
        self.white_agent = white_agent
        self.metrics_collector = metrics_collector
        self._moves_without_capture = 0

    def play_match(self, step: bool = False) -> Optional[str]:
        if self.metrics_collector:
            self.metrics_collector.start_match(self.black_agent, self.white_agent)

        if step:
            print("\n=== Modo passo a passo ativado ===")
            print(self.board)

        starting_player = self.board.current_player
        first_move = True
        self._moves_without_capture = 0

        while not self.is_game_over():
            result = self.play_turn(step=step)
            if not result:
                break

            if step and (self.board.current_player == starting_player or self.is_game_over()) and not first_move:
                input("Pressione Enter para prosseguir para o próximo turno...")

            first_move = False

        winner = self.board.get_winner()
        if self.is_draw():
            winner = None  # Empate

        if self.metrics_collector:
            self.metrics_collector.end_match(winner)

        return winner

    def play_turn(self, step=False):
        agent = self.get_current_agent()
        if agent is None:
            return False

        if step:
            print(f"\nTurno de {self.board.current_player} - agente: {agent.name}")

        player = self.board.current_player
        board_before = self.board.clone()
        move_start_time = time.perf_counter()
        move = agent.get_best_move(self.board.clone())
        move_end_time = time.perf_counter()

        if move is None:
            return False

        # Rastreia jogadas sem captura (para detectar empate)
        if move.captured_positions:
            self._moves_without_capture = 0
        else:
            self._moves_without_capture += 1

        self.board.apply_move(move)
        board_after = self.board.clone()
        elapsed_time = move_end_time - move_start_time
        states_analyzed = getattr(agent, "states_analyzed", 0)

        if self.metrics_collector:
            self.metrics_collector.record_move(
                agent.name,
                move,
                elapsed_time,
                states_analyzed,
                player=player,
                board_before=board_before,
                board_after=board_after,
            )

        if step:
            print(f"Jogada escolhida: {move.start_pos} -> {move.end_pos} | Capturas: {len(move.captured_positions) if move.captured_positions else 0} | Promoção: {move.is_promotion}")
            print(self.board)

        return True

    def get_current_agent(self):
        return self.black_agent if self.board.current_player == "BLACK" else self.white_agent

    def is_draw(self) -> bool:
        return self._moves_without_capture >= MAX_MOVES_WITHOUT_CAPTURE

    def is_game_over(self) -> bool:
        return self.board.is_terminal() or self.is_draw()
