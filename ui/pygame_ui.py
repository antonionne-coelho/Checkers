"""Interface gráfica Pygame para o jogo de Damas.

Modos disponíveis:
- Human vs AI  : jogador humano (cliques) vs agente IA
- AI vs AI     : uma partida entre dois agentes, visualizada em tempo real
- N partidas   : modo torneio — placar ao vivo, gráficos ao final
"""

from __future__ import annotations

import io
import math
import sys
import threading
import time
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pygame

from agents.minimax_agent import MinimaxAgent
from agents.mcts_agent import MCTSAgent
from agents.hybrid_agent import HybridAgent
from agents.human_agent import HumanAgent
from core.board import Board
from core.game import Game
from metrics.metrics_collector import MetricsCollector
from metrics.match_report import MatchReport

# ──────────────────────────── Paleta de Cores ────────────────────────────────
BG_COLOR        = (18, 18, 30)
PANEL_COLOR     = (28, 28, 46)
BORDER_COLOR    = (55, 55, 85)
ACCENT          = (99, 179, 237)
ACCENT2         = (159, 122, 234)
BTN_NORMAL      = (45, 45, 72)
BTN_HOVER       = (65, 65, 100)
BTN_ACTIVE      = (99, 179, 237)
TEXT_PRIMARY    = (230, 230, 255)
TEXT_SECONDARY  = (140, 140, 180)
TEXT_DARK       = (18, 18, 30)

LIGHT_SQ        = (240, 217, 181)   # bege claro
DARK_SQ         = (101,  67,  33)   # marrom escuro
HIGHLIGHT_SQ    = (124, 252,   0, 130)  # verde semitransparente
LAST_MOVE_SQ    = ( 70, 130, 180,  90)

BLACK_PIECE     = ( 30,  30,  30)
BLACK_PIECE_RIM = ( 80,  80,  80)
WHITE_PIECE     = (245, 245, 245)
WHITE_PIECE_RIM = (180, 180, 180)
KING_GOLD       = (255, 200,  50)

WIN_BLACK       = (100, 100, 255)
WIN_WHITE       = (255, 140,  60)
WIN_DRAW        = (120, 200, 120)

# ──────────────────────────── Layout Constantes ───────────────────────────────
SCREEN_W, SCREEN_H = 1100, 720
BOARD_OFFSET_X      =   60
BOARD_OFFSET_Y      =   60
BOARD_SIZE          =  600
SQUARE_SIZE         = BOARD_SIZE // 8
PANEL_X             = BOARD_OFFSET_X + BOARD_SIZE + 30
PANEL_W             = SCREEN_W - PANEL_X - 20


# ─────────────────────────────── Utilitários ─────────────────────────────────

def draw_rounded_rect(surface, color, rect, radius=10, border=0, border_color=None):
    pygame.draw.rect(surface, color, rect, border_radius=radius)
    if border and border_color:
        pygame.draw.rect(surface, border_color, rect, border, border_radius=radius)


def board_pos_to_screen(row, col):
    x = BOARD_OFFSET_X + col * SQUARE_SIZE
    y = BOARD_OFFSET_Y + row * SQUARE_SIZE
    return x, y


def screen_pos_to_board(px, py):
    col = (px - BOARD_OFFSET_X) // SQUARE_SIZE
    row = (py - BOARD_OFFSET_Y) // SQUARE_SIZE
    if 0 <= row < 8 and 0 <= col < 8:
        return row, col
    return None, None


def draw_piece(surface, row, col, piece_type, highlight=False):
    cx = BOARD_OFFSET_X + col * SQUARE_SIZE + SQUARE_SIZE // 2
    cy = BOARD_OFFSET_Y + row * SQUARE_SIZE + SQUARE_SIZE // 2
    r = SQUARE_SIZE // 2 - 6

    is_black = piece_type.startswith("BLACK")
    is_king  = piece_type.endswith("_KING")

    base_color = BLACK_PIECE if is_black else WHITE_PIECE
    rim_color  = BLACK_PIECE_RIM if is_black else WHITE_PIECE_RIM
    shadow_col = (10, 10, 10, 80)

    # Sombra
    shadow_surf = pygame.Surface((r*2+8, r*2+8), pygame.SRCALPHA)
    pygame.draw.circle(shadow_surf, (0, 0, 0, 60), (r+4, r+6), r)
    surface.blit(shadow_surf, (cx - r - 4, cy - r - 2))

    # Borda / aro
    pygame.draw.circle(surface, rim_color, (cx, cy), r)
    pygame.draw.circle(surface, base_color, (cx, cy), r - 3)

    # Gradiente simulado (círculo claro no topo)
    highlight_col = tuple(min(255, c + 60) for c in base_color)
    pygame.draw.circle(surface, highlight_col, (cx - r//4, cy - r//4), r // 3)

    # Coroa para reis
    if is_king:
        font = pygame.font.SysFont("segoeuisymbol,arial,segoeui", r, bold=True)
        crown = font.render("♛", True, KING_GOLD)
        cr = crown.get_rect(center=(cx, cy))
        surface.blit(crown, cr)

    if highlight:
        hl = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
        pygame.draw.circle(hl, (255, 255, 0, 90), (SQUARE_SIZE//2, SQUARE_SIZE//2), r+4)
        surface.blit(hl, (cx - SQUARE_SIZE//2, cy - SQUARE_SIZE//2))


def matplotlib_to_surface(fig):
    """Converte figura matplotlib para surface Pygame."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=96, bbox_inches="tight")
    buf.seek(0)
    img = pygame.image.load(buf, "png")
    buf.close()
    return img


# ─────────────────────────── Classe Principal ────────────────────────────────

class CheckersGUI:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("♟ Damas — IA vs IA")
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        self.clock  = pygame.time.Clock()

        self.font_title  = pygame.font.SysFont("segoeui,arial", 28, bold=True)
        self.font_large  = pygame.font.SysFont("segoeui,arial", 22, bold=True)
        self.font_med    = pygame.font.SysFont("segoeui,arial", 17)
        self.font_small  = pygame.font.SysFont("segoeui,arial", 14)
        self.font_tiny   = pygame.font.SysFont("segoeui,arial", 12)

        # Estado da UI
        self.state       = "menu"   # menu | playing | results
        self.game_mode   = None     # "human_vs_ai" | "ai_vs_ai" | "tournament"

        # Config de agentes
        self.agent_options = ["Minimax", "MCTS", "Hybrid"]
        self.black_agent_idx = 0
        self.white_agent_idx = 0
        self.minimax_depth   = 3
        self.mcts_iters      = 200
        self.n_matches       = 10

        # Estado do jogo (modo single/human)
        self.board           = None
        self.game            = None
        self.black_agent     = None
        self.white_agent     = None
        self.selected_pos    = None
        self.valid_moves_sel = []
        self.last_move       = None
        self.ai_thinking     = False
        self.ai_thread       = None
        self.ai_next_move    = None

        # Torneio
        self.tournament_results  = []   # lista de winners
        self.tournament_running  = False
        self.tournament_thread   = None
        self.tournament_progress = 0    # partidas completas
        self.current_match_moves = 0
        self.tournament_log      = []   # mensagens de log
        self.chart_surface       = None

        # Botões menu
        self._buttons = {}
        self._sliders = {}

    # ────────────────────────── Loop principal ────────────────────────────────

    def run(self):
        while True:
            dt = self.clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                self._handle_event(event)

            self.screen.fill(BG_COLOR)

            if self.state == "menu":
                self._draw_menu()
            elif self.state == "playing":
                self._draw_game()
                self._update_game()
            elif self.state == "tournament":
                self._draw_tournament()
            elif self.state == "results":
                self._draw_results()

            pygame.display.flip()

    # ─────────────────────────── Eventos ─────────────────────────────────────

    def _handle_event(self, event):
        if self.state == "menu":
            self._menu_event(event)
        elif self.state == "playing":
            self._game_event(event)
        elif self.state == "results":
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.state = "menu"
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self._buttons.get("back_btn") and self._buttons["back_btn"].collidepoint(event.pos):
                    self.state = "menu"

    # ─────────────────────────── Menu ────────────────────────────────────────

    def _menu_event(self, event):
        if event.type != pygame.MOUSEBUTTONDOWN:
            return
        mx, my = event.pos

        for key, rect in self._buttons.items():
            if not rect.collidepoint(mx, my):
                continue
            if key == "black_prev":
                self.black_agent_idx = (self.black_agent_idx - 1) % len(self.agent_options)
            elif key == "black_next":
                self.black_agent_idx = (self.black_agent_idx + 1) % len(self.agent_options)
            elif key == "white_prev":
                self.white_agent_idx = (self.white_agent_idx - 1) % len(self.agent_options)
            elif key == "white_next":
                self.white_agent_idx = (self.white_agent_idx + 1) % len(self.agent_options)
            elif key == "depth_minus":
                self.minimax_depth = max(1, self.minimax_depth - 1)
            elif key == "depth_plus":
                self.minimax_depth = min(6, self.minimax_depth + 1)
            elif key == "iters_minus":
                self.mcts_iters = max(50, self.mcts_iters - 50)
            elif key == "iters_plus":
                self.mcts_iters = min(1000, self.mcts_iters + 50)
            elif key == "n_minus":
                self.n_matches = max(1, self.n_matches - 1)
            elif key == "n_plus":
                self.n_matches = min(100, self.n_matches + 1)
            elif key == "start_human":
                self.game_mode = "human_vs_ai"
                self._start_game()
            elif key == "start_ai":
                self.game_mode = "ai_vs_ai"
                self._start_game()
            elif key == "start_tournament":
                self._start_tournament()

    def _draw_menu(self):
        self._buttons.clear()

        # Título
        self._draw_gradient_title()

        # Painel central
        panel = pygame.Rect(80, 100, SCREEN_W - 160, SCREEN_H - 130)
        draw_rounded_rect(self.screen, PANEL_COLOR, panel, 16, 2, BORDER_COLOR)

        # ── Seção Agentes ──
        col1_x = 120
        col2_x = SCREEN_W // 2 + 20
        row_y   = 140

        self._draw_label("Agente BLACK (pretas)", col1_x, row_y)
        self._draw_label("Agente WHITE (brancas)", col2_x, row_y)

        row_y += 30
        self._buttons["black_prev"] = self._draw_arrow_btn("◄", col1_x, row_y)
        self._draw_value_box(self.agent_options[self.black_agent_idx], col1_x + 40, row_y, 160)
        self._buttons["black_next"] = self._draw_arrow_btn("►", col1_x + 210, row_y)

        self._buttons["white_prev"] = self._draw_arrow_btn("◄", col2_x, row_y)
        self._draw_value_box(self.agent_options[self.white_agent_idx], col2_x + 40, row_y, 160)
        self._buttons["white_next"] = self._draw_arrow_btn("►", col2_x + 210, row_y)

        # ── Parâmetros ──
        row_y += 70
        self._draw_label("Profundidade Minimax", col1_x, row_y)
        self._draw_label("Iterações MCTS", col2_x, row_y)

        row_y += 28
        self._buttons["depth_minus"] = self._draw_small_btn("−", col1_x, row_y)
        self._draw_value_box(str(self.minimax_depth), col1_x + 40, row_y, 80)
        self._buttons["depth_plus"]  = self._draw_small_btn("+", col1_x + 130, row_y)

        self._buttons["iters_minus"] = self._draw_small_btn("−", col2_x, row_y)
        self._draw_value_box(str(self.mcts_iters), col2_x + 40, row_y, 90)
        self._buttons["iters_plus"]  = self._draw_small_btn("+", col2_x + 140, row_y)

        # ── N Partidas ──
        row_y += 80
        nx = SCREEN_W // 2 - 120
        self._draw_label("Número de Partidas (modo torneio)", nx, row_y)
        row_y += 28
        self._buttons["n_minus"] = self._draw_small_btn("−", nx, row_y)
        self._draw_value_box(str(self.n_matches), nx + 40, row_y, 80)
        self._buttons["n_plus"]  = self._draw_small_btn("+", nx + 130, row_y)

        # ── Botões de Início ──
        row_y += 90
        bw, bh = 200, 48
        gap = 30
        total = bw * 3 + gap * 2
        bx = (SCREEN_W - total) // 2

        self._buttons["start_human"]      = self._draw_start_btn("👤  Human vs IA",       bx,          row_y, bw, bh, ACCENT)
        self._buttons["start_ai"]         = self._draw_start_btn("🤖  IA vs IA (1 partida)", bx + bw + gap, row_y, bw, bh, ACCENT2)
        self._buttons["start_tournament"] = self._draw_start_btn(f"🏆  Torneio ({self.n_matches}x)",      bx + (bw+gap)*2, row_y, bw, bh, (80, 200, 120))

        # Rodapé
        hint = self.font_tiny.render("Human vs IA: BLACK é você  |  ESC para voltar ao menu a qualquer momento", True, TEXT_SECONDARY)
        self.screen.blit(hint, (SCREEN_W // 2 - hint.get_width() // 2, SCREEN_H - 35))

    def _draw_gradient_title(self):
        title = "♟  DAMAS COM INTELIGÊNCIA ARTIFICIAL"
        surf = self.font_title.render(title, True, ACCENT)
        self.screen.blit(surf, (SCREEN_W // 2 - surf.get_width() // 2, 52))

    def _draw_label(self, text, x, y):
        s = self.font_small.render(text, True, TEXT_SECONDARY)
        self.screen.blit(s, (x, y))

    def _draw_arrow_btn(self, label, x, y):
        rect = pygame.Rect(x, y, 32, 32)
        mx, my = pygame.mouse.get_pos()
        col = BTN_HOVER if rect.collidepoint(mx, my) else BTN_NORMAL
        draw_rounded_rect(self.screen, col, rect, 6, 1, BORDER_COLOR)
        s = self.font_med.render(label, True, TEXT_PRIMARY)
        self.screen.blit(s, (x + rect.w//2 - s.get_width()//2, y + rect.h//2 - s.get_height()//2))
        return rect

    def _draw_small_btn(self, label, x, y):
        rect = pygame.Rect(x, y, 32, 32)
        mx, my = pygame.mouse.get_pos()
        col = BTN_HOVER if rect.collidepoint(mx, my) else BTN_NORMAL
        draw_rounded_rect(self.screen, col, rect, 6, 1, BORDER_COLOR)
        s = self.font_large.render(label, True, ACCENT)
        self.screen.blit(s, (x + rect.w//2 - s.get_width()//2, y + rect.h//2 - s.get_height()//2))
        return rect

    def _draw_value_box(self, text, x, y, w):
        rect = pygame.Rect(x, y, w, 32)
        draw_rounded_rect(self.screen, (40, 40, 65), rect, 6, 1, BORDER_COLOR)
        s = self.font_med.render(text, True, TEXT_PRIMARY)
        self.screen.blit(s, (x + w//2 - s.get_width()//2, y + 16 - s.get_height()//2))

    def _draw_start_btn(self, label, x, y, w, h, color):
        rect = pygame.Rect(x, y, w, h)
        mx, my = pygame.mouse.get_pos()
        alpha_col = tuple(min(255, c + 30) for c in color) if rect.collidepoint(mx, my) else color
        draw_rounded_rect(self.screen, alpha_col, rect, 12, 2, tuple(min(255, c+60) for c in color))
        s = self.font_med.render(label, True, TEXT_DARK if sum(color) > 400 else TEXT_PRIMARY)
        self.screen.blit(s, (x + w//2 - s.get_width()//2, y + h//2 - s.get_height()//2))
        return rect

    # ────────────────────── Iniciar Jogo ─────────────────────────────────────

    def _build_agent(self, agent_name, color):
        if agent_name == "Minimax":
            return MinimaxAgent(color=color, depth=self.minimax_depth)
        elif agent_name == "MCTS":
            return MCTSAgent(color=color, iterations=self.mcts_iters)
        elif agent_name == "Hybrid":
            return HybridAgent(color=color, iterations=self.mcts_iters,
                               minimax_depth=self.minimax_depth, use_parallel=False)
        return MinimaxAgent(color=color, depth=self.minimax_depth)

    def _start_game(self):
        self.board       = Board()
        self.selected_pos    = None
        self.valid_moves_sel = []
        self.last_move       = None
        self.ai_thinking     = False
        self.ai_next_move    = None

        if self.game_mode == "human_vs_ai":
            self.black_agent = HumanAgent(color="BLACK")
            self.white_agent = self._build_agent(self.agent_options[self.white_agent_idx], "WHITE")
        else:
            self.black_agent = self._build_agent(self.agent_options[self.black_agent_idx], "BLACK")
            self.white_agent = self._build_agent(self.agent_options[self.white_agent_idx], "WHITE")

        metrics = MetricsCollector()
        self.game = Game(self.board, self.black_agent, self.white_agent, metrics)
        self.state = "playing"

    # ───────────────────────── Jogo (playing) ────────────────────────────────

    def _game_event(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.state = "menu"
            return

        if self.game_mode != "human_vs_ai":
            return
        if self.board.current_player != "BLACK":  # humano é BLACK
            return
        if self.ai_thinking:
            return

        if event.type == pygame.MOUSEBUTTONDOWN:
            row, col = screen_pos_to_board(*event.pos)
            if row is None:
                return
            self._handle_human_click(row, col)

    def _handle_human_click(self, row, col):
        piece = self.board.grid[row][col]

        # Se clicou em uma peça própria — selecionar
        if piece and piece.startswith("BLACK"):
            self.selected_pos = (row, col)
            all_moves = self.board.get_valid_moves("BLACK")
            self.valid_moves_sel = [m for m in all_moves if m.start_pos == (row, col)]
            return

        # Se há seleção e clicou em destino válido
        if self.selected_pos:
            target_move = next(
                (m for m in self.valid_moves_sel if m.end_pos == (row, col)), None
            )
            if target_move:
                self.board.apply_move(target_move)
                self.last_move = target_move
                self.selected_pos    = None
                self.valid_moves_sel = []
                return

        self.selected_pos    = None
        self.valid_moves_sel = []

    def _update_game(self):
        if self.board.is_terminal() or self.game.is_game_over():
            return

        current_is_human = (
            self.game_mode == "human_vs_ai"
            and self.board.current_player == "BLACK"
        )
        if current_is_human:
            return

        if self.ai_thinking:
            if self.ai_next_move is not None:
                self.board.apply_move(self.ai_next_move)
                self.last_move   = self.ai_next_move
                self.ai_next_move = None
                self.ai_thinking  = False
            return

        # Lança thread para calcular jogada da IA
        agent = self.game.get_current_agent()
        board_snap = self.board.clone()

        def think():
            self.ai_next_move = agent.get_best_move(board_snap)

        self.ai_thinking  = True
        self.ai_thread    = threading.Thread(target=think, daemon=True)
        self.ai_thread.start()

    def _draw_game(self):
        self._draw_board()
        self._draw_side_panel()

    def _draw_board(self):
        # Fundo do tabuleiro
        board_rect = pygame.Rect(BOARD_OFFSET_X - 8, BOARD_OFFSET_Y - 8, BOARD_SIZE + 16, BOARD_SIZE + 16)
        draw_rounded_rect(self.screen, (50, 35, 20), board_rect, 10)

        valid_ends = {m.end_pos for m in self.valid_moves_sel}

        for row in range(8):
            for col in range(8):
                x, y = board_pos_to_screen(row, col)
                rect = pygame.Rect(x, y, SQUARE_SIZE, SQUARE_SIZE)

                # Cor da casa
                if (row + col) % 2 == 0:
                    sq_col = LIGHT_SQ
                else:
                    sq_col = DARK_SQ
                pygame.draw.rect(self.screen, sq_col, rect)

                # Highlight último movimento
                if self.last_move:
                    if (row, col) in (self.last_move.start_pos, self.last_move.end_pos):
                        hl = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                        hl.fill(LAST_MOVE_SQ)
                        self.screen.blit(hl, (x, y))

                # Highlight seleção e destinos válidos
                if self.selected_pos == (row, col):
                    hl = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                    hl.fill((255, 255, 0, 100))
                    self.screen.blit(hl, (x, y))
                elif (row, col) in valid_ends:
                    hl = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                    hl.fill(HIGHLIGHT_SQ)
                    self.screen.blit(hl, (x, y))
                    # Ponto no centro para indicar destino
                    pygame.draw.circle(self.screen, (100, 220, 80, 200),
                                       (x + SQUARE_SIZE//2, y + SQUARE_SIZE//2), 10)

        # Peças
        for row in range(8):
            for col in range(8):
                piece = self.board.grid[row][col]
                if piece:
                    highlight = self.selected_pos == (row, col)
                    draw_piece(self.screen, row, col, piece, highlight)

        # Coordenadas
        for i in range(8):
            lbl = self.font_tiny.render(str(i), True, TEXT_SECONDARY)
            self.screen.blit(lbl, (BOARD_OFFSET_X - 18, BOARD_OFFSET_Y + i * SQUARE_SIZE + SQUARE_SIZE//2 - 6))
            self.screen.blit(lbl, (BOARD_OFFSET_X + i * SQUARE_SIZE + SQUARE_SIZE//2 - 4, BOARD_OFFSET_Y + BOARD_SIZE + 4))

    def _draw_side_panel(self):
        panel = pygame.Rect(PANEL_X, 40, PANEL_W, SCREEN_H - 80)
        draw_rounded_rect(self.screen, PANEL_COLOR, panel, 14, 1, BORDER_COLOR)

        y = 60

        # Título do modo
        mode_name = "👤 Human vs IA" if self.game_mode == "human_vs_ai" else "🤖 IA vs IA"
        t = self.font_large.render(mode_name, True, ACCENT)
        self.screen.blit(t, (PANEL_X + PANEL_W//2 - t.get_width()//2, y))
        y += 40

        # Turno atual
        player = self.board.current_player
        player_col = (130, 130, 255) if player == "BLACK" else (255, 200, 80)
        lbl = "⬤ BLACK" if player == "BLACK" else "⬤ WHITE"
        if self.ai_thinking:
            lbl += "  (pensando...)"
        t = self.font_med.render(lbl, True, player_col)
        self.screen.blit(t, (PANEL_X + 20, y))
        y += 35

        # Contagem de peças
        black_pieces = sum(1 for r in self.board.grid for p in r if p and p.startswith("BLACK"))
        white_pieces = sum(1 for r in self.board.grid for p in r if p and p.startswith("WHITE"))
        black_kings  = sum(1 for r in self.board.grid for p in r if p == "BLACK_KING")
        white_kings  = sum(1 for r in self.board.grid for p in r if p == "WHITE_KING")

        self._panel_info(f"Peças BLACK: {black_pieces}  (♛ {black_kings})", y, (150, 150, 255))
        y += 25
        self._panel_info(f"Peças WHITE: {white_pieces}  (♛ {white_kings})", y, (255, 210, 120))
        y += 35

        # Agentes
        self._panel_info(f"BLACK: {self.black_agent.name}", y, TEXT_SECONDARY)
        y += 22
        self._panel_info(f"WHITE: {self.white_agent.name}", y, TEXT_SECONDARY)
        y += 30

        # ──── Painel de Interpretabilidade ────
        y = self._draw_decision_panel(y)

        # Fim de jogo
        if self.board.is_terminal() or self.game.is_game_over():
            winner = self.board.get_winner()
            if self.game.is_draw():
                msg, col = "🤝 EMPATE!", (120, 220, 120)
            elif winner == "BLACK":
                msg, col = "🏆 BLACK VENCEU!", (130, 130, 255)
            else:
                msg, col = "🏆 WHITE VENCEU!", (255, 200, 80)
            t = self.font_large.render(msg, True, col)
            self.screen.blit(t, (PANEL_X + PANEL_W//2 - t.get_width()//2, y))
            y += 45
            hint = self.font_small.render("ESC → voltar ao menu", True, TEXT_SECONDARY)
            self.screen.blit(hint, (PANEL_X + PANEL_W//2 - hint.get_width()//2, y))

        # Instruções modo human
        if self.game_mode == "human_vs_ai" and not self.board.is_terminal():
            y = SCREEN_H - 100
            self._panel_info("Clique na sua peça para selecionar", y, TEXT_SECONDARY)
            y += 20
            self._panel_info("Clique no destino destacado para mover", y, TEXT_SECONDARY)

    def _draw_decision_panel(self, y):
        """Desenha o painel de interpretabilidade com informações da última decisão do agente."""
        # Determina qual agente acabou de jogar (o que NÃO é o current_player)
        if self.board.current_player == "BLACK":
            last_agent = self.white_agent
        else:
            last_agent = self.black_agent

        info = getattr(last_agent, "last_decision_info", {})
        if not info:
            return y

        # Separador visual
        sep_rect = pygame.Rect(PANEL_X + 12, y, PANEL_W - 24, 2)
        pygame.draw.rect(self.screen, BORDER_COLOR, sep_rect)
        y += 8

        # Título da seção
        t = self.font_med.render("📊 Decisão do Agente", True, ACCENT2)
        self.screen.blit(t, (PANEL_X + 16, y))
        y += 24

        algorithm = info.get("algorithm", "")

        if algorithm == "Hybrid":
            y = self._draw_hybrid_decision(info, y)
        elif algorithm == "MCTS UCT":
            y = self._draw_mcts_decision(info, y)
        elif algorithm == "Minimax α-β":
            y = self._draw_minimax_decision(info, y)

        return y

    def _draw_minimax_decision(self, info, y):
        """Desenha informações de decisão do Minimax."""
        depth = info.get("depth", "?")
        states = info.get("states_analyzed", 0)
        best_score = info.get("best_score", 0)

        self._panel_info(f"Algoritmo: Minimax α-β  (d={depth})", y, TEXT_PRIMARY)
        y += 20
        self._panel_info(f"Estados analisados: {states:,}", y, TEXT_SECONDARY)
        y += 20
        self._panel_info(f"Melhor score: {best_score}", y, (100, 220, 100))
        y += 24

        # Candidatos
        candidates = info.get("candidates", [])
        if candidates:
            self._panel_info("Top movimentos:", y, TEXT_SECONDARY)
            y += 18
            max_score = max(abs(c["score"]) for c in candidates) if candidates else 1
            if max_score == 0:
                max_score = 1
            for i, c in enumerate(candidates[:4]):
                cap_str = f" ⚔{c['captures']}" if c.get("captures", 0) > 0 else ""
                label = f"{c['start']}→{c['end']}{cap_str}"
                score_str = f"{c['score']:+.1f}"

                # Label do movimento
                col = ACCENT if i == 0 else TEXT_SECONDARY
                s = self.font_tiny.render(f" {label}", True, col)
                self.screen.blit(s, (PANEL_X + 16, y))

                # Score
                sc = self.font_tiny.render(score_str, True, col)
                self.screen.blit(sc, (PANEL_X + PANEL_W - 50, y))

                # Barra visual
                bar_x = PANEL_X + 145
                bar_w = PANEL_W - 210
                bar_h = 10
                bar_y_pos = y + 4
                # Fundo da barra
                pygame.draw.rect(self.screen, (40, 40, 65),
                                 pygame.Rect(bar_x, bar_y_pos, bar_w, bar_h),
                                 border_radius=3)
                # Preenchimento proporcional
                if max_score > 0:
                    fill_ratio = max(0, (c["score"] + max_score) / (2 * max_score))
                    fill_w = max(2, int(bar_w * fill_ratio))
                    bar_color = ACCENT if i == 0 else (80, 80, 130)
                    pygame.draw.rect(self.screen, bar_color,
                                     pygame.Rect(bar_x, bar_y_pos, fill_w, bar_h),
                                     border_radius=3)
                y += 18

        return y

    def _draw_mcts_decision(self, info, y):
        """Desenha informações de decisão do MCTS."""
        iters = info.get("iterations", "?")
        states = info.get("states_analyzed", 0)
        best_visits = info.get("best_visits", 0)
        best_wr = info.get("best_winrate", 0)

        self._panel_info(f"Algoritmo: MCTS UCT  (n={iters})", y, TEXT_PRIMARY)
        y += 20
        self._panel_info(f"Estados analisados: {states:,}", y, TEXT_SECONDARY)
        y += 20
        wr_pct = f"{best_wr * 100:.1f}%"
        self._panel_info(f"Melhor: {best_visits} visitas, {wr_pct} winrate", y, (100, 220, 100))
        y += 24

        # Candidatos
        candidates = info.get("candidates", [])
        if candidates:
            self._panel_info("Top movimentos:", y, TEXT_SECONDARY)
            y += 18
            max_visits = max(c["visits"] for c in candidates) if candidates else 1
            if max_visits == 0:
                max_visits = 1
            for i, c in enumerate(candidates[:4]):
                cap_str = f" ⚔{c['captures']}" if c.get("captures", 0) > 0 else ""
                label = f"{c['start']}→{c['end']}{cap_str}"
                wr = f"{c['winrate']*100:.0f}%"
                visits_str = f"v={c['visits']}"

                col = ACCENT if i == 0 else TEXT_SECONDARY
                s = self.font_tiny.render(f" {label}", True, col)
                self.screen.blit(s, (PANEL_X + 16, y))

                sc = self.font_tiny.render(f"{visits_str} {wr}", True, col)
                self.screen.blit(sc, (PANEL_X + PANEL_W - 75, y))

                # Barra proporcional às visitas
                bar_x = PANEL_X + 145
                bar_w = PANEL_W - 235
                bar_h = 10
                bar_y_pos = y + 4
                pygame.draw.rect(self.screen, (40, 40, 65),
                                 pygame.Rect(bar_x, bar_y_pos, bar_w, bar_h),
                                 border_radius=3)
                fill_w = max(2, int(bar_w * c["visits"] / max_visits))
                bar_color = ACCENT if i == 0 else (80, 80, 130)
                pygame.draw.rect(self.screen, bar_color,
                                 pygame.Rect(bar_x, bar_y_pos, fill_w, bar_h),
                                 border_radius=3)
                y += 18

        return y

    def _draw_hybrid_decision(self, info, y):
        """Desenha informações de decisão do Hybrid."""
        chosen = info.get("chosen_by", "?")
        mm_score = info.get("minimax_score", "—")
        mc_score = info.get("mcts_score", "—")
        mm_time = info.get("minimax_time", 0)
        mc_time = info.get("mcts_time", 0)
        states = info.get("states_analyzed", 0)

        self._panel_info("Algoritmo: Hybrid (MCTS + Minimax)", y, TEXT_PRIMARY)
        y += 20
        self._panel_info(f"Estados analisados: {states:,}", y, TEXT_SECONDARY)
        y += 24

        # Qual venceu
        chosen_label = "Minimax" if chosen == "minimax" else "MCTS"
        chosen_color = (100, 220, 100)
        t = self.font_med.render(f"✓ Escolhido: {chosen_label}", True, chosen_color)
        self.screen.blit(t, (PANEL_X + 16, y))
        y += 24

        # Comparação lado a lado
        mm_color = ACCENT if chosen == "minimax" else TEXT_SECONDARY
        mc_color = ACCENT if chosen == "mcts" else TEXT_SECONDARY

        self._panel_info(f"Minimax: {info.get('minimax_move', '—')}", y, mm_color)
        y += 18
        self._panel_info(f"  Score: {mm_score}  ({mm_time:.3f}s)", y, mm_color)
        y += 22

        self._panel_info(f"MCTS:    {info.get('mcts_move', '—')}", y, mc_color)
        y += 18
        self._panel_info(f"  Score: {mc_score}  ({mc_time:.3f}s)", y, mc_color)
        y += 22

        # Barra comparativa
        if mm_score is not None and mc_score is not None and isinstance(mm_score, (int, float)) and isinstance(mc_score, (int, float)):
            bar_x = PANEL_X + 16
            bar_w = PANEL_W - 32
            bar_h = 14
            pygame.draw.rect(self.screen, (40, 40, 65),
                             pygame.Rect(bar_x, y, bar_w, bar_h),
                             border_radius=4)
            total = abs(mm_score) + abs(mc_score)
            if total > 0:
                mm_ratio = (mm_score + total) / (2 * total)
                mm_w = max(2, int(bar_w * mm_ratio))
                pygame.draw.rect(self.screen, (99, 130, 237),
                                 pygame.Rect(bar_x, y, mm_w, bar_h),
                                 border_radius=4)
                # Label no meio
                mm_lbl = self.font_tiny.render("MM", True, TEXT_PRIMARY)
                mc_lbl = self.font_tiny.render("MC", True, TEXT_PRIMARY)
                self.screen.blit(mm_lbl, (bar_x + 4, y + 1))
                self.screen.blit(mc_lbl, (bar_x + bar_w - 20, y + 1))
            y += 20

        return y

    def _panel_info(self, text, y, color):
        s = self.font_small.render(text, True, color)
        self.screen.blit(s, (PANEL_X + 16, y))

    # ──────────────────────── Torneio ─────────────────────────────────────────

    def _start_tournament(self):
        self.tournament_results  = []
        self.tournament_running  = True
        self.tournament_progress = 0
        self.tournament_log      = []
        self.chart_surface       = None
        self.state               = "tournament"

        black_name = self.agent_options[self.black_agent_idx]
        white_name = self.agent_options[self.white_agent_idx]

        def run_tournament():
            for i in range(self.n_matches):
                if not self.tournament_running:
                    break

                # Alterna quem começa para equilibrar vantagem do primeiro jogador
                if i % 2 == 0:
                    b_agent = self._build_agent(black_name, "BLACK")
                    w_agent = self._build_agent(white_name, "WHITE")
                    roles = ("BLACK", "WHITE")  # (black_side_agent, white_side_agent)
                else:
                    # Inverte: agente black_name joga como WHITE, white_name como BLACK
                    b_agent = self._build_agent(white_name, "BLACK")
                    w_agent = self._build_agent(black_name, "WHITE")
                    roles = ("WHITE", "BLACK")

                board   = Board()
                metrics = MetricsCollector()
                game    = Game(board, b_agent, w_agent, metrics)
                winner  = game.play_match()

                # Normaliza: quem ganhou em termos de agente original (black_name / white_name)?
                if winner is None:
                    agent_winner = None
                elif winner == "BLACK":
                    agent_winner = black_name if i % 2 == 0 else white_name
                else:
                    agent_winner = white_name if i % 2 == 0 else black_name

                self.tournament_results.append(agent_winner)
                self.tournament_progress += 1
                total_moves = sum(len(m["moves"]) for m in metrics.matches)
                self.tournament_log.append(
                    f"Partida {i+1:3d}: {'EMPATE' if agent_winner is None else agent_winner + ' venceu':20s}  ({total_moves} lances)"
                )

            self.tournament_running = False
            self._build_charts()
            self.state = "results"

        self.tournament_thread = threading.Thread(target=run_tournament, daemon=True)
        self.tournament_thread.start()

    def _build_charts(self):
        results   = self.tournament_results
        black_name = self.agent_options[self.black_agent_idx]
        white_name = self.agent_options[self.white_agent_idx]

        n_black = results.count(black_name)
        n_white = results.count(white_name)
        n_draw  = results.count(None)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), facecolor="#12121e")
        fig.suptitle(f"Resultados: {black_name} (⬤) vs {white_name} (○) — {len(results)} partidas",
                     color="white", fontsize=13, fontweight="bold")

        # Pizza
        ax1 = axes[0]
        ax1.set_facecolor("#1c1c2e")
        labels  = []
        sizes   = []
        colors_ = []
        if n_black > 0:
            labels.append(f"{black_name}\n{n_black} vitórias")
            sizes.append(n_black)
            colors_.append("#6366f1")
        if n_white > 0:
            labels.append(f"{white_name}\n{n_white} vitórias")
            sizes.append(n_white)
            colors_.append("#f59e0b")
        if n_draw > 0:
            labels.append(f"Empate\n{n_draw}")
            sizes.append(n_draw)
            colors_.append("#6ee7b7")

        if sizes:
            wedges, texts, autotexts = ax1.pie(
                sizes, labels=labels, colors=colors_,
                autopct="%1.0f%%", startangle=90,
                textprops={"color": "white", "fontsize": 10},
                wedgeprops={"edgecolor": "#12121e", "linewidth": 2}
            )
            for at in autotexts:
                at.set_fontsize(11)
                at.set_fontweight("bold")
        ax1.set_title("Distribuição de Vitórias", color="white", fontsize=11)

        # Linha cumulativa
        ax2 = axes[1]
        ax2.set_facecolor("#1c1c2e")
        ax2.spines[:].set_color("#333355")
        ax2.tick_params(colors="white")
        ax2.xaxis.label.set_color("white")
        ax2.yaxis.label.set_color("white")
        ax2.set_title("Placar Acumulado", color="white", fontsize=11)

        cum_black = []
        cum_white = []
        cum_draw  = []
        cb = cw = cd = 0
        for r in results:
            if r == black_name:
                cb += 1
            elif r == white_name:
                cw += 1
            else:
                cd += 1
            cum_black.append(cb)
            cum_white.append(cw)
            cum_draw.append(cd)

        xs = list(range(1, len(results) + 1))
        ax2.plot(xs, cum_black, color="#6366f1", linewidth=2, label=black_name)
        ax2.plot(xs, cum_white, color="#f59e0b", linewidth=2, label=white_name)
        if any(r is None for r in results):
            ax2.plot(xs, cum_draw, color="#6ee7b7", linewidth=1.5, linestyle="--", label="Empate")
        ax2.legend(facecolor="#1c1c2e", edgecolor="#333355", labelcolor="white", fontsize=9)
        ax2.set_xlabel("Partida #", color="white")
        ax2.set_ylabel("Vitórias acumuladas", color="white")
        ax2.grid(True, color="#222244", linewidth=0.5)

        plt.tight_layout()
        self.chart_surface = matplotlib_to_surface(fig)
        plt.close(fig)

    # ────────────────────── Tela Torneio (progresso) ─────────────────────────

    def _draw_tournament(self):
        total = self.n_matches
        done  = self.tournament_progress
        prog  = done / max(total, 1)

        # Título
        t = self.font_title.render("🏆  Torneio em andamento...", True, ACCENT)
        self.screen.blit(t, (SCREEN_W//2 - t.get_width()//2, 40))

        # Placar
        black_name = self.agent_options[self.black_agent_idx]
        white_name = self.agent_options[self.white_agent_idx]
        n_black = self.tournament_results.count(black_name)
        n_white = self.tournament_results.count(white_name)
        n_draw  = self.tournament_results.count(None)

        score_y = 110
        score_txt = f"⬤ {black_name}   {n_black}  —  {n_draw}  —  {n_white}   {white_name} ○"
        s = self.font_large.render(score_txt, True, TEXT_PRIMARY)
        self.screen.blit(s, (SCREEN_W//2 - s.get_width()//2, score_y))

        # Barra de progresso
        bar_x, bar_y, bar_w, bar_h = 120, 160, SCREEN_W - 240, 28
        draw_rounded_rect(self.screen, BTN_NORMAL, pygame.Rect(bar_x, bar_y, bar_w, bar_h), 8)
        if prog > 0:
            draw_rounded_rect(self.screen, ACCENT, pygame.Rect(bar_x, bar_y, int(bar_w * prog), bar_h), 8)
        pct = self.font_small.render(f"{done}/{total} partidas", True, TEXT_PRIMARY)
        self.screen.blit(pct, (SCREEN_W//2 - pct.get_width()//2, bar_y + 5))

        # Log das últimas partidas
        log_panel = pygame.Rect(120, 210, SCREEN_W - 240, SCREEN_H - 270)
        draw_rounded_rect(self.screen, PANEL_COLOR, log_panel, 10, 1, BORDER_COLOR)
        log_lines = self.tournament_log[-20:]
        ly = 220
        for line in log_lines:
            col = ACCENT if "venceu" in line else (TEXT_SECONDARY if "EMPATE" in line else TEXT_PRIMARY)
            sl = self.font_small.render(line, True, col)
            self.screen.blit(sl, (140, ly))
            ly += 22

    # ──────────────────────── Tela de Resultados ─────────────────────────────

    def _draw_results(self):
        self._buttons.clear()

        black_name = self.agent_options[self.black_agent_idx]
        white_name = self.agent_options[self.white_agent_idx]
        results    = self.tournament_results
        n_black    = results.count(black_name)
        n_white    = results.count(white_name)
        n_draw     = results.count(None)
        total      = len(results)

        # Título
        t = self.font_title.render("🏆  Resultados do Torneio", True, ACCENT)
        self.screen.blit(t, (SCREEN_W//2 - t.get_width()//2, 18))

        # Placar destacado
        panel = pygame.Rect(60, 56, SCREEN_W - 120, 70)
        draw_rounded_rect(self.screen, PANEL_COLOR, panel, 12, 1, BORDER_COLOR)

        def pct(n): return f"{n/total*100:.0f}%" if total else "0%"
        sc = self.font_large.render(
            f"⬤ {black_name}  {n_black} ({pct(n_black)})   |   🤝 {n_draw} empates   |   {n_white} ({pct(n_white)})  {white_name} ○",
            True, TEXT_PRIMARY)
        self.screen.blit(sc, (SCREEN_W//2 - sc.get_width()//2, 76))

        # Gráficos
        if self.chart_surface:
            cw, ch = self.chart_surface.get_size()
            scale  = min((SCREEN_W - 120) / cw, (SCREEN_H - 200) / ch)
            scaled = pygame.transform.smoothscale(self.chart_surface,
                                                  (int(cw*scale), int(ch*scale)))
            self.screen.blit(scaled, (SCREEN_W//2 - scaled.get_width()//2, 140))

        # Botão voltar
        self._buttons["back_btn"] = self._draw_start_btn(
            "← Voltar ao Menu", SCREEN_W//2 - 110, SCREEN_H - 56, 220, 42, ACCENT)


def main():
    gui = CheckersGUI()
    gui.run()


if __name__ == "__main__":
    main()
