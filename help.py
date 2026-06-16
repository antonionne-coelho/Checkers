"""Módulo de ajuda em português brasileiro para o projeto Checkers."""


def get_help_text_br() -> str:
    """Retorna o texto de ajuda em português brasileiro."""
    return """
╔══════════════════════════════════════════════════════════════════╗
║              JOGO DE DAMAS — AJUDA (Português BR)               ║
╚══════════════════════════════════════════════════════════════════╝

USO:
  python main.py [opções]

OPÇÕES PRINCIPAIS:
  --gui                        Inicia a interface gráfica Pygame (recomendado)
  --mode MODO                  Define o modo de partida (padrão: "Minimax vs Minimax")
  --depth N                    Profundidade de busca do Minimax (padrão: 3)
  --iterations N               Número de iterações do MCTS (padrão: 100)
  --workers N                  Número de processos paralelos (padrão: 4)
  --step                       Exibe cada turno e aguarda Enter para continuar
  --benchmark                  Executa uma sequência de partidas automáticas de benchmark
  --benchmark-output DIR       Diretório de saída para relatórios CSV/JSON do benchmark
  --dataset-output ARQUIVO     Caminho CSV para exportar dataset de aprendizado
  --help-br                    Exibe esta mensagem de ajuda em português

AGENTES DISPONÍVEIS:
  minimax              Minimax com poda alfa-beta (profundidade fixa)
  minimax-parallel     Minimax com poda alfa-beta paralelo (múltiplos workers)
  mcts                 Monte Carlo Tree Search com UCT
  mcts-process         MCTS usando múltiplos processos do sistema
  hybrid               Agente híbrido: Minimax + MCTS em paralelo
  human                Jogador humano (entrada pelo teclado/mouse)

FORMATO DO MODO (--mode):
  Use "AGENTE1 vs AGENTE2" ou "AGENTE1;AGENTE2"
  Exemplos:
    "MCTS vs Minimax"
    "human vs minimax"
    "hybrid vs mcts"
    "minimax-parallel vs minimax"

EXEMPLOS DE USO:
  python main.py --gui
      → Abre a interface gráfica Pygame

  python main.py --mode "MCTS vs Minimax" --iterations 200 --depth 4
      → Partida MCTS (200 iterações) vs Minimax (profundidade 4)

  python main.py --mode "human vs minimax" --depth 5
      → Você (humano) joga contra o Minimax com profundidade 5

  python main.py --mode "hybrid vs mcts" --iterations 150
      → Partida Hybrid vs MCTS com 150 iterações

  python main.py --benchmark --depth 3 --iterations 100 --benchmark-output relatorios/
      → Executa benchmark completo e salva relatórios em relatorios/

  python main.py --mode "minimax-parallel vs minimax" --depth 4 --workers 4
      → Minimax paralelo (4 workers) vs Minimax clássico

HEURÍSTICA DE AVALIAÇÃO (7 componentes):
  Material            Diferença de peças (peão=1.0, rei=2.5)        peso: 10.0
  Mobilidade          Diferença de movimentos válidos disponíveis    peso:  1.5
  Posição             Valor posicional (centro, avanço, back row)    peso:  1.5
  Reis                Valor adicional por reis no tabuleiro          peso:  5.0
  Captura             Capturas disponíveis (bônus multi-captura)     peso:  3.5
  Promoção            Proximidade de peões à linha de promoção       peso:  2.0
  Segurança           Peças com suporte aliado na diagonal traseira  peso:  1.0

REGRAS DO JOGO:
  - Tabuleiro 8×8, peças pretas iniciam
  - Captura obrigatória: quando há capturas disponíveis, deve capturar
  - Captura múltipla: é possível encadear várias capturas em um turno
  - Promoção: ao atingir a última fileira, a peça vira rei (dama)
  - Empate: detectado por repetição de posições ou excesso de turnos sem captura
"""
