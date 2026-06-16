# API - Core

Documentação dos componentes principais do motor de jogo.

## Board

`Board` mantém o estado do tabuleiro e as regras de movimento.

::: core.board.Board

### Exemplo: Criar e imprimir tabuleiro

```python
from core.board import Board

b = Board()
print(b)
```

## Game

`Game` executa o loop de partida, alterna jogadores e encerra o jogo.

::: core.game.Game

### Exemplo: Jogar uma partida automática

```python
from core.game import Game
from agents.minimax_agent import MinimaxAgent

p1 = MinimaxAgent(depth=3)
p2 = MinimaxAgent(depth=3)

game = Game(p1, p2)
game.play_match()
```

## Move

Representa uma ação de mover uma peça de um ponto a outro.

::: core.move.Move

## Regras

Funções de regras e geração de jogadas válidas.

::: core.rules

### Exemplo: Gerar movimentos válidos

```python
from core.rules import get_valid_moves

moves = get_valid_moves(board, 'BLACK')
for m in moves:
    print(m)
```
