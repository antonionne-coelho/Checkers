# API - AI

Documentação dos componentes de avaliação e MCTS.

::: ai.evaluator.BoardEvaluator

## Exemplo: Usar o avaliador

```python
from ai.evaluator import BoardEvaluator

e = BoardEvaluator()
score = e.evaluate(board, 'BLACK')
print(score)
```

::: ai.mcts_node.MCTSNode

## Exemplo: Instanciar nó MCTS

```python
from ai.mcts_node import MCTSNode

root = MCTSNode(state=board, parent=None, prior=0.0)
```
