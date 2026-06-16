# API - Agentes

Esta página documenta os principais agentes de jogo.

::: agents.base_agent.Agent

## Exemplo de uso: Agent base

```python
from agents.base_agent import Agent

# Agent é uma classe abstrata; use-a como referência de tipo.

def play_with_agent(agent: Agent):
    move = agent.select_move(board)
```

::: agents.minimax_agent.MinimaxAgent

::: agents.minimax_agent_parallel.MinimaxAgentParallel

## Exemplo: MinimaxAgent

```python
from agents.minimax_agent import MinimaxAgent

agent = MinimaxAgent(depth=4)
move = agent.select_move(board)
```

::: agents.mcts_agent.MCTSAgent

::: agents.mcts_agent_parallel.MCTSAgentParallel

::: agents.mcts_agent_process.MCTSAgentProcess

## Exemplo: MCTSAgent

```python
from agents.mcts_agent import MCTSAgent

agent = MCTSAgent(iterations=1000)
move = agent.select_move(board)
```

::: agents.hybrid_agent.HybridAgent

## Exemplo: HybridAgent

```python
from agents.hybrid_agent import HybridAgent

agent = HybridAgent(mcts_iterations=500, minimax_depth=2)
move = agent.select_move(board)
```

::: agents.human_agent.HumanAgent

## Exemplo: HumanAgent

```python
from agents.human_agent import HumanAgent

agent = HumanAgent()
# HumanAgent normalmente interage via terminal ou PygameView.
```
