# Estrutura do Projeto

Este projeto está organizado em módulos que separam regras, agentes, lógica de jogo e métricas.

- `main.py`: ponto de entrada da aplicação e interface de linha de comando
- `help.py`: ajuda em texto para `--help-br`
- `agents/`: implementações de agentes para o jogo
  - `base_agent.py`: classe base para agentes
  - `minimax_agent.py`: agente Minimax com poda alfa-beta
  - `minimax_agent_parallel.py`: agente Minimax que paraleliza movimentos raiz em processos
  - `mcts_agent.py`: agente Monte Carlo Tree Search (MCTS)
  - `mcts_agent_parallel.py`: agente MCTS com iterações paralelas usando threads
  - `mcts_agent_process.py`: agente MCTS com simulações paralelas usando processos
  - `hybrid_agent.py`: agente híbrido que combina MCTS e Minimax
  - `human_agent.py`: agente que lê jogadas do usuário
- `core/`: lógica central do jogo
  - `board.py`: representa o tabuleiro e aplica movimentos
  - `game.py`: executa a ordem de turnos e integra métricas
  - `move.py`: representa movimentos de damas
  - `rules.py`: regras do jogo e geração de jogadas válidas
- `ai/`: componentes de IA adicionais
  - `batch_evaluator.py`: interface de avaliação em lote para CPU/GPU
  - `batch_evaluator_benchmark.py`: benchmark comparativo de backends de avaliação
  - `batch_evaluator_factory.py`: seleciona avaliadores em lote por backend
  - `board_encoder.py`: codifica tabuleiros em vetores numéricos
  - `directml_batch_evaluator.py`: avaliador opcional baseado em torch-directml
  - `evaluator.py`: heurísticas para avaliar posições de tabuleiro
  - `gpu/`: detecta fabricantes e backends GPU antes de existir um avaliador acelerado
    - `backends/onnx_directml.py`: detecta `onnxruntime-directml` como alternativa DirectML para Python 3.14
  - `minimax_worker.py`: avalia movimentos Minimax em processos independentes
  - `mcts_node.py`: nó de árvore MCTS com seleção e expansão
  - `mcts_worker.py`: executa simulações MCTS em processos independentes
  - `search_utils.py`: utilitários de busca auxiliares
  - `vector_batch_evaluator.py`: avaliador em lote CPU baseado em vetores
- `metrics/`: coleta e exportação de métricas de partidas
  - `metrics_collector.py`: registra movimentos, estatísticas e snapshots de tabuleiro
  - `match_report.py`: gera relatórios de partidas, CSV/JSON e dataset de aprendizado
- `tests/`: testes unitários do projeto
- `ui/`: visualização via Pygame (não documentada aqui em detalhe)

## Observações

- O projeto foi projetado para ser modular e extensível.
- Cada agente compartilha a mesma interface básica definida em `agents/base_agent.py`.
