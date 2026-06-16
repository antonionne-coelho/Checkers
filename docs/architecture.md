# Arquitetura

O projeto está organizado em cinco áreas principais:

- **Core**: lógica de jogo e tabuleiro
- **Agents**: implementação de jogadores IA e humano
- **AI**: heurísticas e componentes de busca
- **Metrics**: coleta e relatório de estatísticas
- **UI**: visualização opcional com Pygame

## Core

`core/board.py` oferece a estrutura do tabuleiro, clonagem, aplicação de movimentos e detecção de fim de jogo.

`core/game.py` controla o loop de jogo, alterna jogadores e registra métricas.

## Agents

A interface `agents/base_agent.py` define o método `get_best_move(board)` que todos os agentes implementam.

Agentes disponíveis:

- `MinimaxAgent`: busca por minimax com poda alfa-beta.
- `MinimaxAgentParallel`: avalia movimentos raiz em processos separados.
- `MCTSAgent`: árvore de busca Monte Carlo Tree Search com política de rollout guiada.
- `MCTSAgentParallel`: executa iterações MCTS em paralelo usando threads.
- `MCTSAgentProcess`: executa simulações MCTS independentes usando processos.
- `HybridAgent`: combina MCTS com uma avaliação interna de Minimax para selecionar o melhor movimento.
- `HumanAgent`: permite que um usuário jogue no terminal.

## AI

`ai/evaluator.py` calcula uma pontuação de posição usando:

- material
- mobilidade
- posição no tabuleiro
- valor de reis
- potencial de captura

`ai/mcts_node.py` armazena o estado de cada nó MCTS, incluindo prior, visitas e crianças.

`ai/batch_evaluator.py` define a interface de avaliação em lote usada pelo MCTS. A implementação inicial `PythonBatchEvaluator` mantém a heurística atual em CPU, mas deixa o contrato pronto para backends GPU.

`ai/vector_batch_evaluator.py` implementa o backend `cpu-vector`, que avalia lotes a partir de uma codificação numérica de 64 casas gerada por `ai/board_encoder.py`.

`ai/batch_evaluator_benchmark.py` compara backends de avaliação em lote em posições determinísticas, reportando tempo, checksum e divergência contra o backend `python`.

`ai/batch_evaluator_factory.py` seleciona o avaliador em lote a partir de `--gpu-backend`, com suporte aos nomes `auto`, `python`, `cpu-vector`, `cuda`, `directml`, `onnx-directml`, `opencl` e `metal`. Enquanto não houver implementação GPU real, pedidos GPU falham cedo com uma mensagem explícita.

`ai/directml_batch_evaluator.py` é o primeiro backend GPU opcional. Ele usa `torch-directml` para calcular as features numéricas em tensores DirectML quando o pacote e o dispositivo estão disponíveis. Features que dependem de regras legais, como mobilidade e potencial de captura, continuam em CPU nesta fase.

`ai/gpu/backends/onnx_directml.py` detecta `onnxruntime-directml`, uma alternativa DirectML compatível com Python 3.14 no Windows. O avaliador ONNX ainda deve ser implementado para executar a heurística por esse backend.

`ai/gpu/` centraliza a detecção/validação inicial de solicitações GPU. O detector identifica fabricantes e backends como CUDA, DirectML, OpenCL e Metal/MPS, mas o projeto ainda não executa busca na GPU. Para isso, a próxima evolução deve criar um avaliador em lote compatível com o backend escolhido.

## Métricas e Dataset

`metrics/metrics_collector.py` armazena movimentos, tempos, estados analisados, jogador da vez e snapshots do tabuleiro antes/depois de cada jogada.

`metrics/match_report.py` gera relatório de console, exporta dados de benchmark em CSV/JSON e exporta um dataset CSV de aprendizado por jogada com resultado final por jogador.

## Fluxo de partida

1. `main.py` cria agentes com base em argumentos CLI.
2. `Game.play_match` executa os turnos enquanto o jogo não termina.
3. Cada turno chama o `get_best_move` do agente atual.
4. Movimentos são aplicados no tabuleiro e métricas são registradas.
5. Ao final, `MatchReport` imprime um resumo dos resultados e pode exportar dataset/benchmark.
