# Uso

Esta seção explica como executar o projeto e quais opções de linha de comando estão disponíveis.

## Executando uma partida

Use o arquivo `main.py` para iniciar um jogo:

```bash
python main.py --mode "MCTS vs Minimax" --iterations 100 --depth 3
```

## Opções de linha de comando

- `--mode`: define os agentes que jogam. Valores válidos:
  - `minimax`
  - `minimax-parallel`
  - `mcts`
  - `mcts-process`
  - `hybrid`
  - `human`

  Exemplos:

  - `Minimax vs Minimax`
  - `minimax-parallel vs minimax`
  - `MCTS vs Minimax`
  - `mcts-process vs minimax`
  - `Hybrid vs Human`

- `--benchmark`: executa uma sequência de partidas automáticas para comparar agentes.
- `--benchmark-evaluators`: compara backends de avaliação em lote em posições determinísticas.
- `--benchmark-evaluator-backends`: backends separados por vírgula para comparar. Padrão: `python,cpu-vector`.
- `--benchmark-positions`: número de posições determinísticas no benchmark de avaliadores.
- `--benchmark-seed`: semente usada para gerar posições do benchmark de avaliadores.
- `--iterations`: número de iterações para agentes `MCTS` e `Hybrid`.
- `--depth`: profundidade de busca para `Minimax` e `Hybrid`.
- `--workers`: número de processos de trabalho para agentes baseados em processos.
- `--workers-gpu`: reserva uma solicitação de workers GPU. Atualmente falha de forma explícita até existir um avaliador GPU em lote.
- `--gpu-backend`: escolhe o backend de avaliação em lote. Valores: `auto`, `python`, `cpu-vector`, `cuda`, `directml`, `onnx-directml`, `opencl`, `metal`.
- `--dataset-output`: caminho CSV para exportar um dataset de aprendizado por jogada.
- `--step`: mostra cada turno e aguarda `Enter` para prosseguir quando não houver jogador humano.
- `--benchmark-output`: diretório de saída para relatórios CSV e JSON de benchmark.
- `-h` ou `--help`: exibe a ajuda em inglês.
- `--help-br`: exibe a ajuda em Português do Brasil.

## Exemplos

Executar uma partida rápida entre MCTS e Minimax:

```bash
python main.py --mode "MCTS vs Minimax" --iterations 50 --depth 2
```

Executar uma partida usando MCTS com multiprocessing:

```bash
python main.py --mode "mcts-process vs minimax" --iterations 100 --workers 4
```

Executar uma partida usando Minimax paralelo:

```bash
python main.py --mode "minimax-parallel vs minimax" --depth 4 --workers 4
```

Exportar um dataset CSV para refinar decisões futuras do MCTS:

```bash
python main.py --mode "mcts-process vs minimax" --iterations 200 --depth 3 --workers 4 --dataset-output data/mcts_dataset.csv
```

Executar uma partida com passo a passo interativo:

```bash
python main.py --mode "MCTS vs Minimax" --step
```

Gerar benchmarks e salvar resultados:

```bash
python main.py --benchmark --iterations 200 --depth 3 --benchmark-output benchmark_reports
```

Comparar backends de avaliação em lote:

```bash
python main.py --benchmark-evaluators --benchmark-positions 64 --benchmark-evaluator-backends python,cpu-vector
```

## GPU

O projeto ainda não executa MCTS ou Minimax na GPU. O argumento `--workers-gpu` existe como guarda explícita: se usado agora, o CLI detecta fabricantes/backends conhecidos e informa que é necessário implementar um avaliador em lote compatível antes de habilitar aceleração real.

A camada `ai/gpu/` foi preparada para backends como CUDA, DirectML, OpenCL e Metal/MPS. O MCTS continuará controlando a árvore na CPU; a GPU deve entrar na próxima fase como avaliador em lote de tabuleiros.

O contrato de avaliação em lote já existe em `ai/batch_evaluator.py`. A factory `ai/batch_evaluator_factory.py` seleciona o backend solicitado por `--gpu-backend`. Hoje ela usa `PythonBatchEvaluator`, que preserva a heurística atual em CPU. Backends GPU futuros devem implementar o mesmo contrato para pontuar vários tabuleiros de uma vez.

O backend `cpu-vector` já codifica o tabuleiro como um vetor numérico de 64 posições e calcula as mesmas heurísticas do avaliador atual. Ele serve como ponte para backends GPU futuros, que receberão uma representação parecida.

Use `--benchmark-evaluators` para medir tempo e divergência de score entre backends antes de trocar o backend usado pelo MCTS.

O backend `directml` é opcional e usa `torch-directml` quando disponível. Para testar em Windows com GPU DirectX 12 compatível:

```bash
pip install torch-directml
python main.py --mode "MCTS vs Minimax" --workers-gpu 1 --gpu-backend directml
```

Se `torch-directml` não estiver instalado, não conseguir criar o dispositivo DirectML ou não tiver wheel para sua versão do Python, o CLI mostra uma mensagem clara e mantém os backends CPU disponíveis.

No momento, os wheels publicados de `torch-directml` chegam até CPython 3.12. Em Python 3.14, use os backends `python` ou `cpu-vector`, ou crie um ambiente separado com Python 3.12 para testar `torch-directml`:

```bash
py -3.12 -m venv .venv-directml
.venv-directml\Scripts\activate
pip install torch-directml
python main.py --mode "MCTS vs Minimax" --workers-gpu 1 --gpu-backend directml
```

Alternativa para Python 3.14 no Windows: `onnxruntime-directml` publica wheel `cp314-win_amd64` e pode substituir `torch-directml` como backend DirectML futuro:

```bash
pip install onnxruntime-directml
python main.py --mode "MCTS vs Minimax" --workers-gpu 1 --gpu-backend onnx-directml
```

O detector já reconhece `onnx-directml`. A implementação do avaliador ONNX ainda precisa ser adicionada antes de executar a heurística na GPU por esse caminho.
