# Tutorial: Rodando Benchmarks

Este tutorial mostra como executar benchmarks entre agentes e coletar resultados.

## Exemplo rápido

1. Instale dependências (opcionalmente em um virtualenv):

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Execute o benchmark padrão e salve os resultados:

```bash
python main.py --benchmark --iterations 200 --depth 3 --benchmark-output benchmark_reports
```

3. Os relatórios serão gerados em `benchmark_reports/benchmark_report.csv` e `benchmark_reports/benchmark_report.json`.

## Interpretação rápida

- `match_index`: índice da partida no benchmark.
- `agent`: agente que realizou a jogada.
- `captures`: posições capturadas no movimento.
- `states_analyzed`: número aproximado de estados analisados pelo agente.

## Dicas

- Para comparações mais confiáveis, aumente `--iterations` para agentes MCTS.
- Use `--depth` ao comparar Minimax com versões rasas do Hybrid.
