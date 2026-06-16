# API - Métricas

Documentação das classes de coleta e relatório de métricas.

::: metrics.metrics_collector.MetricsCollector

## Exemplo: Coletar métricas simples

```python
from metrics.metrics_collector import MetricsCollector

mc = MetricsCollector()
mc.record_move('BLACK', (2,1), (3,2), 0.05)
mc.flush_to_csv('metrics.csv')
```

::: metrics.match_report.MatchReport

## Exemplo: Gerar relatório

```python
from metrics.match_report import MatchReport

report = MatchReport.from_collector(mc)
report.to_json('report.json')
```
