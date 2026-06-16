# Checkers

Bem-vindo à documentação do projeto **Checkers**, uma implementação em Python de um jogo de damas com suporte a agentes de IA.

## Visão Geral

Este projeto inclui:

- Implementação de regras de damas e representação do tabuleiro
- Agentes de IA `Minimax`, `MCTS` e `Hybrid`
- Agent em modo humano para interação no terminal
- Coleta de métricas de partidas e exportação em CSV/JSON
- Interface de linha de comando com `main.py`

## Começando

Instale as dependências do MkDocs para gerar a documentação localmente:

```bash
pip install -r requirements.txt
```

Inicie o servidor de documentação:

```bash
mkdocs serve
```

Abra `http://127.0.0.1:8000` no navegador para ver o site de documentação.

## Principais tópicos

- `Uso`: como executar partidas, opções de linha de comando e exemplos
- `Arquitetura`: componentes e fluxo de jogo
- `API`: documentação das classes principais e dos agentes
- `Estrutura do Projeto`: visão geral dos arquivos e pastas
