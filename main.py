import argparse
import os
from typing import Optional, Tuple

from agents.base_agent import Agent
from core.board import Board
from core.game import Game
from agents.hybrid_agent import HybridAgent
from agents.human_agent import HumanAgent
from agents.minimax_agent import MinimaxAgent
from agents.minimax_agent_parallel import MinimaxAgentParallel
from agents.mcts_agent import MCTSAgent
from agents.mcts_agent_process import MCTSAgentProcess
from ai.batch_evaluator_benchmark import (
    benchmark_batch_evaluators,
    format_benchmark_results,
)
from ai.batch_evaluator_factory import (
    SUPPORTED_BATCH_BACKENDS,
    BatchEvaluatorUnavailable,
    create_batch_evaluator,
)
from help import get_help_text_br
from metrics.match_report import MatchReport
from metrics.metrics_collector import MetricsCollector


def run_match(
    black_agent: Agent, white_agent: Agent, step: bool = False
) -> Tuple[Optional[str], MetricsCollector]:
    board = Board()
    metrics = MetricsCollector()
    game = Game(
        board=board,
        black_agent=black_agent,
        white_agent=white_agent,
        metrics_collector=metrics,
    )
    winner = game.play_match(step=step)
    return winner, metrics


def create_agent(
    token: str,
    color: str,
    depth: int,
    iterations: int,
    workers: int = 4,
    batch_evaluator=None,
) -> Agent:
    token = token.lower()
    if token == "minimax":
        return MinimaxAgent(color=color, depth=depth)
    if token in {"minimax-parallel", "minimax_parallel"}:
        return MinimaxAgentParallel(color=color, depth=depth, num_workers=workers)
    if token == "mcts":
        return MCTSAgent(
            color=color,
            iterations=iterations,
            batch_evaluator=batch_evaluator,
        )
    if token in {"mcts-process", "mcts_process"}:
        return MCTSAgentProcess(color=color, iterations=iterations, num_workers=workers)
    if token == "hybrid":
        return HybridAgent(
            color=color,
            iterations=iterations,
            minimax_depth=depth,
            batch_evaluator=batch_evaluator,
        )
    if token == "human":
        return HumanAgent(color=color)
    raise ValueError(f"Modo de agente desconhecido: {token}")


def parse_match_mode(mode: str) -> Tuple[str, str]:
    normalized = mode.strip().lower().replace(" ", "")
    if ";" in normalized:
        parts = [part.strip() for part in normalized.split(";") if part.strip()]
    elif "vs" in normalized:
        parts = [part.strip() for part in normalized.split("vs") if part.strip()]
    else:
        raise ValueError("Modo inválido. Use algo como 'MCTS vs Minimax'.")

    if len(parts) != 2:
        raise ValueError(
            "O modo deve conter exatamente dois agentes, separados por 'vs' ou ';'."
        )

    return parts[0], parts[1]


def run_single_mode(
    mode: str,
    depth: int,
    iterations: int,
    step: bool = False,
    workers: int = 4,
    dataset_output: Optional[str] = None,
    workers_gpu: int = 0,
    gpu_backend: str = "auto",
) -> MetricsCollector:
    black_token, white_token = parse_match_mode(mode)
    batch_evaluator = create_batch_evaluator(
        backend=gpu_backend,
        gpu_enabled=workers_gpu > 0,
        strict_gpu=True,
    )
    black_agent = create_agent(
        black_token, "BLACK", depth, iterations, workers, batch_evaluator
    )
    white_agent = create_agent(
        white_token, "WHITE", depth, iterations, workers, batch_evaluator
    )

    print(f"\n=== {black_token.upper()} vs {white_token.upper()} ===")
    if step and "human" not in mode.lower():
        winner, metrics = run_match(black_agent, white_agent, step=step)
    else:
        winner, metrics = run_match(black_agent, white_agent)

    print(f"Winner: {winner}")
    report = MatchReport(metrics)
    report.generate_console_report()
    if dataset_output:
        report.export_learning_dataset_csv(dataset_output)
        print(f"Dataset CSV exportado para: {dataset_output}")
    return metrics


def run_benchmark(
    depth: int,
    iterations: int,
    output_dir: Optional[str] = None,
    workers: int = 4,
    dataset_output: Optional[str] = None,
    workers_gpu: int = 0,
    gpu_backend: str = "auto",
) -> None:
    modes = [
        "Minimax vs Minimax",
        "MCTS vs MCTS",
        "MCTS vs Minimax",
        "Hybrid vs MCTS",
        "Hybrid vs Minimax",
        "Hybrid vs Hybrid",
    ]

    all_metrics = MetricsCollector()

    for mode in modes:
        if "human" in mode.lower():
            print(
                f"\n=== Ignorando modo {mode} no benchmark automático porque é interativo ==="
            )
            continue
        metrics = run_single_mode(
            mode,
            depth,
            iterations,
            workers=workers,
            workers_gpu=workers_gpu,
            gpu_backend=gpu_backend,
        )
        all_metrics.matches.extend(metrics.matches)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "benchmark_report.csv")
        json_path = os.path.join(output_dir, "benchmark_report.json")
        report = MatchReport(all_metrics)
        report.export_to_csv(csv_path)
        report.export_to_json(json_path)
        print(f"Benchmark exportado para: {csv_path} e {json_path}")

    if dataset_output:
        report = MatchReport(all_metrics)
        report.export_learning_dataset_csv(dataset_output)
        print(f"Dataset CSV exportado para: {dataset_output}")


def run_evaluator_benchmark(backends: str, positions: int, seed: int) -> None:
    backend_list = [
        backend.strip() for backend in backends.split(",") if backend.strip()
    ]
    if not backend_list:
        raise ValueError(
            "Informe ao menos um backend em --benchmark-evaluator-backends."
        )

    results = benchmark_batch_evaluators(
        backend_list,
        position_count=positions,
        seed=seed,
    )
    print("=== Batch Evaluator Benchmark ===")
    print(f"Positions: {positions}")
    print(f"Seed: {seed}")
    print(format_benchmark_results(results))


def main():
    parser = argparse.ArgumentParser(
        description="Runs Checkers matches with Minimax, MCTS, Human and Hybrid agents.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "EXAMPLES:\n"
            "  python main.py --gui                                    (interface gráfica Pygame)\n"
            '  python main.py --mode "MCTS vs Minimax" --iterations 100 --depth 3\n'
            '  python main.py --mode "mcts-process vs minimax" --iterations 100 --workers 4\n'
            '  python main.py --mode "minimax-parallel vs minimax" --depth 4 --workers 4\n'
            "  python main.py --benchmark --iterations 200 --depth 4 --benchmark-output reports\n"
            '  python main.py --mode "MCTS vs Minimax" --step\n'
            "  python main.py --help-br\n"
        ),
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Inicia a interface gráfica Pygame (recomendado).",
    )
    parser.add_argument(
        "--mode",
        default="Minimax vs Minimax",
        help=(
            "Match mode. Valid agents: minimax, minimax-parallel, mcts, mcts-process, hybrid, human.\n"
            "Separate agents with 'vs' or ';', e.g.: 'MCTS vs Minimax'."
        ),
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run a sequence of benchmark matches between automatic agents (generates report if --benchmark-output is used).",
    )
    parser.add_argument(
        "--benchmark-evaluators",
        action="store_true",
        help="Compare batch evaluator backends on deterministic board positions.",
    )
    parser.add_argument(
        "--benchmark-evaluator-backends",
        default="python,cpu-vector",
        help="Comma-separated batch evaluator backends to benchmark.",
    )
    parser.add_argument(
        "--benchmark-positions",
        type=int,
        default=32,
        help="Number of deterministic board positions for evaluator benchmarks.",
    )
    parser.add_argument(
        "--benchmark-seed",
        type=int,
        default=0,
        help="Random seed used to generate evaluator benchmark positions.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of MCTS iterations for MCTS and Hybrid agents.",
    )
    parser.add_argument(
        "--step",
        action="store_true",
        help="Show each turn and wait for Enter to proceed in matches without human player.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Minimax search depth and minimax version of Hybrid.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes for process-based agents.",
    )
    parser.add_argument(
        "--workers-gpu",
        type=int,
        default=0,
        help="Reserved GPU worker request. Fails fast until a GPU evaluator is implemented.",
    )
    parser.add_argument(
        "--gpu-backend",
        default="auto",
        choices=sorted(SUPPORTED_BATCH_BACKENDS),
        help="Batch evaluator backend to use when GPU workers are requested.",
    )
    parser.add_argument(
        "--benchmark-output",
        default=None,
        help="Output directory for CSV/JSON benchmark report.",
    )
    parser.add_argument(
        "--dataset-output",
        default=None,
        help="Output CSV path for a move-level learning dataset.",
    )
    parser.add_argument(
        "--help-br",
        action="store_true",
        help="Show help message in Brazilian Portuguese.",
    )
    args = parser.parse_args()

    if args.gui:
        try:
            from ui.pygame_ui import main as gui_main
        except ModuleNotFoundError as exc:
            missing_package = exc.name or "uma dependência"
            parser.error(
                f"Não foi possível iniciar a interface gráfica: pacote '{missing_package}' não encontrado. "
                "Instale as dependências com `pip install -r requirements.txt` e tente novamente."
            )
        gui_main()
        return

    if args.help_br:
        print(get_help_text_br())
        return

    if args.benchmark_evaluators:
        try:
            run_evaluator_benchmark(
                args.benchmark_evaluator_backends,
                args.benchmark_positions,
                args.benchmark_seed,
            )
        except (BatchEvaluatorUnavailable, ValueError) as exc:
            parser.error(str(exc))
        return

    if args.benchmark:
        try:
            run_benchmark(
                depth=args.depth,
                iterations=args.iterations,
                output_dir=args.benchmark_output,
                workers=args.workers,
                dataset_output=args.dataset_output,
                workers_gpu=args.workers_gpu,
                gpu_backend=args.gpu_backend,
            )
        except BatchEvaluatorUnavailable as exc:
            parser.error(str(exc))
        return

    try:
        run_single_mode(
            args.mode,
            args.depth,
            args.iterations,
            step=args.step,
            workers=args.workers,
            dataset_output=args.dataset_output,
            workers_gpu=args.workers_gpu,
            gpu_backend=args.gpu_backend,
        )
    except BatchEvaluatorUnavailable as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
