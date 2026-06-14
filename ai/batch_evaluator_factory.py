from dataclasses import dataclass

from ai.batch_evaluator import BatchEvaluator, PythonBatchEvaluator
from ai.directml_batch_evaluator import DirectMLBatchEvaluator
from ai.gpu import detect_gpu_environment
from ai.vector_batch_evaluator import VectorizedCPUBatchEvaluator


SUPPORTED_BATCH_BACKENDS = {
    "auto",
    "cpu-vector",
    "python",
    "cuda",
    "directml",
    "onnx-directml",
    "opencl",
    "metal",
}


class BatchEvaluatorUnavailable(RuntimeError):
    """Raised when a requested batch evaluator backend cannot be created."""


@dataclass(frozen=True)
class BatchEvaluatorRequest:
    backend: str = "auto"
    gpu_enabled: bool = False
    strict_gpu: bool = True


def create_batch_evaluator(
    backend: str = "auto",
    gpu_enabled: bool = False,
    strict_gpu: bool = True,
) -> BatchEvaluator:
    request = BatchEvaluatorRequest(
        backend=_normalize_backend(backend),
        gpu_enabled=gpu_enabled,
        strict_gpu=strict_gpu,
    )

    if request.backend not in SUPPORTED_BATCH_BACKENDS:
        raise ValueError(f"Backend de avaliação desconhecido: {backend}")

    if request.backend == "python" or (
        request.backend == "auto" and not request.gpu_enabled
    ):
        return PythonBatchEvaluator()

    if request.backend == "cpu-vector":
        return VectorizedCPUBatchEvaluator()

    environment = detect_gpu_environment()
    selected_backend = _select_gpu_backend(request.backend, environment)
    if selected_backend is None:
        if request.strict_gpu:
            raise BatchEvaluatorUnavailable(
                _build_unavailable_message(request.backend, environment)
            )
        return PythonBatchEvaluator()

    if selected_backend.name == "torch-directml":
        return DirectMLBatchEvaluator()

    if request.strict_gpu:
        raise BatchEvaluatorUnavailable(
            f"O backend GPU '{selected_backend.name}' foi detectado, mas ainda não "
            "existe implementação de BatchEvaluator para ele."
        )
    return PythonBatchEvaluator()


def _normalize_backend(backend: str) -> str:
    return backend.strip().lower().replace("_", "-")


def _select_gpu_backend(backend: str, environment):
    if backend == "auto":
        return environment.preferred_backend

    backend_names = _backend_aliases().get(backend, {backend})
    for candidate in environment.available_backends:
        if candidate.name in backend_names:
            return candidate
    return None


def _build_unavailable_message(backend: str, environment=None) -> str:
    reason_text = _backend_reason_text(backend, environment)
    if backend == "auto":
        message = (
            "Nenhum backend GPU disponível para avaliação em lote. "
            "Use --gpu-backend python, --gpu-backend cpu-vector, ou rode sem "
            "--workers-gpu por enquanto. Em Python 3.14 no Windows, "
            "considere --gpu-backend onnx-directml quando o pacote "
            "onnxruntime-directml estiver instalado."
        )
        return f"{message} {reason_text}".strip()

    message = (
        f"O backend GPU '{backend}' não está disponível para avaliação em lote. "
        "Instale/configure o backend correspondente ou use --gpu-backend python. "
        "Em Python 3.14 no Windows, onnxruntime-directml é uma alternativa "
        "compatível para DirectML."
    )
    return f"{message} {reason_text}".strip()


def _backend_reason_text(backend: str, environment) -> str:
    if environment is None:
        return ""

    if backend == "auto":
        candidates = environment.backends
    else:
        backend_names = _backend_aliases().get(backend, {backend})
        candidates = [
            candidate
            for candidate in environment.backends
            if candidate.name in backend_names
        ]

    reasons = [f"{candidate.name}: {candidate.reason}" for candidate in candidates]
    if not reasons:
        return ""
    return "Detalhes: " + "; ".join(reasons)


def _backend_aliases() -> dict[str, set[str]]:
    return {
        "cuda": {"torch-cuda", "cupy", "numba-cuda"},
        "directml": {"torch-directml"},
        "onnx-directml": {"onnx-directml"},
        "opencl": {"pyopencl"},
        "metal": {"torch-mps"},
    }
