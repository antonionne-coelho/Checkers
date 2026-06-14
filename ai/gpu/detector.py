import platform
import subprocess
from dataclasses import dataclass

from ai.gpu.backends.base import GPUBackendInfo
from ai.gpu.backends import cuda, directml, metal, onnx_directml, opencl


@dataclass(frozen=True)
class GPUDetectionResult:
    vendors: list[str]
    backends: list[GPUBackendInfo]

    @property
    def available_backends(self) -> list[GPUBackendInfo]:
        return [backend for backend in self.backends if backend.available]

    @property
    def preferred_backend(self) -> GPUBackendInfo | None:
        for vendor in self.vendors:
            for backend in self.available_backends:
                if backend.vendor == vendor or backend.vendor == "cross-vendor":
                    return backend
        return self.available_backends[0] if self.available_backends else None


def detect_gpu_environment() -> GPUDetectionResult:
    return GPUDetectionResult(
        vendors=detect_gpu_vendors(),
        backends=[
            *cuda.detect(),
            *directml.detect(),
            *onnx_directml.detect(),
            *opencl.detect(),
            *metal.detect(),
        ],
    )


def detect_gpu_vendors() -> list[str]:
    names = _detect_gpu_names()
    vendors = []
    for name in names:
        normalized = name.lower()
        if "nvidia" in normalized:
            vendors.append("nvidia")
        elif "amd" in normalized or "radeon" in normalized:
            vendors.append("amd")
        elif "intel" in normalized:
            vendors.append("intel")
        elif "apple" in normalized:
            vendors.append("apple")

    return list(dict.fromkeys(vendors))


def validate_gpu_workers_request(workers_gpu: int) -> None:
    if workers_gpu <= 0:
        return

    environment = detect_gpu_environment()
    backend = environment.preferred_backend
    backend_text = backend.name if backend is not None else "nenhum backend disponível"
    vendor_text = ", ".join(environment.vendors) if environment.vendors else "GPU não identificada"

    raise RuntimeError(
        "--workers-gpu ainda não executa agentes na GPU. "
        f"GPU detectada: {vendor_text}. Backend preferido: {backend_text}. "
        "Use --workers para paralelismo CPU por enquanto; a próxima fase é "
        "implementar um avaliador em lote que use o backend escolhido."
    )


def _detect_gpu_names() -> list[str]:
    system = platform.system().lower()
    if system == "windows":
        return _detect_windows_gpu_names()
    if system == "darwin":
        return ["Apple GPU"]
    return _detect_linux_gpu_names()


def _detect_windows_gpu_names() -> list[str]:
    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
    ]
    return _run_name_command(command)


def _detect_linux_gpu_names() -> list[str]:
    names = _run_name_command(["lspci"])
    return [
        line
        for line in names
        if "vga" in line.lower() or "3d controller" in line.lower()
    ]


def _run_name_command(command: list[str]) -> list[str]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except Exception:
        return []

    if completed.returncode != 0:
        return []

    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]
