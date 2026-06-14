import importlib.util

from ai.gpu.backends.base import GPUBackendInfo


def detect() -> list[GPUBackendInfo]:
    if importlib.util.find_spec("torch") is None:
        return [
            GPUBackendInfo(
                name="torch-mps",
                vendor="apple",
                framework="torch",
                available=False,
                reason="PyTorch is not installed.",
            )
        ]

    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return [
                GPUBackendInfo(
                    name="torch-mps",
                    vendor="apple",
                    framework="torch",
                    available=True,
                    reason="PyTorch reports Metal/MPS availability.",
                )
            ]
        return [
            GPUBackendInfo(
                name="torch-mps",
                vendor="apple",
                framework="torch",
                available=False,
                reason="PyTorch is installed, but Metal/MPS is not available.",
            )
        ]
    except Exception as exc:
        return [
            GPUBackendInfo(
                name="torch-mps",
                vendor="apple",
                framework="torch",
                available=False,
                reason=f"PyTorch MPS detection failed: {exc}",
            )
        ]
