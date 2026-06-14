import importlib.util

from ai.gpu.backends.base import GPUBackendInfo


def detect() -> list[GPUBackendInfo]:
    """Detect NVIDIA/CUDA-oriented Python backends."""
    return [
        _detect_torch_cuda(),
        _detect_cupy(),
        _detect_numba_cuda(),
    ]


def _detect_torch_cuda() -> GPUBackendInfo:
    if importlib.util.find_spec("torch") is None:
        return GPUBackendInfo(
            name="torch-cuda",
            vendor="nvidia",
            framework="torch",
            available=False,
            reason="PyTorch is not installed.",
        )

    try:
        import torch

        if torch.cuda.is_available():
            return GPUBackendInfo(
                name="torch-cuda",
                vendor="nvidia",
                framework="torch",
                available=True,
                reason="PyTorch reports CUDA availability.",
            )
        return GPUBackendInfo(
            name="torch-cuda",
            vendor="nvidia",
            framework="torch",
            available=False,
            reason="PyTorch is installed, but CUDA is not available.",
        )
    except Exception as exc:
        return GPUBackendInfo(
            name="torch-cuda",
            vendor="nvidia",
            framework="torch",
            available=False,
            reason=f"PyTorch CUDA detection failed: {exc}",
        )


def _detect_cupy() -> GPUBackendInfo:
    if importlib.util.find_spec("cupy") is None:
        return GPUBackendInfo(
            name="cupy",
            vendor="nvidia",
            framework="cupy",
            available=False,
            reason="CuPy is not installed.",
        )

    return GPUBackendInfo(
        name="cupy",
        vendor="nvidia",
        framework="cupy",
        available=True,
        reason="CuPy is installed.",
    )


def _detect_numba_cuda() -> GPUBackendInfo:
    if importlib.util.find_spec("numba") is None:
        return GPUBackendInfo(
            name="numba-cuda",
            vendor="nvidia",
            framework="numba",
            available=False,
            reason="Numba is not installed.",
        )

    try:
        from numba import cuda

        if cuda.is_available():
            return GPUBackendInfo(
                name="numba-cuda",
                vendor="nvidia",
                framework="numba",
                available=True,
                reason="Numba reports CUDA availability.",
            )
        return GPUBackendInfo(
            name="numba-cuda",
            vendor="nvidia",
            framework="numba",
            available=False,
            reason="Numba is installed, but CUDA is not available.",
        )
    except Exception as exc:
        return GPUBackendInfo(
            name="numba-cuda",
            vendor="nvidia",
            framework="numba",
            available=False,
            reason=f"Numba CUDA detection failed: {exc}",
        )
