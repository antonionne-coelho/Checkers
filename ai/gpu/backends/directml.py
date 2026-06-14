import importlib.util

from ai.gpu.backends.base import GPUBackendInfo
from ai.python_compat import current_python_tag, is_current_python_tag_supported


SUPPORTED_TORCH_DIRECTML_TAGS = {"cp38", "cp39", "cp310", "cp311", "cp312"}


def detect() -> list[GPUBackendInfo]:
    if not is_current_python_tag_supported(SUPPORTED_TORCH_DIRECTML_TAGS):
        return [
            GPUBackendInfo(
                name="torch-directml",
                vendor="cross-vendor",
                framework="torch-directml",
                available=False,
                reason=(
                    "torch-directml does not publish wheels for "
                    f"{current_python_tag()}; use Python 3.12 or another "
                    "supported CPython version for DirectML."
                ),
            )
        ]

    if importlib.util.find_spec("torch_directml") is None:
        return [
            GPUBackendInfo(
                name="torch-directml",
                vendor="cross-vendor",
                framework="torch-directml",
                available=False,
                reason="torch-directml is not installed.",
            )
        ]

    try:
        import torch_directml

        torch_directml.device()
    except Exception as exc:
        return [
            GPUBackendInfo(
                name="torch-directml",
                vendor="cross-vendor",
                framework="torch-directml",
                available=False,
                reason=f"torch-directml device creation failed: {exc}",
            )
        ]

    return [
        GPUBackendInfo(
            name="torch-directml",
            vendor="cross-vendor",
            framework="torch-directml",
            available=True,
            reason="torch-directml is installed.",
        )
    ]
