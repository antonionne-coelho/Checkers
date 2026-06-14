import importlib.util

from ai.gpu.backends.base import GPUBackendInfo


def detect() -> list[GPUBackendInfo]:
    if importlib.util.find_spec("pyopencl") is None:
        return [
            GPUBackendInfo(
                name="pyopencl",
                vendor="cross-vendor",
                framework="pyopencl",
                available=False,
                reason="PyOpenCL is not installed.",
            )
        ]

    return [
        GPUBackendInfo(
            name="pyopencl",
            vendor="cross-vendor",
            framework="pyopencl",
            available=True,
            reason="PyOpenCL is installed.",
        )
    ]
