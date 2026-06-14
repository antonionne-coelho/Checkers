import importlib.util

from ai.gpu.backends.base import GPUBackendInfo


def detect() -> list[GPUBackendInfo]:
    if importlib.util.find_spec("onnxruntime") is None:
        return [
            GPUBackendInfo(
                name="onnx-directml",
                vendor="cross-vendor",
                framework="onnxruntime-directml",
                available=False,
                reason="onnxruntime-directml is not installed.",
            )
        ]

    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
    except Exception as exc:
        return [
            GPUBackendInfo(
                name="onnx-directml",
                vendor="cross-vendor",
                framework="onnxruntime-directml",
                available=False,
                reason=f"ONNX Runtime provider detection failed: {exc}",
            )
        ]

    if "DmlExecutionProvider" in providers:
        return [
            GPUBackendInfo(
                name="onnx-directml",
                vendor="cross-vendor",
                framework="onnxruntime-directml",
                available=True,
                reason="ONNX Runtime reports DmlExecutionProvider availability.",
            )
        ]

    return [
        GPUBackendInfo(
            name="onnx-directml",
            vendor="cross-vendor",
            framework="onnxruntime-directml",
            available=False,
            reason=(
                "ONNX Runtime is installed, but DmlExecutionProvider is not "
                f"available. Providers: {providers}"
            ),
        )
    ]
