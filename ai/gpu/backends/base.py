from dataclasses import dataclass


@dataclass(frozen=True)
class GPUBackendInfo:
    """Describes one GPU execution backend supported by the project."""

    name: str
    vendor: str
    framework: str
    available: bool
    reason: str
