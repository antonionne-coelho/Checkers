import sys


def current_python_tag() -> str:
    version = sys.version_info
    return f"cp{version.major}{version.minor}"


def is_current_python_tag_supported(supported_tags: set[str]) -> bool:
    return current_python_tag() in supported_tags
