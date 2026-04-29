"""
Root package initialization for the custom robot extension.
"""

import os
import toml


# Convenience function to get the extension path
def get_extension_path():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load extension config
try:
    config_path = os.path.join(get_extension_path(), "pyproject.toml")
    config = toml.load(config_path)
    __version__ = config["project"]["version"]
except Exception:
    __version__ = "0.1.0"

# Import submodules to trigger registration
from . import tasks as tasks, config as config, wrappers as wrappers  # noqa: E402
