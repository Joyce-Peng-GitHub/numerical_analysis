"""Expose solver classes from submodules in this package."""

import pkgutil
import importlib
import inspect
from types import ModuleType

__all__ = []

for finder, name, ispkg in pkgutil.iter_modules(__path__):
    try:
        mod: ModuleType = importlib.import_module(f"{__name__}.{name}")
    except Exception:
        # Skip modules that cannot be imported to avoid breaking package import
        continue

    for attr_name, attr in vars(mod).items():
        # Export classes that follow the "<Name>Solver" convention
        if not attr_name.startswith("_") and inspect.isclass(attr):
            globals()[attr_name] = attr
            __all__.append(attr_name)

__all__ = sorted(set(__all__))
