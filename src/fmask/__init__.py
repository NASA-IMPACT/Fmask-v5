import importlib as _importlib
import sys as _sys
from importlib.machinery import ModuleSpec as _ModuleSpec

from ._paths import FMASK_DATA_ENV, get_data_root

__version__ = "5.0.1"

# ---------------------------------------------------------------------------
# Pickle compatibility
# ---------------------------------------------------------------------------
# The pre-packaged .pk / .pt model files were serialised when the library
# modules lived as bare top-level names (e.g. ``lightgbmlib``).  Unpickling
# them calls ``import lightgbmlib``, which now fails.  This finder intercepts
# those imports and redirects them to their new ``fmask.*`` locations without
# eagerly importing the heavy dependencies (torch, gdal, …).

_COMPAT_MODULES = frozenset((
    "bitlib",
    "constant",
    "fmasklib",
    "lightgbmlib",
    "phylib",
    "predictor",
    "satellite",
    "unetlib",
    "utils",
))


class _PickleCompatLoader:
    def create_module(self, spec):
        # Return the already-aliased module rather than creating a blank one.
        return _sys.modules.get(spec.name)

    def exec_module(self, module):
        pass  # module is already fully initialised


class _PickleCompatFinder:
    def find_spec(self, fullname, path, target=None):
        if fullname not in _COMPAT_MODULES:
            return None
        # Import the real submodule and register the bare name as an alias.
        real = _importlib.import_module(f"fmask.{fullname}")
        _sys.modules[fullname] = real
        return _ModuleSpec(fullname, _PickleCompatLoader())


_sys.meta_path.append(_PickleCompatFinder())
