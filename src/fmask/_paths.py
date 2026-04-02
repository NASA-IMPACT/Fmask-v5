"""Ancillary-data path resolution for Fmask.

The ``data/`` and ``model/`` directories are not bundled with the Python
source; they must be installed separately (see ``fmask-data install``).
By default they are expected to live inside the ``fmask`` package directory
so that a conda/pip install and an editable checkout both work identically.
Set the ``FMASK_DATA`` environment variable to point elsewhere when needed.
"""

import os
from pathlib import Path

#: Environment variable that overrides the ancillary-data root directory.
FMASK_DATA_ENV = "FMASK_DATA"


def get_data_root() -> Path:
    """Return the root directory that contains ``data/`` and ``model/``.

    Resolution order:

    1. The ``FMASK_DATA`` environment variable (if set and non-empty).
    2. The ``fmask`` package directory — i.e. the directory that contains
       this file.  ``data/`` and ``model/`` are expected to be siblings of
       the Python source files, which works for both editable installs and
       normal installed packages.
    """
    env = os.environ.get(FMASK_DATA_ENV, "").strip()
    if env:
        return Path(env)
    return Path(__file__).parent
