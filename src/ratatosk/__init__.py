"""Copyright (c) 2026 Neil Shephard. All rights reserved.

ratatosk: Validated electrochemical data-processing, analysis and integrity tool for battery research.
"""

from __future__ import annotations

from importlib.metadata import version

from packaging.version import Version

__version__ = version("ratatosk")
__release__ = ".".join(__version__.split(".")[:-2])
RATATOSK_VERSION = Version(__version__)
if RATATOSK_VERSION.is_prerelease and RATATOSK_VERSION.is_devrelease:
    RATATOSK_BASE_VERSION = str(RATATOSK_VERSION.base_version)
    RATATOSK_COMMIT = str(RATATOSK_VERSION).split("+")[1]
else:
    RATATOSK_BASE_VERSION = str(RATATOSK_VERSION)
    RATATOSK_COMMIT = ""

__all__ = ["__version__"]
