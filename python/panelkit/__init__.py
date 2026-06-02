"""panelkit — fast, from-scratch causal-inference estimators for panel data.

The compiled core lives in :mod:`panelkit._panelkit` (Rust, via PyO3). This
package re-exports a small, sklearn-style Python surface on top of it.
"""

from ._panelkit import version as _version
from .estimators import (
    AugmentedSC,
    CallawaySantAnna,
    CPASC,
    GoodmanBacon,
    MCNNM,
    SunAbraham,
    SyntheticControl,
    SyntheticDiD,
    TWFE,
)

__all__ = [
    "__version__",
    "SyntheticControl",
    "AugmentedSC",
    "SyntheticDiD",
    "MCNNM",
    "CPASC",
    "TWFE",
    "CallawaySantAnna",
    "SunAbraham",
    "GoodmanBacon",
]

__version__ = _version()
