"""panelkit — fast, from-scratch causal-inference estimators for panel data.

The compiled core lives in :mod:`panelkit._panelkit` (Rust, via PyO3). This
package re-exports a small, sklearn-style Python surface on top of it.
"""

from ._panelkit import version as _version
from .estimators import AugmentedSC, MCNNM, SyntheticControl, SyntheticDiD

__all__ = [
    "__version__",
    "SyntheticControl",
    "AugmentedSC",
    "SyntheticDiD",
    "MCNNM",
]

__version__ = _version()
