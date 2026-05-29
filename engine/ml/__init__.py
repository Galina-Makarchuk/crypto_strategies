"""ML scaffolding for the swing-pivot prediction model.

Three sub-modules:

- ``labels``   — oracle swing pivots used as supervised targets.
- ``features`` — causal feature transformers (every column is read-safe at bar i).
- ``splits``   — purged k-fold with embargo, the right CV for sequential price data.

Strategy code lives in ``strategies/swing_zigzag_ml.py``. This package only
contains the offline training scaffolding.
"""

from .features import (
    FEATURE_COLUMNS,
    FEATURE_COLUMNS_T3,
    build_feature_frame,
    build_feature_frame_t3,
)
from .labels import LABEL_HOLD, LABEL_LONG, LABEL_SHORT, oracle_swing_labels
from .order_flow import (
    OFI_FEATURE_COLUMNS,
    build_orderflow_features,
    merge_orderflow_features,
)
from .splits import PurgedKFold, walk_forward_iter

__all__ = [
    "FEATURE_COLUMNS",
    "FEATURE_COLUMNS_T3",
    "LABEL_HOLD",
    "LABEL_LONG",
    "LABEL_SHORT",
    "OFI_FEATURE_COLUMNS",
    "PurgedKFold",
    "build_feature_frame",
    "build_feature_frame_t3",
    "build_orderflow_features",
    "merge_orderflow_features",
    "oracle_swing_labels",
    "walk_forward_iter",
]
