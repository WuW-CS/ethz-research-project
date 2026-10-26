"""
All functions ignore pixels where `pred` or `ref` is NaN (use the same mask for all methods,
otherwise the comparison is unfair).
"""

import numpy as np