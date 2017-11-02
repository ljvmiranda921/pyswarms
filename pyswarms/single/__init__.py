"""
The :mod:`pyswarms.single` module implements various techniques in
continuous single-objective optimization. These require only one
objective function that can be optimized in a continuous space.
"""

from .global_best import GlobalBestPSO
from .local_best import LocalBestPSO

__all__ = [
    "GlobalBestPSO",
    "LocalBestPSO"
    ]
