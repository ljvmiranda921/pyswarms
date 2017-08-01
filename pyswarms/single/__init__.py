"""
The :mod:`pyswarms.single` module implements various techniques in
continuous single-objective optimization. These require only one
objective function that can be optimized in a continuous space.
"""

from .gb import GBestPSO
from .lb import LBestPSO

__all__ = [
    "GBestPSO",
    "LBestPSO"
    ]