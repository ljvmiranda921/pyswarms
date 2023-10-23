"""
The :mod:`pyswarms.decorators` module implements a decorator that
can be used to simplify the task of writing the cost function for
an optimization run. The decorator can be directly called by using
:code:`@pyswarms.cost`.
"""
from .decorators import cost

__all__ = ["cost"]
