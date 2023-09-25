"""
The mod:`pyswarms.utils.plotters` module implements various
visualization capabilities to interact with your swarm. Here,
ou can plot cost history and animate your swarm in both 2D or
3D spaces.
"""

# Import from pyswarms
from pyswarms.utils.plotters import formatters, plotters
from pyswarms.utils.plotters.plotters import plot_contour, plot_cost_history, plot_surface

__all__ = ["plotters", "formatters"]
