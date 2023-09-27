from typing import Any, List, Literal, Tuple, TypedDict

import numpy as np
import numpy.typing as npt

from pyswarms.backend.handlers import OptionsHandler

# Bounds for constrained optimization
BoundsList = Tuple[List[int], List[int]] | Tuple[List[float], List[float]]
BoundsArray = Tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]
Bounds = BoundsList | BoundsArray

# Velocity clamps
ClampList = Tuple[List[int], List[int]] | Tuple[List[float], List[float]]
ClampArray = Tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]
ClampFloat = Tuple[float, float]
Clamp = ClampArray | ClampList | ClampFloat

# Particle position and velocity types
Position = npt.NDArray[np.floating[Any] | np.integer[Any]]
Velocity = npt.NDArray[np.floating[Any]]

# Handlers
BoundaryStrategy = Literal["nearest", "random", "shrink", "reflective", "intermediate", "periodic"]
OptionsStrategy = Literal["exp_decay", "lin_variation", "random", "nonlin_mod"]

SwarmOption = Literal["c1", "c2", "w"]
class SwarmOptions(TypedDict):
    """Used for initialising options handlers. Options can either be constant or varied using OptionsHandlers.
    Valid values are:
    * float:
        Use a constant value.
    * Tuple[OptionsStrategy, float]:
        Use the default factory with a given strategy and initial value.
    * OptionsHandler:
        Use an already initialised OptionsHandler object.

    Valid option strategies are ["exp_decay", "lin_variation", "random", "nonlin_mod"]

    Attributes
    ----------
    c1 : float|Tuple[OptionsStrategy, float]|OptionsHandler
        cognitive parameter
    c2 : float|Tuple[OptionsStrategy, float]|OptionsHandler
        social parameter
    w : float|Tuple[OptionsStrategy, float]|OptionsHandler
        inertia parameter
    """
    c1: float|Tuple[OptionsStrategy, float]|OptionsHandler
    c2: float|Tuple[OptionsStrategy, float]|OptionsHandler
    w: float|Tuple[OptionsStrategy, float]|OptionsHandler
