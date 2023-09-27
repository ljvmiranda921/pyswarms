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
    c1: float|OptionsStrategy|OptionsHandler
    c2: float|OptionsStrategy|OptionsHandler
    w: float|OptionsStrategy|OptionsHandler
