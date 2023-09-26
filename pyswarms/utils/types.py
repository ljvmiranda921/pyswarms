from typing import Any, List, Tuple, TypedDict

import numpy as np
import numpy.typing as npt

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


class SwarmOptions(TypedDict):
    c1: float
    c2: float
    w: float
