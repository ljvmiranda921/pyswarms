# Import standard library
from typing import Any, List, Tuple

# Import modules
import numpy as np
import numpy.typing as npt

# Bounds for constrained optimization
BoundsList = Tuple[List[float], List[float]]
BoundsArray = Tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]
Bounds = BoundsList | BoundsArray

# Velocity clamps
ClampArray = Tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]
ClampList = Tuple[List[float], List[float]]
ClampFloat = Tuple[float, float]
Clamp = ClampArray | ClampList | ClampFloat

# Particle position and velocity types
Position = npt.NDArray[np.floating[Any] | np.integer[Any]]
Velocity = npt.NDArray[np.floating[Any]]
