from typing import TYPE_CHECKING, Any, Literal, Tuple, TypedDict

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from pyswarms.backend.handlers import OptionsHandler

# Bounds for constrained optimization
Bound = int|float|Tuple[int|float, ...]|npt.NDArray[Any]
Bounds = Tuple[Bound, Bound]

# Velocity clamps
Clamp = Tuple[Bound, Bound]

# Particle position and velocity types
Position = npt.NDArray[np.floating[Any] | np.integer[Any]]
Velocity = npt.NDArray[np.floating[Any]]

# Handlers
BoundaryStrategy = Literal["nearest", "random", "shrink", "reflective", "intermediate", "periodic"]
VelocityStrategy = Literal["unmodified", "adjust", "invert", "zero"]
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

    c1: float | Tuple[OptionsStrategy, float] | "OptionsHandler"
    c2: float | Tuple[OptionsStrategy, float] | "OptionsHandler"
    w: float | Tuple[OptionsStrategy, float] | "OptionsHandler"
