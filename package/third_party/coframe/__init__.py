"""CoFrame: self-validating adaptive frame meshes for video diffusion."""

from .config import CoFrameConfig
from .controller import AdaptiveMeshController
from .ode_budget import ODEPathBudgetController
from .selection import coverage_interleaved_select, rhyme_select, uniform_select

__all__ = [
    "AdaptiveMeshController",
    "CoFrameConfig",
    "ODEPathBudgetController",
    "coverage_interleaved_select",
    "rhyme_select",
    "uniform_select",
]
__version__ = "0.1.0"
