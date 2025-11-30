import copy as cp
from typing import Tuple, Any


class MonadicEquationVisualizer:
    """
    Base class for visualizing iterative root-finding methods (Monadic Equation Solvers).
    Implements a Factory Pattern in __new__ to automatically dispatch to the correct
    visualizer subclass based on the provided solver instance.
    """
    DEFAULT_SAMPLE_NUM = 256
    DEFAULT_FIGURE_SIZE = (16, 9)
    DEFAULT_INTERVAL_MS = 1000
    DEFAULT_TICK_SIZE = 12
    DEFAULT_LABEL_SIZE = 16
    DEFAULT_TITLE_SIZE = 20

    def __new__(cls, solver: Any):
        """
        Factory method: If instantiated directly, return an instance of the 
        appropriate subclass based on the solver's type.
        """
        if cls is MonadicEquationVisualizer:
            solver_type = type(solver).__name__

            # Local imports are used here to prevent circular import errors 
            # (since subclasses import this base class).
            # We assume these files are in the same package (relative import).

            if solver_type == 'BisectionSolver':
                from .bisection_visualizer import BisectionVisualizer
                return super().__new__(BisectionVisualizer)

            elif solver_type == 'NewtonSolver':
                from .newton_visualizer import NewtonVisualizer
                return super().__new__(NewtonVisualizer)

            elif solver_type == 'NewtonDownhillSolver':
                from .newton_downhill_visualizer import NewtonDownhillVisualizer
                return super().__new__(NewtonDownhillVisualizer)

            elif solver_type == 'AitkenSolver':
                from .aitken_visualizer import AitkenVisualizer
                return super().__new__(AitkenVisualizer)

            else:
                raise ValueError(f"No compatible visualizer found for solver type: {solver_type}")

        # If cls is already a subclass (e.g. NewtonVisualizer(solver)), 
        # just create the object normally.
        return super().__new__(cls)

    def __init__(self, solver: Any):
        """
        Initialize the visualizer with a deep copy of the solver to prevent
        modifying the original solver instance during visualization routines.
        """
        # Note: When __new__ returns a subclass instance, Python automatically 
        # calls this __init__ method on that instance.
        self.solver = cp.deepcopy(solver)

    def animate(
            self,
            sample_num: int = DEFAULT_SAMPLE_NUM,
            figure_size: Tuple[int, int] = DEFAULT_FIGURE_SIZE,
            tick_size: int = DEFAULT_TICK_SIZE,
            label_size: int = DEFAULT_LABEL_SIZE,
            title_size: int = DEFAULT_TITLE_SIZE,
            interval_ms: int = DEFAULT_INTERVAL_MS
    ):
        """
        Abstract method to generate the animation.
        """
        raise NotImplementedError("Subclasses must implement the animate method.")
