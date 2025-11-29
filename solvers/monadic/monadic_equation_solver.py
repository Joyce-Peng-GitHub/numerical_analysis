import typing

from solvers.solution_trace import SolutionTrace

UnaryFunction: typing.TypeAlias = typing.Callable[[float], float]


class MonadicEquationSolver:
    def __init__(self, func: UnaryFunction | None = None):
        self.function: UnaryFunction = func
        self.trace = SolutionTrace()


class MonadicEquationSolverNotConvergedException(Exception):
    def __init__(self, solver: MonadicEquationSolver):
        super().__init__(
            f"Method {solver.__class__.__name__} did not converge after {len(solver.trace.steps)} iterations."
        )
