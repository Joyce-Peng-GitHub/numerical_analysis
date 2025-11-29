import typing

from solvers.solution_trace import SolutionTrace

UnaryFunction: typing.TypeAlias = typing.Callable[[float], float]


class MonadicEquationSolver:
    def __init__(self, func: UnaryFunction | None = None):
        self.function: UnaryFunction = func
        self.trace = SolutionTrace()
