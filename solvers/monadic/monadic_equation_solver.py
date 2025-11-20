import typing

from solvers.monadic.history import History

UnaryFunction: typing.TypeAlias = typing.Callable[[float], float]


class MonadicEquationSolver:
    def __init__(self, func: UnaryFunction | None = None):
        self.function: UnaryFunction = func
        self.history: History = History()
