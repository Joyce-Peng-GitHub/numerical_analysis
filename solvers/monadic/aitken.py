import dataclasses
import math

from solvers.solution_trace import Step, SolutionTrace
from solvers.monadic.monadic_equation_solver import UnaryFunction, MonadicEquationSolver, \
    MonadicEquationSolverNotConvergedException


@dataclasses.dataclass
class AitkenStep(Step):
    x: float
    y: float
    z: float
    slope: float


class AitkenSolver(MonadicEquationSolver):
    DEFAULT_MAX_ITERATIONS = 32

    def __init__(self, function: UnaryFunction, is_fixed_point: bool = False):
        super().__init__(function if is_fixed_point else lambda x: x + function(x))

    def solve(
            self,
            guess: float,
            tolerance: float,
            raise_exception_if_no_convergence: bool = False,
            max_iterations: int = DEFAULT_MAX_ITERATIONS
    ) -> float:
        for iteration in range(max_iterations):
            x = guess
            y = self.function(x)
            if math.isclose(y, x, abs_tol=tolerance):
                self.trace.final_result = x
                self.trace.has_converged = True
                return x
            z = self.function(y)
            slope = (z - y) / (y - x)
            step = AitkenStep(iteration, x, y, z, slope)
            self.trace.steps.append(step)
            guess = (x * z - y ** 2) / (x - 2 * y + z)

        self.trace.final_result = guess
        self.trace.has_converged = False
        if raise_exception_if_no_convergence:
            raise MonadicEquationSolverNotConvergedException(self)
        return guess
