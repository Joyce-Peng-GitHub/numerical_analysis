import dataclasses
import math

from solvers.monadic.calculus import DEFAULT_STEP_SIZE
from solvers.monadic.monadic_equation_solver import UnaryFunction, MonadicEquationSolver, \
    MonadicEquationSolverNotConvergedException
from solvers.solution_trace import Step
import solvers.monadic.calculus as calculus


@dataclasses.dataclass
class NewtonStep(Step):
    guess: float
    function_value: float
    derivative_value: float


class NewtonSolver(MonadicEquationSolver):
    DEFAULT_MAX_ITERATIONS = 128

    def __init__(self, function: UnaryFunction | None = None):
        super().__init__(function)

    def solve(
            self,
            guess: float,
            tolerance: float,
            raise_exception_if_no_convergence: bool = False,
            max_iterations: int = DEFAULT_MAX_ITERATIONS,
            step_size: float = DEFAULT_STEP_SIZE
    ):
        derivative = calculus.get_derivative_of(self.function, step_size)

        for iteration in range(max_iterations):
            function_value, derivative_value = self.function(guess), derivative(guess)

            if math.isclose(function_value, 0, abs_tol=tolerance):
                self.trace.final_result = guess
                self.trace.has_converged = True
                return guess

            step = NewtonStep(iteration, guess, function_value, derivative_value)
            self.trace.steps.append(step)
            guess -= function_value / derivative_value

        self.trace.final_result = guess
        self.trace.has_converged = False
        if raise_exception_if_no_convergence:
            raise MonadicEquationSolverNotConvergedException(self)
        return guess
