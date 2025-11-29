import dataclasses
import math

from solvers.monadic import calculus
from solvers.monadic.calculus import DEFAULT_STEP_SIZE
from solvers.solution_trace import Step, SolutionTrace
from solvers.monadic.monadic_equation_solver import UnaryFunction, MonadicEquationSolver, \
    MonadicEquationSolverNotConvergedException


@dataclasses.dataclass
class NewtonDownhillStep(Step):
    x: float
    x_function_value: float
    x_derivative_value: float
    damping_factor: float


class NewtonDownhillSolver(MonadicEquationSolver):
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
    ) -> float:
        derivative = calculus.get_derivative_of(self.function, step_size)
        for iteration in range(max_iterations):
            x = guess
            x_function_value = self.function(x)
            if math.isclose(x_function_value, 0, abs_tol=tolerance):
                self.trace.final_result = x
                self.trace.has_converged = True
                return x
            x_derivative_value = derivative(x)
            damping_factor_denominator = 1
            while (abs(self.function(x - x_function_value / (damping_factor_denominator * x_derivative_value))) >=
                   abs(x_function_value) and not
                   math.isclose(x_function_value / (damping_factor_denominator * x_derivative_value), 0,
                                abs_tol=tolerance)):
                damping_factor_denominator <<= 1
            step = NewtonDownhillStep(iteration, x, x_function_value, x_derivative_value, damping_factor_denominator)
            self.trace.steps.append(step)
            guess = x - x_function_value / (damping_factor_denominator * x_derivative_value)

        self.trace.final_result = guess
        self.trace.has_converged = False
        if raise_exception_if_no_convergence:
            raise MonadicEquationSolverNotConvergedException(self)
        return guess
