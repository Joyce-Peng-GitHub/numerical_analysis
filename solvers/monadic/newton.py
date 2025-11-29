import dataclasses
import math

from solvers.monadic.monadic_equation_solver import UnaryFunction, MonadicEquationSolver
from solvers.solution_trace import Step


@dataclasses.dataclass
class NewtonStep(Step):
    x: float
    function_value: float
    derivative_value: float


class NewtonSolver(MonadicEquationSolver):
    def __init__(self, function: UnaryFunction | None = None, derivative: UnaryFunction | None = None):
        super().__init__(function)
        self.derivative: UnaryFunction = derivative

    def solve(self, initial_guess: float, tolerance: float, max_iterations: int = 100) -> float:
        if not math.isfinite(initial_guess):
            raise ValueError(f"Newton's method requires a finite initial guess: got {initial_guess}")

        self.trace.clear()  # initialize solution trace

        x = initial_guess
        iteration = 0

        while iteration < max_iterations:
            function_value = self.function(x)
            derivative_value = self.derivative(x)

            # save step to history
            step = NewtonStep(iteration, x, function_value, derivative_value)
            self.trace.steps.append(step)

            if math.isclose(function_value, 0, abs_tol=tolerance):
                self.trace.final_result = x
                self.trace.has_converged = True
                return x

            if math.isclose(derivative_value, 0, abs_tol=1e-15):
                raise ValueError(f"Derivative is zero at x={x}, Newton's method cannot continue")

            x_new = x - function_value / derivative_value

            if abs(x_new - x) < tolerance:
                self.trace.final_result = x_new
                self.trace.has_converged = True
                return x_new

            x = x_new
            iteration += 1

        # Did not converge within max_iterations
        self.trace.final_result = x
        self.trace.has_converged = False
        raise ValueError(f"Newton's method did not converge within {max_iterations} iterations")
