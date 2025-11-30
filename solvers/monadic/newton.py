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
    # Epsilon for derivative to prevent division by zero
    DERIVATIVE_TOLERANCE = 1e-15

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
        self.trace.clear()
        derivative = calculus.get_derivative_of(self.function, step_size)

        for iteration in range(max_iterations):
            function_value = self.function(guess)
            derivative_value = derivative(guess)

            if math.isclose(function_value, 0, abs_tol=tolerance):
                self.trace.final_result = guess
                self.trace.has_converged = True
                return guess

            # Safety Check: Avoid Division by Zero
            if math.isclose(derivative_value, 0, abs_tol=self.DERIVATIVE_TOLERANCE):
                break

            step = NewtonStep(
                iteration=iteration,
                guess=guess,
                function_value=function_value,
                derivative_value=derivative_value,
            )
            self.trace.steps.append(step)

            difference = -function_value / derivative_value
            new_guess = guess + difference
            if abs(difference) < tolerance:
                self.trace.final_result = new_guess
                self.trace.has_converged = True
                return new_guess

            guess = new_guess

        self.trace.final_result = guess
        self.trace.has_converged = False
        if raise_exception_if_no_convergence:
            raise MonadicEquationSolverNotConvergedException(self)

        return guess
