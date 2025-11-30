import dataclasses
import math

from solvers.monadic.interval import Interval
from solvers.monadic.monadic_equation_solver import UnaryFunction, MonadicEquationSolver
from solvers.solution_trace import Step


@dataclasses.dataclass
class BisectionStep(Step):
    left: float
    right: float
    middle: float
    middle_function_value: float


class BisectionSolver(MonadicEquationSolver):
    def __init__(self, function: UnaryFunction | None = None):
        super().__init__(function)

    def solve(self, interval: Interval, tolerance: float) -> float:
        left, right = interval.left, interval.right

        if not interval.is_finite():
            raise ValueError(f"Bisection requires a finite interval: got left={left:%r}, right={right:%r}")

        left_function_value, right_function_value = self.function(left), self.function(right)

        self.trace.clear()  # initialize solution trace

        if interval.include_left and left_function_value == 0:
            self.trace.final_result = left
            self.trace.has_converged = True
            return left
        if interval.include_right and right_function_value == 0:
            self.trace.final_result = left
            self.trace.has_converged = True
            return right

        left_function_value_is_negative: bool = (left_function_value < 0)
        if left_function_value_is_negative == (right_function_value < 0):
            raise ValueError(f"Function values at interval endpoints must have opposite signs: "
                             f"f({left:.6g})={left_function_value:.6g}, f({right:.6g})={right_function_value:.6g}")

        iteration = 0
        while right - left > tolerance:
            middle: float = (left + right) / 2
            middle_function_value: float = self.function(middle)
            middle_function_value_is_negative: bool = (middle_function_value < 0)

            # save step to history
            step = BisectionStep(iteration, left, right, middle, middle_function_value)
            self.trace.steps.append(step)

            if math.isclose(middle_function_value, 0, abs_tol=tolerance):
                self.trace.final_result = middle
                self.trace.has_converged = True
                return middle
            if middle_function_value_is_negative == left_function_value_is_negative:
                left = middle
                left_function_value_is_negative = middle_function_value_is_negative
            else:
                right = middle

            iteration += 1

        result = (left + right) / 2
        self.trace.final_result = result
        self.trace.has_converged = True
        return result
