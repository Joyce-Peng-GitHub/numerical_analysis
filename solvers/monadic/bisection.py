from solvers.monadic.interval import *
from solvers.monadic.monadic_equation_solver import *


class BisectionSolver(MonadicEquationSolver):
    def __init__(self, func: UnaryFunction | None = None):
        super().__init__(func)

    def solve(self, interval: Interval, tolerance: float) -> float:
        left, right = interval.left, interval.right

        if not interval.is_finite():
            raise ValueError(f"Bisection requires a finite interval: got left={left:%r}, right={right:%r}")

        function_left_value, function_right_value = self.function(left), self.function(right)

        # Initialize history
        self.history.clear()
        self.history.init = (left, right, function_left_value, function_right_value)

        if interval.include_left and function_left_value == 0:
            self.history.final = (left, right, function_left_value, function_right_value)
            return left
        if interval.include_right and function_right_value == 0:
            self.history.final = (left, right, function_left_value, function_right_value)
            return right

        function_left_value_is_negative: bool = (function_left_value < 0)
        if function_left_value_is_negative == (function_right_value < 0):
            raise ValueError(f"Function values at interval endpoints must have opposite signs: "
                             f"f({left:%r})={function_left_value:%r}, f({right:%r})={function_right_value:%r}")

        while right - left > tolerance:
            middle: float = (left + right) / 2
            function_middle_value: float = self.function(middle)
            function_middle_value_is_negative: bool = (function_middle_value < 0)

            # save step to history
            self.history.steps.append((left, right, middle, function_middle_value))

            if math.isclose(function_middle_value, 0, abs_tol=tolerance):
                return middle
            if function_middle_value_is_negative == function_left_value_is_negative:
                left = middle
                function_left_value_is_negative = function_middle_value_is_negative
            else:
                right = middle

        result = (left + right) / 2
        self.history.final = (left, right, result, self.function(result))
        return result
