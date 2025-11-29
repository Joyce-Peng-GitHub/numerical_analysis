from solvers.monadic.monadic_equation_solver import UnaryFunction

DEFAULT_STEP_SIZE = 1e-5  # determined by features of double precision floating point numbers


def get_derivative_of(function: UnaryFunction, step_size: float = DEFAULT_STEP_SIZE) -> UnaryFunction:
    return lambda x: (function(x + step_size) - function(x- step_size)) / (2 * step_size)
