import math
import unittest

from solvers.monadic.newton import NewtonSolver


class TestNewtonSolver(unittest.TestCase):
    def test_newton_simple_root(self):
        # f(x) = x, f'(x) = 1, root at x = 0
        solver = NewtonSolver(function=lambda x: x, derivative=lambda x: 1)
        root = solver.solve(initial_guess=0.5, tolerance=1e-8)
        self.assertTrue(math.isclose(root, 0.0, abs_tol=1e-8))

    def test_newton_quadratic_root(self):
        # f(x) = x^2 - 4, f'(x) = 2x, roots at x = -2, 2
        solver = NewtonSolver(function=lambda x: x ** 2 - 4, derivative=lambda x: 2 * x)
        root = solver.solve(initial_guess=1.0, tolerance=1e-8)
        self.assertTrue(math.isclose(root, 2.0, abs_tol=1e-8))

    def test_newton_negative_root(self):
        # f(x) = x^2 - 4, f'(x) = 2x, roots at x = -2, 2
        solver = NewtonSolver(function=lambda x: x ** 2 - 4, derivative=lambda x: 2 * x)
        root = solver.solve(initial_guess=-1.0, tolerance=1e-8)
        self.assertTrue(math.isclose(root, -2.0, abs_tol=1e-8))

    def test_newton_cubic_root(self):
        # f(x) = x^3 - x - 2, f'(x) = 3x^2 - 1, root approximately at x = 1.5214
        solver = NewtonSolver(function=lambda x: x ** 3 - x - 2, derivative=lambda x: 3 * x ** 2 - 1)
        root = solver.solve(initial_guess=1.5, tolerance=1e-10)
        self.assertTrue(math.isclose(root, 1.5213797, rel_tol=0, abs_tol=1e-7))

    def test_newton_non_finite_initial_guess_raises(self):
        solver = NewtonSolver(function=lambda x: x, derivative=lambda x: 1)
        with self.assertRaises(ValueError):
            solver.solve(initial_guess=float("inf"), tolerance=1e-8)

    def test_newton_zero_derivative_raises(self):
        # f(x) = x^3 - 1, f'(x) = 3x^2, derivative is 0 at x = 0
        solver = NewtonSolver(function=lambda x: x ** 3 - 1, derivative=lambda x: 3 * x ** 2)
        with self.assertRaises(ValueError):
            solver.solve(initial_guess=0.0, tolerance=1e-8)

    def test_newton_max_iterations_raises(self):
        # A function that oscillates and doesn't converge quickly
        solver = NewtonSolver(function=lambda x: math.sin(x), derivative=lambda x: math.cos(x))
        with self.assertRaises(ValueError):
            solver.solve(initial_guess=1.4, tolerance=1e-15, max_iterations=2)

    def test_newton_convergence_with_trace(self):
        # f(x) = x^2 - 2, f'(x) = 2x, root at sqrt(2)
        solver = NewtonSolver(function=lambda x: x ** 2 - 2, derivative=lambda x: 2 * x)
        root = solver.solve(initial_guess=1.0, tolerance=1e-10)
        self.assertTrue(math.isclose(root, math.sqrt(2), abs_tol=1e-10))
        self.assertTrue(solver.trace.has_converged)
        self.assertTrue(len(solver.trace.steps) > 0)


if __name__ == '__main__':
    unittest.main()
