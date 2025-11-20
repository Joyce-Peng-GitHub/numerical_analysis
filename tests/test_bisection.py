import math
import unittest

from solvers.monadic.bisection import BisectionSolver
from solvers.monadic.interval import Interval


class TestBisectionSolver(unittest.TestCase):
    def test_bisection_simple_root(self):
        solver = BisectionSolver(func=lambda x: x)
        interval = Interval(-1.0, 1.0)
        root = solver.solve(interval, tolerance=1e-8)
        self.assertTrue(math.isclose(root, 0.0, abs_tol=1e-8))

    def test_bisection_left_endpoint_root(self):
        solver = BisectionSolver(func=lambda x: x + 1)
        interval = Interval(-1.0, 2.0)
        root = solver.solve(interval, tolerance=1e-8)
        self.assertTrue(math.isclose(root, -1.0, abs_tol=0.0))

    def test_bisection_right_endpoint_root(self):
        solver = BisectionSolver(func=lambda x: x - 1)
        interval = Interval(-2.0, 1.0)
        root = solver.solve(interval, tolerance=1e-8)
        self.assertTrue(math.isclose(root, 1.0, abs_tol=0.0))

    def test_bisection_same_sign_raises(self):
        solver = BisectionSolver(func=lambda x: x ** 2 + 1)
        interval = Interval(-1.0, 1.0)
        with self.assertRaises(ValueError):
            solver.solve(interval, tolerance=1e-8)

    def test_bisection_non_finite_interval_raises(self):
        solver = BisectionSolver(func=lambda x: x)
        interval = Interval(float("inf"), 1.0)
        with self.assertRaises(ValueError):
            solver.solve(interval, tolerance=1e-8)

    def test_bisection_convergence_polynomial(self):
        solver = BisectionSolver(func=lambda x: x ** 3 - x - 2)
        interval = Interval(1.0, 2.0)
        tol = 1e-10
        root = solver.solve(interval, tolerance=tol)
        self.assertTrue(math.isclose(root, 1.5213797, rel_tol=0, abs_tol=1e-7))

    def test_bisection_middle_zero_early_return(self):
        solver = BisectionSolver(func=lambda x: x)
        interval = Interval(-1.0, 1.0)
        root = solver.solve(interval, tolerance=1e-8)
        self.assertTrue(math.isclose(root, 0.0, abs_tol=1e-8))


if __name__ == '__main__':
    unittest.main()
