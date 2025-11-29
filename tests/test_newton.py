import unittest
import math
from solvers.monadic.newton import NewtonSolver


class TestNewtonSolver(unittest.TestCase):
    def test_newton_converges_to_positive_sqrt2(self):
        solver = NewtonSolver(lambda x: x * x - 2)
        root = solver.solve(guess=1.0, tolerance=1e-12, max_iterations=50)
        expected = math.sqrt(2)
        self.assertTrue(math.isclose(root, expected, rel_tol=0, abs_tol=1e-10))
        self.assertTrue(solver.trace.has_converged)
        self.assertTrue(math.isclose(solver.trace.final_result, root, abs_tol=1e-12))
        self.assertGreater(len(solver.trace.steps), 0)

    def test_newton_converges_to_negative_sqrt2(self):
        solver = NewtonSolver(lambda x: x * x - 2)
        root = solver.solve(guess=-1.0, tolerance=1e-12, max_iterations=50)
        expected = -math.sqrt(2)
        self.assertTrue(math.isclose(root, expected, rel_tol=0, abs_tol=1e-10))
        self.assertTrue(solver.trace.has_converged)

    def test_newton_converges_to_cuberoot_of_8(self):
        solver = NewtonSolver(lambda x: x**3 - 8)
        root = solver.solve(guess=3.0, tolerance=1e-12, max_iterations=50)
        expected = 2.0
        self.assertTrue(math.isclose(root, expected, rel_tol=0, abs_tol=1e-10))
        self.assertTrue(solver.trace.has_converged)
        self.assertTrue(math.isclose(solver.trace.final_result, root, abs_tol=1e-12))

    def test_newton_non_convergence_with_small_max_iterations(self):
        solver = NewtonSolver(lambda x: x * x - 2)
        root = solver.solve(guess=10.0, tolerance=1e-12, max_iterations=1)
        # With only one iteration we should not have converged for this guess
        self.assertFalse(solver.trace.has_converged)
        self.assertEqual(solver.trace.final_result, root)
        self.assertEqual(len(solver.trace.steps), 1)

    def test_newton_step_contents(self):
        solver = NewtonSolver(lambda x: x * x - 2)
        root = solver.solve(guess=1.0, tolerance=1e-12, max_iterations=1)
        first = solver.trace.steps[0]
        # initial guess and function value should match expected
        self.assertEqual(first.guess, 1.0)
        self.assertTrue(math.isclose(first.function_value, -1.0, abs_tol=1e-12))
        # derivative should be close to 2 for f(x)=x^2-2 at x=1
        self.assertTrue(math.isclose(first.derivative_value, 2.0, rel_tol=1e-3))

if __name__ == '__main__':
    unittest.main()
