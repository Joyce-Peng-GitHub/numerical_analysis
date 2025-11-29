import unittest
import math

from solvers.monadic.aitken import AitkenSolver
from solvers.monadic.monadic_equation_solver import MonadicEquationSolverNotConvergedException


class TestAitkenSolver(unittest.TestCase):
    def test_aitken_converges_for_linear_contraction(self):
        # f(x) = -0.5*x -> self.function = x + f(x) = 0.5*x (contraction), fixed point at 0
        solver = AitkenSolver(lambda x: -0.5 * x)
        root = solver.solve(guess=4.0, tolerance=1e-12, max_iterations=10)
        self.assertTrue(math.isclose(root, 0.0, abs_tol=1e-12))
        self.assertTrue(solver.trace.has_converged)
        self.assertEqual(solver.trace.final_result, root)
        self.assertGreater(len(solver.trace.steps), 0)

    def test_aitken_step_contents(self):
        solver = AitkenSolver(lambda x: -0.5 * x)
        solver.solve(guess=4.0, tolerance=1e-12, max_iterations=1)
        first = solver.trace.steps[0]
        # For guess=4.0 and g(x)=0.5*x we expect y=2.0, z=1.0 and slope=0.5
        self.assertEqual(first.x, 4.0)
        self.assertEqual(first.y, 2.0)
        self.assertEqual(first.z, 1.0)
        self.assertAlmostEqual(first.slope, 0.5, places=12)

    def test_aitken_raises_if_no_convergence_requested(self):
        solver = AitkenSolver(lambda x: -0.5 * x)
        with self.assertRaises(MonadicEquationSolverNotConvergedException):
            # With only one iteration we do not reach the convergence-detection step -> should raise
            solver.solve(guess=4.0, tolerance=1e-12, max_iterations=1, raise_exception_if_no_convergence=True)

    def test_aitken_converges_to_nonzero_fixed_point(self):
        # g(x) = 0.5*x + 1  -> f(x) = g(x) - x = -0.5*x + 1 ; fixed point at 2
        solver = AitkenSolver(lambda x: -0.5 * x + 1)
        root = solver.solve(guess=0.0, tolerance=1e-12, max_iterations=50)
        expected = 2.0
        self.assertTrue(math.isclose(root, expected, abs_tol=1e-10))
        self.assertTrue(solver.trace.has_converged)
        self.assertAlmostEqual(solver.trace.final_result, expected, places=12)

    def test_aitken_converges_from_negative_guess(self):
        solver = AitkenSolver(lambda x: -0.5 * x + 1)
        root = solver.solve(guess=-10.0, tolerance=1e-12, max_iterations=50)
        expected = 2.0
        self.assertTrue(math.isclose(root, expected, abs_tol=1e-10))
        self.assertTrue(solver.trace.has_converged)

    # new tests
    def test_aitken_step_iteration_numbers(self):
        solver = AitkenSolver(lambda x: -0.5 * x)
        max_iter = 2
        solver.solve(guess=4.0, tolerance=1e-12, max_iterations=max_iter)
        steps = solver.trace.steps
        # The solver may converge faster than max_iterations; ensure we did not exceed the limit
        self.assertLessEqual(len(steps), max_iter)
        # If steps were recorded, ensure their iteration indices are sequential starting at 0
        for idx, step in enumerate(steps):
            self.assertEqual(step.iteration, idx)

    def test_aitken_slope_for_nonzero_fixed_point(self):
        # g(x) = 0.5*x + 1  -> f(x) = -0.5*x + 1 ; fixed point at 2
        solver = AitkenSolver(lambda x: -0.5 * x + 1)
        solver.solve(guess=0.0, tolerance=1e-12, max_iterations=1)
        first = solver.trace.steps[0]
        # For guess=0.0 we expect x=0.0, y=1.0, z=1.5 and slope=0.5
        self.assertEqual(first.x, 0.0)
        self.assertEqual(first.y, 1.0)
        self.assertEqual(first.z, 1.5)
        self.assertAlmostEqual(first.slope, 0.5, places=12)


if __name__ == '__main__':
    unittest.main()
