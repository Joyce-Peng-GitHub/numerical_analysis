import unittest
import math

from solvers.monadic.newton_downhill import NewtonDownhillSolver
from solvers.monadic.monadic_equation_solver import MonadicEquationSolverNotConvergedException


class TestNewtonDownhillSolver(unittest.TestCase):
    def test_linear_converges_in_one_step(self):
        solver = NewtonDownhillSolver(lambda x: 2.0 * x)
        root = solver.solve(guess=1.0, tolerance=1e-10, max_iterations=10)
        self.assertTrue(math.isclose(root, 0.0, abs_tol=1e-10))
        self.assertTrue(solver.trace.has_converged)
        self.assertEqual(len(solver.trace.steps), 1)

    def test_already_at_root_immediate_return(self):
        solver = NewtonDownhillSolver(lambda x: x)
        root = solver.solve(guess=0.0, tolerance=1e-12, max_iterations=5)
        self.assertTrue(math.isclose(root, 0.0, abs_tol=1e-12))
        self.assertTrue(solver.trace.has_converged)
        self.assertEqual(len(solver.trace.steps), 0)

    def test_negative_initial_guess_linear(self):
        solver = NewtonDownhillSolver(lambda x: 2.0 * x)
        root = solver.solve(guess=-5.0, tolerance=1e-12, max_iterations=10)
        self.assertTrue(math.isclose(root, 0.0, abs_tol=1e-12))
        self.assertTrue(solver.trace.has_converged)

    def test_quadratic_positive_root(self):
        solver = NewtonDownhillSolver(lambda x: x * x - 2.0)
        root = solver.solve(guess=1.5, tolerance=1e-12, max_iterations=50)
        expected = math.sqrt(2.0)
        self.assertTrue(math.isclose(root, expected, rel_tol=0, abs_tol=1e-10))
        self.assertTrue(solver.trace.has_converged)

    def test_quadratic_negative_root(self):
        solver = NewtonDownhillSolver(lambda x: x * x - 2.0)
        root = solver.solve(guess=-1.5, tolerance=1e-12, max_iterations=50)
        expected = -math.sqrt(2.0)
        self.assertTrue(math.isclose(root, expected, rel_tol=0, abs_tol=1e-10))
        self.assertTrue(solver.trace.has_converged)

    def test_cubic_root(self):
        solver = NewtonDownhillSolver(lambda x: x**3 - 8.0)
        root = solver.solve(guess=3.0, tolerance=1e-12, max_iterations=50)
        expected = 2.0
        self.assertTrue(math.isclose(root, expected, rel_tol=0, abs_tol=1e-10))
        self.assertTrue(solver.trace.has_converged)

    def test_no_convergence_raises_when_requested(self):
        solver = NewtonDownhillSolver(lambda x: x * x - 2.0)
        with self.assertRaises(MonadicEquationSolverNotConvergedException):
            # only one iteration allowed: should not converge for this guess and should raise
            solver.solve(guess=10.0, tolerance=1e-12, max_iterations=1, raise_exception_if_no_convergence=True)

    def test_trace_iteration_indices_and_within_limit(self):
        solver = NewtonDownhillSolver(lambda x: x * x - 2.0)
        max_iter = 5
        solver.solve(guess=3.0, tolerance=1e-12, max_iterations=max_iter)
        steps = solver.trace.steps
        self.assertLessEqual(len(steps), max_iter)
        for idx, step in enumerate(steps):
            self.assertEqual(step.iteration, idx)

    def test_final_result_matches_returned_value(self):
        solver = NewtonDownhillSolver(lambda x: x**3 - 27.0)
        returned = solver.solve(guess=4.0, tolerance=1e-12, max_iterations=50)
        self.assertEqual(returned, solver.trace.final_result)

    def test_step_fields_and_damping_power_of_two(self):
        # Use a function where full Newton step is acceptable (damping should be 1),
        # check step fields and that damping is a positive power of two.
        solver = NewtonDownhillSolver(lambda x: x * x - 2.0)
        solver.solve(guess=1.5, tolerance=1e-12, max_iterations=10)
        steps = solver.trace.steps
        self.assertGreaterEqual(len(steps), 1)
        first = steps[0]
        self.assertTrue(hasattr(first, 'iteration'))
        self.assertTrue(hasattr(first, 'x'))
        self.assertTrue(hasattr(first, 'x_function_value'))
        self.assertTrue(hasattr(first, 'x_derivative_value'))
        self.assertTrue(hasattr(first, 'damping_factor'))
        df = first.damping_factor
        self.assertIsInstance(df, int)
        # check df is positive and power of two
        self.assertGreaterEqual(df, 1)
        self.assertEqual(df & (df - 1), 0)

    def test_converges_for_transcendental_function_cos_minus_x(self):
        # Newton on f(x) = cos(x) - x converges to a root near 0.739... for a good initial guess
        target = 0.7390851332151607
        solver = NewtonDownhillSolver(lambda x: math.cos(x) - x)
        root = solver.solve(guess=0.5, tolerance=1e-12, max_iterations=200)
        self.assertTrue(math.isclose(root, target, rel_tol=1e-12, abs_tol=1e-9))
        self.assertTrue(solver.trace.has_converged)


if __name__ == '__main__':
    unittest.main()
