import math
from solvers.monadic.aitken import AitkenSolver
from solvers.monadic.bisection import BisectionSolver
from solvers.monadic.interval import Interval
from solvers.monadic.newton import NewtonSolver
from visualizers.monadic.monadic_equation_visualizer import MonadicEquationVisualizer


def main():
    f = lambda x: 2 * math.sin(x + 4) + 0.05 * x ** 2 - x
    solver = NewtonSolver(f)
    solver.solve(guess=2, tolerance=1e-3)
    solver.trace.print()
    visualizer = MonadicEquationVisualizer(solver)
    visualizer.animate(interval_ms=1000)


if __name__ == "__main__":
    main()
