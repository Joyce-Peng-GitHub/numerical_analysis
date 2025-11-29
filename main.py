import math

from solvers.monadic.aitken import AitkenSolver
from solvers.monadic.bisection import BisectionSolver
from solvers.monadic.interval import Interval
from solvers.monadic.newton import NewtonSolver
from visualizers.monadic.aitken_visualizer import AitkenVisualizer
from visualizers.monadic.bisection_visualizer import BisectionVisualizer
from visualizers.monadic.newton_visualizer import NewtonVisualizer


def main():
    f = lambda x: 4 * math.sin(0.5 * x + 2) + 0.1 * x ** 2 - 0.4 * x - 5
    solver = AitkenSolver(f)
    root = solver.solve(guess=10.0, tolerance=1e-3, raise_exception_if_no_convergence=False, max_iterations=50)
    for step in solver.trace.steps:
        print(step)
    print(solver.trace.final_result)
    visualizer = AitkenVisualizer(solver)
    visualizer.animate()


if __name__ == '__main__':
    main()
