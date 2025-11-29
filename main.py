import math

from solvers.monadic.bisection import BisectionSolver
from solvers.monadic.interval import Interval
from solvers.monadic.newton import NewtonSolver
from visualizers.monadic.bisection_visualizer import BisectionVisualizer
from visualizers.monadic.newton_visualizer import NewtonVisualizer


def main():
    f = lambda x: x - 2
    solver = NewtonSolver(f)
    root = solver.solve(guess=2.0, tolerance=1e-3, max_iterations=50)
    print(root)
    visualizer = NewtonVisualizer(solver)
    visualizer.animate()


if __name__ == '__main__':
    main()
