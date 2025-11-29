from solvers.monadic.bisection import BisectionSolver
from solvers.monadic.interval import Interval
from visualizers.monadic.bisection_visualizer import BisectionVisualizer


def main():
    f = lambda x: x ** 3 + 0.2 * x ** 2 + 5 * x - 1
    solver = BisectionSolver(f)
    solver.solve(Interval(-2, 2), 1e-3)
    visualizer = BisectionVisualizer(solver)
    visualizer.animate()


if __name__ == '__main__':
    main()
