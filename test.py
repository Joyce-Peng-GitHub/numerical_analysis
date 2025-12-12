import math

import numpy as np

from solvers.linear_system.gauss import GaussSolver
from solvers.monadic.aitken import AitkenSolver
from solvers.monadic.newton_downhill import NewtonDownhillSolver
from visualizers.monadic.monadic_equation_visualizer import MonadicEquationVisualizer


def main():
    coefficients = np.array([[10, -19, -2], [-20, 40, 1], [1, 4, 5]])
    biases = np.array([[3, 4, 5], [1, 2, 3]]).T
    solver = GaussSolver()
    solver.solve(coefficients, biases)
    solver.trace.print()


if __name__ == "__main__":
    main()
