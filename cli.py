import math
import numpy as np

from solvers.monadic import *
from visualizers.monadic.monadic_equation_visualizer import MonadicEquationVisualizer


class Cli:
    def __init__(self):
        pass

    def run_monadic(self):
        print("-" * 80)
        print("Select a solver (Ctrl + C to exit):")
        print("- BisectionSolver")
        print("- NewtonSolver")
        print("- NewtonDownhillSolver")
        solver = eval(input())()
        print("-" * 60)
        solver.function = eval("lambda x: " +
                               input("Input the function (math and np imported): "
                                     "f(x) = lambda x: "))
        print("-" * 60)
        parameters = input("Input the parameters: ")
        exec("solver.solve(" + parameters + ")")
        print("-" * 60)
        print("Solution trace:")
        for step in solver.trace.steps:
            print("\t", step, sep="")
        print("Final result:", solver.trace.final_result)
        print("Has converged:", solver.trace.has_converged)
        print("-" * 80)
        visualizer = MonadicEquationVisualizer(solver)
        parameters = input("Parameters for visualizer (optional):")
        print("-" * 60)
        print("Starting animation...")
        exec("visualizer.animate(" + parameters + ")")
        print("Animation complete.")
