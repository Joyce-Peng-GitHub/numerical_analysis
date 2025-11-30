import dataclasses
import typing


@dataclasses.dataclass
class Step:
    iteration: int


@dataclasses.dataclass
class SolutionTrace:
    steps: typing.List[Step] = dataclasses.field(default_factory=list)
    final_result: typing.Any = None
    has_converged: bool = False

    def clear(self):
        self.steps.clear()
        self.final_result = None
        self.has_converged = False

    def print(self):
        print("Solution Trace:")
        for step in self.steps:
            print("\t", step, sep="")
        print(f"Final Result: {self.final_result}")
        print(f"Has Converged: {self.has_converged}")
