import math


class Interval:
    def __init__(self, left: float, right: float, include_left: bool = True, include_right: bool = True):
        self.left, self.right, self.include_left, self.include_right = left, right, include_left, include_right

    def is_finite(self) -> bool:
        return math.isfinite(self.left) and math.isfinite(self.right)

    def contains(self, value: float) -> bool:
        return (self.left < value < self.right or
                (self.include_left and value == self.left) or
                (self.include_right and value == self.right))

    def __contains__(self, value: float) -> bool:
        return self.contains(value)
