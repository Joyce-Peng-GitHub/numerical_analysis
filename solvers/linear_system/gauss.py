import numpy as np
import dataclasses

from solvers.solution_trace import Step, SolutionTrace


@dataclasses.dataclass
class GaussStep(Step):
    matrix_snapshot: np.ndarray
    description: str = ""

    def __repr__(self):
        return f"Step {self.iteration}: {self.description}\n{np.array2string(self.matrix_snapshot, precision=2, suppress_small=True)}\n"


class GaussSolver:
    def __init__(self):
        self.trace = SolutionTrace()

    def solve(self, coefficients: np.ndarray, bias: np.ndarray) -> np.ndarray:
        self.trace.clear()

        if coefficients.ndim != 2:
            raise ValueError("coefficients must be a 2D matrix.")

        row_number, column_number = coefficients.shape

        if bias.ndim == 1:
            bias = bias.reshape(-1, 1)
        elif bias.ndim != 2:
            raise ValueError("bias must be a 1D vector or a 2D matrix.")

        if bias.shape[0] != row_number:
            raise ValueError("Number of rows of coefficients and bias must match.")

        augmented_matrix = np.hstack((coefficients, bias)).astype(float)

        step_count = 0
        self.trace.steps.append(GaussStep(
            iteration=step_count,
            matrix_snapshot=np.copy(augmented_matrix),
            description="Initial Augmented Matrix"
        ))

        for i in range(column_number):
            pivot_row = i + np.argmax(np.abs(augmented_matrix[i:, i]))

            if np.isclose(augmented_matrix[pivot_row, i], 0):
                raise ValueError("Matrix is singular (det=0), cannot solve.")

            if pivot_row != i:
                augmented_matrix[[i, pivot_row]] = augmented_matrix[[pivot_row, i]]
                step_count += 1
                self.trace.steps.append(GaussStep(
                    iteration=step_count,
                    matrix_snapshot=np.copy(augmented_matrix),
                    description=f"Pivoting: Swapped Row {i} and Row {pivot_row}"
                ))

            current_pivot_val = augmented_matrix[i, i]
            changed = False

            for j in range(i + 1, column_number):
                if not np.isclose(augmented_matrix[j, i], 0):
                    factor = augmented_matrix[j, i] / current_pivot_val
                    augmented_matrix[j] = augmented_matrix[j] - factor * augmented_matrix[i]
                    changed = True

            if changed:
                step_count += 1
                self.trace.steps.append(GaussStep(
                    iteration=step_count,
                    matrix_snapshot=np.copy(augmented_matrix),
                    description=f"Elimination: Cleared column {i} below pivot"
                ))

        # Back Substitution: compute result of shape (column_number, rhs_count)
        rhs_count = bias.shape[1]
        result = np.zeros((column_number, rhs_count), dtype=float)
        for i in range(column_number - 1, -1, -1):
            if i + 1 < column_number:
                sum_ax = augmented_matrix[i, i + 1:column_number].dot(
                    result[i + 1:column_number, :])  # shape (rhs_count,)
            else:
                sum_ax = np.zeros(rhs_count, dtype=float)

            rhs_vals = augmented_matrix[i, column_number:]  # shape (rhs_count,)
            result[i, :] = (rhs_vals - sum_ax) / augmented_matrix[i, i]

        self.trace.final_result = result
        self.trace.has_converged = True
        return result
