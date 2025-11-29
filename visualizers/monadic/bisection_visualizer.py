import copy as cp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from typing import Tuple

# Adjust imports based on your actual project structure
from solvers.monadic.bisection import BisectionSolver


class BisectionVisualizer:
    DEFAULT_SAMPLE_NUM = 256
    DEFAULT_FIGURE_SIZE = (16, 9)
    DEFAULT_INTERVAL_MS = 1000

    def __init__(self, solver: BisectionSolver):
        self.solver = cp.deepcopy(solver)

    def animate(
            self,
            sample_num: int = DEFAULT_SAMPLE_NUM,
            figure_size: Tuple[int, int] = DEFAULT_FIGURE_SIZE,
            interval_ms: int = DEFAULT_INTERVAL_MS
    ):
        figure, (axes_global, axes_zoom) = plt.subplots(1, 2, figsize=figure_size)
        figure.suptitle(f"Bisection Method: {len(self.solver.trace.steps)} Iterations", fontsize=14)

        # --- 1. GLOBAL VIEW SETUP ---
        if len(self.solver.trace.steps) == 0:
            raise ValueError("Cannot visualize an empty trace.")
        init_step = self.solver.trace.steps[0]

        # Determine global bounds based on the initial interval + 20% padding
        init_width = init_step.right - init_step.left
        pad = init_width * 0.2
        global_x_min, global_x_max = init_step.left - pad, init_step.right + pad

        # Generate a high-res curve for global view
        x_global = np.linspace(global_x_min, global_x_max, sample_num)
        y_global = np.array([self.solver.function(x) for x in x_global])

        axes_global.plot(x_global, y_global, 'k-', alpha=0.3, label='f(x)')
        axes_global.axhline(0, color='black', linewidth=1)
        axes_global.set_title("Global Context (Fixed View)")
        axes_global.set_xlim(global_x_min, global_x_max)
        axes_global.set_ylim(y_global.min(), y_global.max())

        # Global markers
        global_line_left = axes_global.axvline(init_step.left, color='green', alpha=0.5)
        global_line_right = axes_global.axvline(init_step.right, color='red', alpha=0.5)
        global_mid_point, = axes_global.plot([], [], 'bo', markersize=4)

        # The "Camera Box" (Purple Rectangle)
        # Represents exactly what is seen in the right-hand plot
        zoom_rectangle = Rectangle((0, 0), 1, 1, fill=False, edgecolor='purple', linestyle='--', linewidth=2,
                                   label='Camera View')
        axes_global.add_patch(zoom_rectangle)

        # --- 2. ZOOM VIEW SETUP ---
        zoom_curve, = axes_zoom.plot([], [], 'b-', linewidth=2)
        axes_zoom.axhline(0, color='black', linewidth=1)
        axes_zoom.set_title("Microscope View (Adaptive Zoom)")

        zoom_line_left = axes_zoom.axvline(0, color='green', linestyle='--', label='a (Left)')
        zoom_line_right = axes_zoom.axvline(0, color='red', linestyle='--', label='b (Right)')
        zoom_mid_point, = axes_zoom.plot([], [], 'bo', markersize=8, zorder=5, label='Midpoint')

        info_text = axes_zoom.text(0.02, 0.95, '', transform=axes_zoom.transAxes,
                                   verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        def update(frame):
            step = self.solver.trace.steps[frame]
            interval_width = step.right - step.left

            # --- STEP A: CALCULATE ZOOM GEOMETRY FIRST ---
            # We must know the "Zoom View" limits before we can draw the rectangle

            # X-Limits: Interval +/- 50% padding
            # This creates a "View Frame" centered on the interval
            view_pad_x = interval_width * 0.5
            zoom_x_min = step.left - view_pad_x
            zoom_x_max = step.right + view_pad_x

            # Generate Y-Data for this specific X-Range
            x_zoom = np.linspace(zoom_x_min, zoom_x_max, 200)
            y_zoom = np.array([self.solver.function(x) for x in x_zoom])

            # Y-Limits: Min/Max of the curve segment +/- 20% padding
            y_min_raw, y_max_raw = y_zoom.min(), y_zoom.max()

            # Safety for flat lines
            if y_max_raw == y_min_raw:
                y_max_raw += 1.0
                y_min_raw -= 1.0

            y_range = y_max_raw - y_min_raw
            view_pad_y = y_range * 0.2

            zoom_y_min = y_min_raw - view_pad_y
            zoom_y_max = y_max_raw + view_pad_y

            # --- STEP B: UPDATE GLOBAL VIEW (The Rectangle) ---
            global_line_left.set_xdata([step.left, step.left])
            global_line_right.set_xdata([step.right, step.right])
            global_mid_point.set_data([step.middle], [step.middle_function_value])

            # ERROR FIX: Set bounds using the calculated ZOOM limits (both X and Y)
            rect_width = zoom_x_max - zoom_x_min
            rect_height = zoom_y_max - zoom_y_min
            zoom_rectangle.set_bounds(zoom_x_min, zoom_y_min, rect_width, rect_height)

            # --- STEP C: UPDATE ZOOM VIEW ---
            # Now we just apply the limits we calculated in Step A
            axes_zoom.set_xlim(zoom_x_min, zoom_x_max)
            axes_zoom.set_ylim(zoom_y_min, zoom_y_max)

            zoom_curve.set_data(x_zoom, y_zoom)
            zoom_line_left.set_xdata([step.left, step.left])
            zoom_line_right.set_xdata([step.right, step.right])
            zoom_mid_point.set_data([step.middle], [step.middle_function_value])

            info_text.set_text(
                f"Iter: {step.iteration}\n"
                f"Interval: [{step.left:.6f}, {step.right:.6f}]\n"
                f"Width: {interval_width:.2e}\n"
                f"f(mid): {step.middle_function_value:.2e}"
            )

            return global_line_left, global_line_right, zoom_rectangle, zoom_curve, zoom_line_left, zoom_line_right, zoom_mid_point

        animation = FuncAnimation(figure, update, frames=len(self.solver.trace.steps),
                                  interval=interval_ms, blit=False, repeat=False)

        plt.tight_layout()
        plt.show()
