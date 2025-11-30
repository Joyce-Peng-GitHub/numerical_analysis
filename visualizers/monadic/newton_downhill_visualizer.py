import copy as cp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from typing import Tuple

# Adjust imports based on your actual project structure
from solvers.monadic.newton_downhill import NewtonDownhillSolver


class NewtonDownhillVisualizer:
    DEFAULT_SAMPLE_NUM = 256
    DEFAULT_FIGURE_SIZE = (16, 9)
    DEFAULT_TICK_SIZE = 12
    DEFAULT_LABEL_SIZE = 16
    DEFAULT_TITLE_SIZE = 20
    DEFAULT_INTERVAL_MS = 1000

    def __init__(self, solver: NewtonDownhillSolver):
        self.solver = cp.deepcopy(solver)

    def animate(
            self,
            sample_num: int = DEFAULT_SAMPLE_NUM,
            figure_size: Tuple[int, int] = DEFAULT_FIGURE_SIZE,
            tick_size: int = DEFAULT_TICK_SIZE,
            label_size: int = DEFAULT_LABEL_SIZE,
            title_size: int = DEFAULT_TITLE_SIZE,
            interval_ms: int = DEFAULT_INTERVAL_MS
    ):
        if len(self.solver.trace.steps) == 0:
            raise ValueError("Cannot visualize an empty trace.")

        figure, (axes_global, axes_zoom) = plt.subplots(1, 2, figsize=figure_size)
        figure.suptitle(f"Newton Downhill Method: {len(self.solver.trace.steps)} Iterations", fontsize=title_size)

        # --- 1. GLOBAL VIEW SETUP ---
        # We need to scan all steps to find the global bounds.
        # Unlike standard Newton, we have 'x' and the calculated 'next' which depends on damping.
        all_x_points = []
        for step in self.solver.trace.steps:
            all_x_points.append(step.x)
            # Calculate where this step went
            if step.x_derivative_value != 0:
                # next = x - f(x) / (damping * f'(x))
                next_val = step.x - (step.x_function_value / (step.damping_factor * step.x_derivative_value))
                all_x_points.append(next_val)

        min_x, max_x = min(all_x_points), max(all_x_points)
        span = max_x - min_x
        if span == 0: span = 1.0

        pad = span * 0.2
        global_x_min, global_x_max = min_x - pad, max_x + pad

        # Generate Global Curve
        x_global = np.linspace(global_x_min, global_x_max, sample_num)
        y_global = np.array([self.solver.function(x) for x in x_global])

        axes_global.plot(x_global, y_global, "k-", alpha=0.3, label=r"f(x)")
        axes_global.axhline(0, color="black", linewidth=1)
        axes_global.set_title("Global Context", fontsize=title_size)
        axes_global.tick_params(labelsize=tick_size)
        axes_global.set_xlabel(r"$x$", loc="center", fontsize=label_size)
        axes_global.set_xlim(global_x_min, global_x_max)

        # Handle Y-limits (clip extreme values for readability)
        y_vis_min, y_vis_max = y_global.min(), y_global.max()
        axes_global.set_ylim(y_vis_min, y_vis_max)

        # Global Markers
        global_current_point, = axes_global.plot([], [], "go", markersize=4, label="Current")
        global_next_point, = axes_global.plot([], [], "rx", markersize=4, label="Next (Damped)")

        # The "Camera Box"
        zoom_rectangle = Rectangle((0, 0), 1, 1, fill=False, edgecolor="purple", linestyle="--", linewidth=2,
                                   label="Camera View")
        axes_global.add_patch(zoom_rectangle)
        axes_global.legend(fontsize=label_size)

        # --- 2. ZOOM VIEW SETUP ---
        zoom_curve, = axes_zoom.plot([], [], "b-", linewidth=2, alpha=0.6)
        axes_zoom.axhline(0, color="black", linewidth=1)
        axes_zoom.set_title("Tangent & Damping View", fontsize=title_size)
        axes_zoom.tick_params(labelsize=tick_size)
        axes_zoom.set_xlabel(r"$x$", loc="center", fontsize=label_size)

        # Visual Elements:
        vline_current = axes_zoom.axvline(0, color="green", linestyle=":", alpha=0.5)

        # The Tangent Line (Represents the full, undamped Newton vector)
        tangent_line, = axes_zoom.plot([], [], "r--", linewidth=1.5, alpha=0.4, label="Tangent (Undamped)")

        # Points
        point_current, = axes_zoom.plot([], [], "go", markersize=8, zorder=5, label=r"$(x_n, f(x_n))$")

        # Ghost Point: Where standard Newton WOULD have gone (damping = 1)
        point_ghost, = axes_zoom.plot([], [], "rx", markersize=6, alpha=0.4, label=r"Standard Newton")

        # Actual Point: Where we actually went (damping >= 1)
        point_next, = axes_zoom.plot([], [], "rx", markersize=8, markeredgewidth=2, zorder=5,
                                     label=r"Actual ($x_{n+1}$)")

        info_text = axes_zoom.text(0.02, 0.95, "", transform=axes_zoom.transAxes,
                                   verticalalignment="top",
                                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
                                   fontsize=label_size)
        axes_zoom.legend(fontsize=10, loc='lower right')

        def update(frame):
            step = self.solver.trace.steps[frame]

            # 1. Calculate Standard Newton Step (Undamped target)
            if step.x_derivative_value == 0:
                undamped_next = step.x
            else:
                undamped_next = step.x - (step.x_function_value / step.x_derivative_value)

            # 2. Calculate Actual Step (Damped target)
            if step.x_derivative_value == 0:
                actual_next = step.x
            else:
                # Formula: x - f(x) / (damping * f'(x))
                actual_next = step.x - (step.x_function_value / (step.damping_factor * step.x_derivative_value))

            # --- STEP A: CALCULATE ZOOM GEOMETRY ---
            # Focus bounds must include: Current X, Undamped Target, and Actual Target
            points_of_interest = [step.x, undamped_next, actual_next]
            focus_x_min = min(points_of_interest)
            focus_x_max = max(points_of_interest)

            dist = focus_x_max - focus_x_min
            if dist == 0: dist = 0.1

            view_pad_x = dist * 0.5
            zoom_x_min = focus_x_min - view_pad_x
            zoom_x_max = focus_x_max + view_pad_x

            # Generate Y-Data for Zoom
            x_zoom = np.linspace(zoom_x_min, zoom_x_max, 200)
            y_zoom = np.array([self.solver.function(x) for x in x_zoom])

            # Y-Limits: Must include 0 (axis) and f(x_current)
            focus_y_vals = [0, step.x_function_value]
            y_min_raw, y_max_raw = min(focus_y_vals), max(focus_y_vals)
            y_range = y_max_raw - y_min_raw
            if y_range == 0: y_range = 1.0

            view_pad_y = y_range * 0.4
            zoom_y_min = y_min_raw - view_pad_y
            zoom_y_max = y_max_raw + view_pad_y

            # --- STEP B: UPDATE GLOBAL VIEW ---
            global_current_point.set_data([step.x], [step.x_function_value])
            global_next_point.set_data([actual_next], [0])

            rect_width = zoom_x_max - zoom_x_min
            rect_height = zoom_y_max - zoom_y_min
            zoom_rectangle.set_bounds(zoom_x_min, zoom_y_min, rect_width, rect_height)

            # --- STEP C: UPDATE ZOOM VIEW ---
            axes_zoom.set_xlim(zoom_x_min, zoom_x_max)
            axes_zoom.set_ylim(zoom_y_min, zoom_y_max)

            zoom_curve.set_data(x_zoom, y_zoom)

            # Vertical Line (Current Guess)
            vline_current.set_xdata([step.x, step.x])

            # Tangent Line: ALWAYS draw the full tangent to the Undamped intercept
            # This visually explains "Where we wanted to go" vs "Where we went"
            tangent_line.set_data([step.x, undamped_next], [step.x_function_value, 0])

            # Points
            point_current.set_data([step.x], [step.x_function_value])
            point_ghost.set_data([undamped_next], [0])  # The faint "if we didn't damp" point
            point_next.set_data([actual_next], [0])  # The actual "damped" point

            info_text.set_text(
                rf"$n={step.iteration}$""\n"
                rf"$\lambda=1/{step.damping_factor}$ (Damping)""\n"
                rf"$x^{{(n)}}={step.x:.4g}$""\n"
                rf"$f(x^{{(n)}})={step.x_function_value:.4g}$""\n"
                rf"$x^{{(n+1)}}={actual_next:.4g}$"
            )

            return global_current_point, global_next_point, zoom_rectangle, zoom_curve, tangent_line, point_current, point_ghost, point_next, vline_current

        animation = FuncAnimation(
            figure,
            update,
            frames=len(self.solver.trace.steps),
            interval=interval_ms, blit=False, repeat=False
        )

        plt.tight_layout()
        plt.show()
