import copy as cp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from typing import Tuple

# Adjust imports based on your actual project structure
from solvers.monadic.aitken import AitkenSolver


class AitkenVisualizer:
    DEFAULT_SAMPLE_NUM = 256
    DEFAULT_FIGURE_SIZE = (16, 9)
    DEFAULT_TICK_SIZE = 12
    DEFAULT_LABEL_SIZE = 16
    DEFAULT_TITLE_SIZE = 20
    DEFAULT_INTERVAL_MS = 1500  # Slower default to see the cobweb path

    def __init__(self, solver: AitkenSolver):
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
        figure.suptitle(f"Aitken's Method: {len(self.solver.trace.steps)} Iterations", fontsize=title_size)

        # --- 1. PRE-CALCULATE BOUNDS ---
        # We need to collect all x (input), y (1st iter), z (2nd iter)
        # AND the accelerated guesses to set global bounds.
        all_points = []
        for step in self.solver.trace.steps:
            all_points.extend([step.x, step.y, step.z])
            # Calculate the accelerated guess manually for bounds checking
            # Formula: x - (y-x)^2 / (z - 2y + x) => (xz - y^2)/(x - 2y + z)
            denominator = step.x - 2 * step.y + step.z
            if denominator != 0:
                acc_guess = (step.x * step.z - step.y ** 2) / denominator
                all_points.append(acc_guess)

        min_val, max_val = min(all_points), max(all_points)
        span = max_val - min_val
        if span == 0: span = 1.0
        pad = span * 0.2

        global_min = min_val - pad
        global_max = max_val + pad

        # --- 2. GLOBAL VIEW SETUP ---
        # Generate curve for g(x)
        x_global = np.linspace(global_min, global_max, sample_num)
        y_global = np.array([self.solver.function(x) for x in x_global])

        # Plot g(x)
        axes_global.plot(x_global, y_global, "-", color="blue", label=r"$y=\varphi(x)$")
        # Plot y = x (The Fixed Point Diagonal)
        axes_global.plot(x_global, x_global, "--", color="black", label=r"$y=x$")

        axes_global.set_title("Global Context (Fixed View)", fontsize=title_size)
        axes_global.tick_params(labelsize=tick_size)
        axes_global.set_xlabel(r"$x$", loc="center", fontsize=label_size)
        axes_global.set_xlim(global_min, global_max)
        axes_global.set_ylim(global_min, global_max)  # Square aspect often helps fixed-point viz
        axes_global.grid(True, linestyle="--", alpha=0.5)

        # Global Markers
        global_current_point, = axes_global.plot([], [], "go", markersize=4, label="Start ($x^{{(n)}}$)")
        global_acc_point, = axes_global.plot([], [], "rx", markersize=5, label="Accelerated ($x^{{(n+1)}}$)")

        # The "Camera Box"
        zoom_rectangle = Rectangle((0, 0), 1, 1, fill=False, edgecolor="purple", linestyle="--", linewidth=2,
                                   label="Camera View")
        axes_global.add_patch(zoom_rectangle)
        axes_global.legend(fontsize=label_size)

        # --- 3. ZOOM VIEW SETUP ---
        zoom_curve, = axes_zoom.plot([], [], linestyle="-", color="blue", linewidth=2, label=r"$y=\varphi(x)$")
        zoom_diagonal, = axes_zoom.plot([], [], linestyle="--", color="black", linewidth=1.5, label=r"$y=x$")

        axes_zoom.set_title("Cobweb & Acceleration View", fontsize=title_size)
        axes_zoom.tick_params(labelsize=tick_size)
        axes_zoom.set_xlabel(r"$x$", loc="center", fontsize=label_size)

        # Visual Elements:
        # 1. The Cobweb path: (x,x)->(x,y)->(y,y)->(y,z)->(z,z)
        cobweb_line, = axes_zoom.plot([], [], "g:", linewidth=1.5, alpha=0.8, label="Standard Iterations")
        # 2. Points on the path
        points_scatter, = axes_zoom.plot([], [], "go", markersize=6, alpha=0.6)
        # 3. The Accelerated Result
        acc_marker, = axes_zoom.plot([], [], "rx", markersize=10, markeredgewidth=3, zorder=10, label="Aitken Result")

        info_text = axes_zoom.text(0.02, 0.95, "", transform=axes_zoom.transAxes,
                                   verticalalignment="top",
                                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
                                   fontsize=label_size)
        axes_zoom.legend(fontsize=label_size, loc="lower right")
        axes_zoom.grid(True, linestyle="--", alpha=0.5)

        def update(frame):
            step = self.solver.trace.steps[frame]

            # Calculate the accelerated guess (Step Result)
            # Note: The trace stores inputs. The result of this step is x for the next step.
            denominator = step.x - 2 * step.y + step.z
            if denominator == 0:
                # Fallback if Aitken fails (rare in solver, but possible in vis)
                acc_guess = step.z
            else:
                acc_guess = (step.x * step.z - step.y ** 2) / denominator

            # --- STEP A: CALCULATE ZOOM GEOMETRY ---
            # Focus on bounding box containing: x, y, z, and acc_guess
            xs_of_interest = [step.x, step.y, step.z, acc_guess]

            local_min = min(xs_of_interest)
            local_max = max(xs_of_interest)
            dist = local_max - local_min
            if dist == 0: dist = 0.1

            view_pad = dist * 0.4
            zoom_min = local_min - view_pad
            zoom_max = local_max + view_pad

            # Generate Y-Data for Zoom (Square aspect usually best for Cobweb,
            # but we stick to function limits to prevent squashing)
            x_zoom = np.linspace(zoom_min, zoom_max, 200)
            y_zoom = np.array([self.solver.function(x) for x in x_zoom])

            # Y-Limits: Must include the diagonal (y=x) and the curve values
            y_vals_zoom = list(y_zoom) + list(x_zoom)  # include diagonal range
            zoom_y_min = min(y_vals_zoom)
            zoom_y_max = max(y_vals_zoom)

            # --- STEP B: UPDATE GLOBAL VIEW ---
            global_current_point.set_data([step.x], [step.x])  # Start on diagonal
            global_acc_point.set_data([acc_guess], [acc_guess])  # Result on diagonal

            rect_width = zoom_max - zoom_min
            rect_height = zoom_y_max - zoom_y_min
            zoom_rectangle.set_bounds(zoom_min, zoom_y_min, rect_width, rect_height)

            # --- STEP C: UPDATE ZOOM VIEW ---
            axes_zoom.set_xlim(zoom_min, zoom_max)
            axes_zoom.set_ylim(zoom_y_min, zoom_y_max)

            zoom_curve.set_data(x_zoom, y_zoom)
            zoom_diagonal.set_data(x_zoom, x_zoom)

            # Cobweb Path Construction:
            # 1. Start at (x, x)
            # 2. Vertical to curve (x, y)  -> y = f(x)
            # 3. Horizontal to diagonal (y, y)
            # 4. Vertical to curve (y, z)  -> z = f(y)
            # 5. Horizontal to diagonal (z, z)
            cobweb_x = [step.x, step.x, step.y, step.y, step.z]
            cobweb_y = [step.x, step.y, step.y, step.z, step.z]

            cobweb_line.set_data(cobweb_x, cobweb_y)
            points_scatter.set_data([step.x, step.y, step.z],
                                    [step.x, step.y, step.z])  # Mark points on diagonal/curve?
            # Actually, let's mark the "inputs" on the diagonal for clarity
            # But standard cobweb visualizes intersection with curve.
            # Let's simplify: Mark the corners of the cobweb.
            points_scatter.set_data(cobweb_x, cobweb_y)

            # The accelerated point is strictly an X-axis adjustment,
            # but we usually visualize it projected onto the diagonal for the next step start.
            acc_marker.set_data([acc_guess], [acc_guess])

            info_text.set_text(
                rf"$n={step.iteration}$""\n"
                rf"$x^{{(n)}}={step.x:.5g}$""\n"
                rf"$\varphi(x^{{(n)}})={step.y:.5g}$""\n"
                rf"$\varphi(\varphi(x^{{(n)}}))={step.z:.5g}$""\n"
                rf"$x^{{(n+1)}}{acc_guess:.5g}$"
            )

            return global_current_point, global_acc_point, zoom_rectangle, zoom_curve, zoom_diagonal, cobweb_line, points_scatter, acc_marker

        animation = FuncAnimation(
            figure,
            update,
            frames=len(self.solver.trace.steps),
            interval=interval_ms, blit=False, repeat=False
        )

        plt.tight_layout()
        plt.show()
