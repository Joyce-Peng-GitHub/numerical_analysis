import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from typing import Tuple, override

from solvers.monadic.newton import NewtonSolver
from visualizers.monadic.monadic_equation_visualizer import MonadicEquationVisualizer


class NewtonVisualizer(MonadicEquationVisualizer):

    @override
    def animate(
            self,
            sample_num: int = MonadicEquationVisualizer.DEFAULT_SAMPLE_NUM,
            figure_size: Tuple[int, int] = MonadicEquationVisualizer.DEFAULT_FIGURE_SIZE,
            tick_size: int = MonadicEquationVisualizer.DEFAULT_TICK_SIZE,
            label_size: int = MonadicEquationVisualizer.DEFAULT_LABEL_SIZE,
            title_size: int = MonadicEquationVisualizer.DEFAULT_TITLE_SIZE,
            interval_ms: int = MonadicEquationVisualizer.DEFAULT_INTERVAL_MS
    ):
        if len(self.solver.trace.steps) == 0:
            raise ValueError("Cannot visualize an empty trace.")

        figure, (axes_global, axes_zoom) = plt.subplots(1, 2, figsize=figure_size)
        figure.suptitle(f"Newton's Method: {len(self.solver.trace.steps)} Iterations", fontsize=title_size)

        # --- 1. GLOBAL VIEW SETUP ---
        # Scan ALL steps to find the true Global Bounds.
        all_guesses = [s.guess for s in self.solver.trace.steps]
        # Include the final calculated "next" guess of the last step
        last_step = self.solver.trace.steps[-1]
        if last_step.derivative_value != 0:
            final_next = last_step.guess - (last_step.function_value / last_step.derivative_value)
            all_guesses.append(final_next)

        min_x, max_x = min(all_guesses), max(all_guesses)
        span = max_x - min_x
        # If span is 0 (e.g. 1 iteration perfect guess), add dummy padding
        if span == 0: span = 1.0

        pad = span * 0.2
        global_x_min, global_x_max = min_x - pad, max_x + pad

        # Generate Global Curve
        x_global = np.linspace(global_x_min, global_x_max, sample_num)
        y_global = np.array([self.solver.function(x) for x in x_global])

        axes_global.plot(x_global, y_global, "k-", alpha=0.3, label=r"f(x)")
        axes_global.axhline(0, color="black", linewidth=1)

        # Styling
        axes_global.set_title("Global Context (Fixed View)", fontsize=title_size)
        axes_global.tick_params(labelsize=tick_size)
        axes_global.set_xlabel(r"$x$", loc="center", fontsize=label_size)
        axes_global.set_xlim(global_x_min, global_x_max)

        # Handle Y-limits just for clarity (clipping extreme asymptotes)
        y_vis_min, y_vis_max = y_global.min(), y_global.max()
        axes_global.set_ylim(y_vis_min, y_vis_max)

        # Global Markers
        global_current_point, = axes_global.plot([], [], "go", markersize=4, label="Current")
        global_next_point, = axes_global.plot([], [], "rx", markersize=4, label="Next")

        # The "Camera Box"
        zoom_rectangle = Rectangle((0, 0), 1, 1, fill=False, edgecolor="purple", linestyle="--", linewidth=2,
                                   label="Camera View")
        axes_global.add_patch(zoom_rectangle)
        axes_global.legend(fontsize=label_size)

        # --- 2. ZOOM VIEW SETUP ---
        zoom_curve, = axes_zoom.plot([], [], "b-", linewidth=2, alpha=0.6)
        axes_zoom.axhline(0, color="black", linewidth=1)
        axes_zoom.set_title("Tangent Line View (Adaptive Zoom)", fontsize=title_size)
        axes_zoom.tick_params(labelsize=tick_size)
        axes_zoom.set_xlabel(r"$x$", loc="center", fontsize=label_size)

        # Visual Elements for Newton:
        vline_current = axes_zoom.axvline(0, color="green", linestyle=":", alpha=0.5)
        tangent_line, = axes_zoom.plot([], [], "r--", linewidth=1.5, label="Tangent")
        point_current, = axes_zoom.plot([], [], "go", markersize=8, zorder=5, label=r"$(x_n, f(x_n))$")
        point_next, = axes_zoom.plot([], [], "rx", markersize=8, zorder=5, label=r"$x^{{(n+1)}}$")

        info_text = axes_zoom.text(0.02, 0.95, "", transform=axes_zoom.transAxes,
                                   verticalalignment="top",
                                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
                                   fontsize=label_size)
        axes_zoom.legend(fontsize=label_size)

        def update(frame):
            step = self.solver.trace.steps[frame]

            # Newton Logic: x_new = x - f(x)/f'(x)
            if step.derivative_value == 0:
                x_next = step.guess
            else:
                x_next = step.guess - (step.function_value / step.derivative_value)

            # --- STEP A: CALCULATE ZOOM GEOMETRY ---
            focus_x_min = min(step.guess, x_next)
            focus_x_max = max(step.guess, x_next)
            dist = focus_x_max - focus_x_min

            if dist == 0: dist = 0.1

            view_pad_x = dist * 0.5
            zoom_x_min = focus_x_min - view_pad_x
            zoom_x_max = focus_x_max + view_pad_x

            x_zoom = np.linspace(zoom_x_min, zoom_x_max, 200)
            y_zoom = np.array([self.solver.function(x) for x in x_zoom])

            focus_y_vals = [0, step.function_value]
            y_min_raw, y_max_raw = min(focus_y_vals), max(focus_y_vals)
            y_range = y_max_raw - y_min_raw
            if y_range == 0: y_range = 1.0

            view_pad_y = y_range * 0.4
            zoom_y_min = y_min_raw - view_pad_y
            zoom_y_max = y_max_raw + view_pad_y

            # --- STEP B: UPDATE GLOBAL VIEW ---
            global_current_point.set_data([step.guess], [step.function_value])
            global_next_point.set_data([x_next], [0])

            rect_width = zoom_x_max - zoom_x_min
            rect_height = zoom_y_max - zoom_y_min
            zoom_rectangle.set_bounds(zoom_x_min, zoom_y_min, rect_width, rect_height)

            # --- STEP C: UPDATE ZOOM VIEW ---
            axes_zoom.set_xlim(zoom_x_min, zoom_x_max)
            axes_zoom.set_ylim(zoom_y_min, zoom_y_max)

            zoom_curve.set_data(x_zoom, y_zoom)
            vline_current.set_xdata([step.guess, step.guess])
            tangent_line.set_data([step.guess, x_next], [step.function_value, 0])
            point_current.set_data([step.guess], [step.function_value])
            point_next.set_data([x_next], [0])

            info_text.set_text(
                rf"$n={step.iteration}$""\n"
                rf"$x^{{(n)}}={step.guess:.4g}$""\n"
                rf"$f(x^{{(n)}})={step.function_value:.4g}$""\n"
                rf"$f'(x^{{(n)}})={step.derivative_value:.4g}$""\n"
                rf"$x^{{(n+1)}}={x_next:.4g}$"
            )

            return global_current_point, global_next_point, zoom_rectangle, zoom_curve, tangent_line, point_current, point_next, vline_current

        animation = FuncAnimation(
            figure,
            update,
            frames=len(self.solver.trace.steps),
            interval=interval_ms, blit=False, repeat=False
        )

        plt.tight_layout()
        plt.show()