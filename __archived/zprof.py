import time
import os

import time
import logging
import json
from threading import Lock
from pathlib import Path
from pprint import pprint
from tqdm import tqdm
from halib import *

import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class zProfiler:
    """A singleton profiler to measure execution time of contexts and steps.

    Args:
        interval_report (int): Frequency of periodic reports (0 to disable).
        stop_to_view (bool): Pause execution to view reports if True (only in debug mode).
        output_file (str): Path to save the profiling report.
        report_format (str): Output format for reports ("json" or "csv").

    Example:
        prof = zProfiler(interval_report=5)
        prof.ctx_start("my_context")
        prof.step_start("my_context", "step1")
        time.sleep(0.1)
        prof.step_end("my_context", "step1")
        prof.ctx_end("my_context")
    """

    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(
        self,
    ):
        if not hasattr(self, "_initialized"):
            self.time_dict = {}
            self._initialized = True

    def ctx_start(self, ctx_name="ctx_default"):
        if not isinstance(ctx_name, str) or not ctx_name:
            raise ValueError("ctx_name must be a non-empty string")
        if ctx_name not in self.time_dict:
            self.time_dict[ctx_name] = {
                "start": time.perf_counter(),
                "step_dict": {},
                "report_count": 0,
            }
        self.time_dict[ctx_name]["report_count"] += 1

    def ctx_end(self, ctx_name="ctx_default", report_func=None):
        if ctx_name not in self.time_dict:
            return
        self.time_dict[ctx_name]["end"] = time.perf_counter()
        self.time_dict[ctx_name]["duration"] = (
            self.time_dict[ctx_name]["end"] - self.time_dict[ctx_name]["start"]
        )

    def step_start(self, ctx_name, step_name):
        if not isinstance(step_name, str) or not step_name:
            raise ValueError("step_name must be a non-empty string")
        if ctx_name not in self.time_dict:
            return
        if step_name not in self.time_dict[ctx_name]["step_dict"]:
            self.time_dict[ctx_name]["step_dict"][step_name] = []
        self.time_dict[ctx_name]["step_dict"][step_name].append([time.perf_counter()])

    def step_end(self, ctx_name, step_name):
        if (
            ctx_name not in self.time_dict
            or step_name not in self.time_dict[ctx_name]["step_dict"]
        ):
            return
        self.time_dict[ctx_name]["step_dict"][step_name][-1].append(time.perf_counter())

    def _step_dict_to_detail(self, ctx_step_dict):
        """
                'ctx_step_dict': {
        │   │   'preprocess': [
        │   │   │   [278090.947465806, 278090.960484853],
        │   │   │   [278091.178424035, 278091.230944486],
        │   │   'infer': [
        │   │   │   [278090.960490534, 278091.178424035],
        │   │   │   [278091.230944486, 278091.251378469],
        │   }
        """
        assert len(ctx_step_dict.keys()) > 1, (
            "step_dict must have only one key (step_name) for detail."
        )

        for step_name, time_list in ctx_step_dict.items():
            normed_ctx_step_dict = {}
            if not isinstance(ctx_step_dict[step_name], list):
                raise ValueError(f"Step data for {step_name} must be a list")
            step_name = list(ctx_step_dict.keys())[0]
            normed_time_ls = []
            for idx, time_data in enumerate(time_list):
                elapsed_time = -1
                if len(time_data) == 2:
                    start, end = time_data[0], time_data[1]
                    elapsed_time = end - start
                normed_time_ls.append((idx, elapsed_time))  # including step
            normed_ctx_step_dict[step_name] = normed_time_ls
            return normed_ctx_step_dict

    def get_report_dict(self, with_detail=False):
        report_dict = {}
        for ctx_name, ctx_dict in self.time_dict.items():
            report_dict[ctx_name] = {
                "duration": ctx_dict.get("duration", 0.0),
                "step_dict": {
                    "summary": {"avg_time": {}, "percent_time": {}},
                    "detail": {},
                },
            }

            if with_detail:
                report_dict[ctx_name]["step_dict"]["detail"] = (
                    self._step_dict_to_detail(ctx_dict["step_dict"])
                )
            avg_time_list = []
            epsilon = 1e-5
            for step_name, step_list in ctx_dict["step_dict"].items():
                durations = []
                try:
                    for time_data in step_list:
                        if len(time_data) != 2:
                            continue
                        start, end = time_data
                        durations.append(end - start)
                except Exception as e:
                    logging.error(
                        f"Error processing step {step_name} in context {ctx_name}: {e}"
                    )
                    continue
                if not durations:
                    continue
                avg_time = sum(durations) / len(durations)
                if avg_time < epsilon:
                    continue
                avg_time_list.append((step_name, avg_time))
            total_avg_time = (
                sum(time for _, time in avg_time_list) or 1e-10
            )  # Avoid division by zero
            for step_name, avg_time in avg_time_list:
                report_dict[ctx_name]["step_dict"]["summary"]["percent_time"][
                    f"per_{step_name}"
                ] = (avg_time / total_avg_time) * 100.0
                report_dict[ctx_name]["step_dict"]["summary"]["avg_time"][
                    f"avg_{step_name}"
                ] = avg_time
            report_dict[ctx_name]["step_dict"]["summary"]["total_avg_time"] = (
                total_avg_time
            )
            report_dict[ctx_name]["step_dict"]["summary"] = dict(
                sorted(report_dict[ctx_name]["step_dict"]["summary"].items())
            )
        return report_dict

    @classmethod
    def plot_formatted_data(
        cls, profiler_data, outdir=None, file_format="png", do_show=False
    ):
        """
        Plot each context in a separate figure with bar + pie charts using consistent legend colors.
        Optionally save each figure in the specified format (png or svg).

        Args:
            profiler_data (dict): Nested profiling data
            outdir (str): Directory to save figures. If None, figures are only shown.
            file_format (str): Target file format, "png" or "svg". Default is "png".
        """
        if outdir is not None:
            os.makedirs(outdir, exist_ok=True)

        if file_format.lower() not in ["png", "svg"]:
            raise ValueError("file_format must be 'png' or 'svg'")
        results = {}  # {context: fig}

        for ctx, ctx_data in profiler_data.items():
            summary = ctx_data["step_dict"]["summary"]
            avg_times = summary["avg_time"]
            percent_times = summary["percent_time"]

            # Create figure with 1 row, 2 columns
            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=[f"Avg Time", f"% Time"],
                specs=[[{"type": "bar"}, {"type": "pie"}]],
            )

            # Extract step names and define a color palette
            step_names = [
                step_name.replace("avg_", "") for step_name in list(avg_times.keys())
            ]
            # Define a color palette (you can customize these colors)
            colors = [
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
                "#7f7f7f",
                "#bcbd22",
                "#17becf",
            ][: len(step_names)]  # Use enough colors for the steps

            # Create a color map to ensure consistency
            color_map = dict(zip(step_names, colors))

            # Bar chart
            fig.add_trace(
                go.Bar(
                    x=step_names,
                    y=list(avg_times.values()),
                    text=[f"{v * 1000:.2f} ms" for v in avg_times.values()],
                    textposition="outside",
                    marker=dict(
                        color=[
                            color_map[name] for name in step_names
                        ],  # Assign consistent colors
                        # Optional: Add patterns if desired (uncomment to use)
                        # pattern=dict(shape=["/", "\\", "x", "+"][:len(step_names)])
                    ),
                    name="",  # Unified legend name
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

            # Pie chart
            fig.add_trace(
                go.Pie(
                    labels=step_names,
                    values=list(percent_times.values()),
                    marker=dict(
                        colors=[
                            color_map[name] for name in step_names
                        ],  # Use same colors
                        # Patterns are not supported in pie charts
                    ),
                    name="",  # Same legend name as bar chart
                    hole=0.4,
                    showlegend=True,
                ),
                row=1,
                col=2,
            )

            # Update layout to customize legend and figure
            fig.update_layout(
                title_text=f"Context Profiler: {ctx}",
                width=1000,
                height=400,
                showlegend=True,
                legend=dict(
                    title="Steps",
                    x=1.05,  # Position legend to the right
                    y=0.5,
                    traceorder="normal",
                ),
                # Ensure consistent hover info
                hovermode="x unified" if fig.data[0].type == "bar" else "closest",
            )

            # Update bar chart axes for clarity
            fig.update_xaxes(title_text="Steps", row=1, col=1)
            fig.update_yaxes(title_text="Avg Time (ms)", row=1, col=1)

            # Show figure
            if do_show:
                fig.show()

            # Save figure if outdir is provided
            if outdir is not None:
                file_path = os.path.join(outdir, f"{ctx}_summary.{file_format.lower()}")
                fig.write_image(file_path)
                print(f"Saved figure: {file_path}")

            results[ctx] = fig
        return results

    def report_and_plot(self, outdir=None, file_format="png", do_show=False):
        """
        Generate the profiling report and plot the formatted data.

        Args:
            outdir (str): Directory to save figures. If None, figures are only shown.
            file_format (str): Target file format, "png" or "svg". Default is "png".
            do_show (bool): Whether to display the plots. Default is False.
        """
        report = self.get_report_dict()
        self.get_report_dict(with_detail=False)
        return self.plot_formatted_data(
            report, outdir=outdir, file_format=file_format, do_show=do_show
        )

    def save_report_dict(self, output_file, with_detail=False):
        try:
            report = self.get_report_dict(with_detail=with_detail)
            with open(output_file, "w") as f:
                json.dump(report, f, indent=4)
        except Exception as e:
            logging.error(f"Failed to save report to {output_file}: {e}")


def function1():
    prof = zProfiler()
    # pprint(prof.time_dict)
    prof.ctx_start("function1")
    for i in range(20):
        prof.step_start("function1", f"step_{i}")
        time.sleep(np.random.uniform(0.01, 0.05))
        prof.step_end("function1", f"step_{i}")
    prof.ctx_end("function1")


def testPlot():
    # --- Test data ---
    test_data = {
        "pipeline": {
            "duration": 0.960867,
            "step_dict": {
                "summary": {
                    "avg_time": {
                        "avg_read_frame": 0.00558,
                        "avg_big_model_infer": 0.03710,
                    },
                    "percent_time": {
                        "per_read_frame": 13.08,
                        "per_big_model_infer": 86.92,
                    },
                    "total_avg_time": 0.04268,
                },
                "detail": {},
            },
        },
        "big_model_infer": {
            "duration": 0.93146,
            "step_dict": {
                "summary": {
                    "avg_time": {
                        "avg_preprocess": 0.01042,
                        "avg_infer": 0.02666,
                    },
                    "percent_time": {
                        "per_preprocess": 28.10,
                        "per_infer": 71.89,
                    },
                    "total_avg_time": 0.03709,
                },
                "detail": {},
            },
        },
        "post_process": {
            "duration": 0.20012,
            "step_dict": {
                "summary": {
                    "avg_time": {
                        "avg_decode": 0.05012,
                        "avg_resize": 0.02000,
                    },
                    "percent_time": {
                        "per_decode": 71.0,
                        "per_resize": 29.0,
                    },
                    "total_avg_time": 0.07012,
                },
                "detail": {},
            },
        },
    }

    # --- Run test ---
    zProfiler.plot_formatted_data(
        test_data, outdir="zprof_plot", file_format="png", do_show=False
    )


from halib import *


def main():
    prof = zProfiler()
    for i in tqdm(range(10)):
        function1()
    pprint(prof.get_report_dict(with_detail=False))
    prof.report_and_plot(outdir="zprof_plot", file_format="png", do_show=False)
    # testPlot()


if __name__ == "__main__":
    main()
