
"""
Gradio entrypoint for the Sensors Anomalies demo.

Users upload a CSV file at runtime, which can be in one of three formats:
1. Long format: Already has series_id, timestamp, signal, value, [label] columns
2. Semi-long format: Has timestamp, signal/sensor, value columns (no series_id) - will be normalized
3. Wide format: Has timestamp + multiple sensor columns - will be melted to long format

This design works both locally and on Hugging Face Spaces without requiring
dataset files in the repository or Kaggle credentials.
"""

from __future__ import annotations

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from sensors_anomalies.algorithms.registry import list_algorithms, run_algorithm
from sensors_anomalies.utils.csv_utils import (
    is_long_format,
    is_semi_long_format,
    read_uploaded_csv,
    transform_wide_to_long,
    validate_and_normalize_long_format,
)


def build_app() -> gr.Blocks:
    """
    Build the Gradio application.

    Returns
    -------
    gradio.Blocks
        The Gradio Blocks app.
    """
    algorithms = list_algorithms()

    with gr.Blocks(title="Sensors Anomalies Demo") as demo:
        gr.Markdown("# 🔍 Sensors Anomalies Detection")
        gr.Markdown(
            "Many scientific or industrial applications rely on sensor data. "
            "Hence, the importance to detect sensor anomalies before to build datasets for further use cases.\n\n"
            "This application allows to test several approaches for such detections. "
            "Upload a CSV file with sensor time-series data and start testing !\n\n"
            "An example dataset that can be used : https://www.kaggle.com/datasets/arashnic/sensor-fault-detection-data"
        )
        gr.Markdown("""
**📝 CSV Format Requirements:**
- **Column separator:** comma `,` or semicolon `;` (auto-detected)
- **Decimal separator:** period `.` only (e.g., 3.14, not 3,14)

**Supported Formats:**
1. **Long format** (ready to use): `series_id`, `timestamp`, `signal`, `value`
2. **Semi-long format** (will be normalized): `timestamp`, `signal/sensor/SensorId`, `value/Value`
3. **Wide format** (will be transformed): `timestamp` + multiple sensor columns (e.g., `sensor_00`, `sensor_01`...)

**Acceptable Column Names:**
- Timestamp: `timestamp`, `Timestamp`, `time`, `Time`, `datetime`, `Datetime`, `date`, `Date`, `Horodatage`, `capture_date`, `measurement_date`
- Signal/Sensor: `signal`, `Signal`, `sensor`, `Sensor`, `SensorId`, `sensor_id`, `SensorName`
- Value: `value`, `Value`, `measurement`, `Measurement`, `reading`, `Reading`
        """)

        gr.Markdown("""
**🤖 Available Algorithms:**

This application provides three **unsupervised** anomaly detection algorithms that work **per-signal** (univariate analysis):

1. **Z-Score Detection**: Statistical method measuring how many standard deviations a point is from the mean (or median). Uses Modified Z-Score by default for robustness.
2. **IQR (Interquartile Range)**: Based on quartiles - detects outliers outside Q1-1.5×IQR and Q3+1.5×IQR bounds (Tukey's method).
3. **Isolation Forest**: Machine learning approach that isolates anomalies by randomly partitioning data. Anomalies require fewer splits to isolate.

All algorithms analyze each sensor signal independently and return anomaly scores (higher = more anomalous).
        """)

        gr.Markdown("## 📁 Data Upload")

        file_upload = gr.File(
            label="Upload CSV",
            file_types=[".csv"],
            type="binary",
        )

        process_btn = gr.Button("📊 Process CSV", variant="primary")

        gr.Markdown("## 📈 Data overview")

        status_output = gr.Textbox(
            label="Status",
            lines=5,
            placeholder="Upload a CSV to get started...",
        )

        data_preview = gr.Dataframe(
            label="Data Preview (Long Format) - First 10 rows",
            interactive=False,
            wrap=True,
        )

        gr.Markdown("## 📊 Raw Data Visualization")
        gr.Markdown("Explore your sensor data before running anomaly detection. Select one or more signals to visualize.")

        with gr.Row():
            raw_signal_selector = gr.CheckboxGroup(
                choices=[],
                label="Select Signals to Plot",
                value=[],
            )

        with gr.Row():
            raw_sample_rate = gr.Number(
                label="Sampling Rate (plot 1 out of K points)",
                value=1,
                minimum=1,
                step=1,
                info="Set K=1 to plot all points, K=10 to plot every 10th point for performance.",
            )

        raw_plot_btn = gr.Button("📈 Visualize Raw Data", variant="secondary")

        raw_signal_plot = gr.Plot(label="Raw Time Series Data")

        gr.Markdown("## 🎯 Anomaly Detection")
        gr.Markdown("Run anomaly detection algorithms on selected signals. Results will be displayed in the table below.")

        with gr.Row():
            detection_signal_selector = gr.CheckboxGroup(
                choices=[],
                label="Select Signals for Detection",
                value=[],
                info="Choose one or more signals to analyze"
            )

        with gr.Row():
            algos = gr.CheckboxGroup(
                choices=algorithms,
                label="Select Algorithms",
                value=algorithms[:1] if algorithms else [],
                info="Choose one or more detection algorithms"
            )

        run_btn = gr.Button("🚀 Run Detection", variant="primary")

        detection_output = gr.Textbox(
            label="Detection Summary",
            lines=10,
            placeholder="Run detection to see results...",
        )

        detection_results_table = gr.Dataframe(
            label="Detected Anomalies (Top 100 per algorithm, sorted by score)",
            interactive=False,
            wrap=True,
        )

        gr.Markdown("## 📊 Anomaly Visualization")
        gr.Markdown("Visualize detected anomalies for a single signal. Anomalous regions are shown as background shading (different colors per algorithm).")

        with gr.Row():
            viz_signal_selector = gr.Dropdown(
                choices=[],
                label="Select Signal to Visualize",
                value=None,
                info="Choose one signal for detailed analysis"
            )

        with gr.Row():
            viz_algorithm_filter = gr.CheckboxGroup(
                choices=[],
                label="Display Algorithms",
                value=[],
                info="Select which algorithms to show on the plot"
            )

        with gr.Row():
            viz_score_threshold = gr.Slider(
                label="Minimum Anomaly Score",
                minimum=0.0,
                maximum=10.0,
                value=0.0,
                step=0.1,
                info="Only show anomalies with score >= this value"
            )

        with gr.Row():
            viz_sample_rate = gr.Number(
                label="Sampling Rate (plot 1 out of K points)",
                value=1,
                minimum=1,
                step=1,
                info="Sampling for display performance only",
            )

        viz_plot_btn = gr.Button("📊 Generate Anomaly Plot", variant="secondary")

        anomaly_plot = gr.Plot(label="Anomaly Detection Results")

        gr.Markdown("## ⏱️ Detect Period of Anomalies")
        gr.Markdown(
            "Detect **anomaly periods** from point-by-point detection results. "
            "A period is identified when **K out of N consecutive points** exceed a score threshold. "
            "**Note:** You must run 'Run Detection' first to generate point-by-point results."
        )
        gr.Markdown("""
**How it works:**
1. First, run the point-by-point detection above to generate anomaly scores for each timestamp
2. Set your parameters below:
   - **K**: Minimum number of anomalous points required in the window
   - **N**: Size of the forward-looking window to check
   - **Score Threshold**: Minimum anomaly score for a point to be considered anomalous
3. The algorithm checks each point: if K out of the next N consecutive points exceed the threshold, that point is marked as part of a period
4. Consecutive marked points are merged into a single period

**Example:** With K=3, N=5, threshold=2.0:
- At each timestamp, look at the next 5 consecutive points (including current)
- If at least 3 of those 5 points have score ≥ 2.0, mark this timestamp as part of an anomaly period
- All consecutive marked timestamps form continuous periods (timestamp_start to timestamp_end)

**Tip:** Use the point-by-point detection visualization above to explore score distributions and set appropriate thresholds.
        """)

        with gr.Row():
            period_k = gr.Number(
                label="K - Minimum anomalous points in window",
                value=3,
                minimum=1,
                step=1,
                info="At least K points must exceed threshold in window"
            )
            period_n = gr.Number(
                label="N - Window size",
                value=5,
                minimum=1,
                step=1,
                info="Size of sliding window to check"
            )
            period_threshold = gr.Number(
                label="Score Threshold",
                value=2.0,
                minimum=0.0,
                step=0.1,
                info="Minimum score for a point to be considered anomalous"
            )

        period_detect_btn = gr.Button("🔍 Detect Anomaly Periods", variant="primary")

        period_detection_output = gr.Textbox(
            label="Period Detection Summary",
            lines=10,
            placeholder="Run period detection to see results...",
        )

        gr.Markdown("## 📋 Period of Anomalies Results")
        gr.Markdown("Statistics and details of detected anomaly periods. Shows the first 10 periods for each algorithm.")

        period_results_table = gr.Dataframe(
            label="Detected Anomaly Periods (First 10 per algorithm)",
            interactive=False,
            wrap=True,
        )

        gr.Markdown("## 📊 Period of Anomalies Visualization")
        gr.Markdown(
            "Visualize anomaly periods as horizontal colored bands on the time series. "
            "Each algorithm's periods are shown in different colors. "
            "Select a signal to see its anomaly periods."
        )

        with gr.Row():
            period_viz_signal = gr.Dropdown(
                choices=[],
                label="Select Signal to Visualize",
                value=None,
                info="Choose one signal for period visualization"
            )

        with gr.Row():
            period_viz_algorithms = gr.CheckboxGroup(
                choices=[],
                label="Display Algorithms",
                value=[],
                info="Select which algorithms' periods to show"
            )

        with gr.Row():
            period_viz_sample_rate = gr.Number(
                label="Sampling Rate (plot 1 out of K points)",
                value=1,
                minimum=1,
                step=1,
                info="Sampling for display performance only",
            )

        period_viz_btn = gr.Button("📊 Visualize Period Anomalies", variant="secondary")

        period_viz_plot = gr.Plot(label="Anomaly Periods Visualization")

        # Store the processed dataframe in state
        processed_df_state = gr.State(value=None)
        available_signals_state = gr.State(value=[])
        detection_results_state = gr.State(value=None)
        period_results_state = gr.State(value=None)

        def process_csv(
            file_bytes: bytes | None,
        ) -> tuple[str, pd.DataFrame | None, pd.DataFrame | None, gr.CheckboxGroup, gr.CheckboxGroup, gr.Dropdown, list[str]]:
            """
            Process uploaded CSV file.

            Parameters
            ----------
            file_bytes : bytes or None
                Uploaded file bytes.

            Returns
            -------
            tuple[str, pd.DataFrame | None, pd.DataFrame | None, gr.CheckboxGroup, gr.CheckboxGroup, gr.Dropdown, list[str]]
                Status message, preview dataframe, full dataframe for state,
                updated raw signal selector, updated detection signal selector,
                updated viz signal dropdown, and available signals list.
            """
            if file_bytes is None:
                return (
                    "❌ Please upload a CSV file",
                    None,
                    None,
                    gr.CheckboxGroup(choices=[], value=[]),
                    gr.CheckboxGroup(choices=[], value=[]),
                    gr.Dropdown(choices=[], value=None),
                    []
                )

            try:
                # Read CSV
                df = read_uploaded_csv(file_bytes)
                status_lines = [f"✅ CSV loaded: {len(df)} rows, {len(df.columns)} columns"]

                # Check if already in long format
                if is_long_format(df):
                    status_lines.append("✅ Data is already in long format (has series_id, timestamp, signal, value)")
                    df_long = validate_and_normalize_long_format(df)
                elif is_semi_long_format(df):
                    status_lines.append("⚙️ Data is in semi-long format (has timestamp, signal/sensor, value)")
                    df_long = transform_wide_to_long(df)
                    status_lines.append("✅ Normalized to long format (added series_id)")
                else:
                    status_lines.append("⚙️ Data is in wide format (has timestamp + multiple sensor columns)")
                    # Always use auto-detection
                    df_long = transform_wide_to_long(df)
                    status_lines.append("✅ Transformed to long format")

                # Ensure signal column is always string type for consistent filtering
                df_long["signal"] = df_long["signal"].astype(str)

                # Get statistics
                n_series = df_long["series_id"].nunique()
                n_signals = df_long["signal"].nunique()
                time_range = df_long["timestamp"].agg(["min", "max"])

                # Detect timezone info
                if isinstance(df_long["timestamp"].dtype, pd.DatetimeTZDtype):
                    tz_info = str(df_long["timestamp"].dt.tz)
                    tz_display = f"  • Timezone: {tz_info}"
                else:
                    tz_display = "  • Timezone: Not specified (timezone-naive)"

                status_lines.extend([
                    "\n📊 Dataset Statistics:",
                    f"  • Series: {n_series}",
                    f"  • Signals: {n_signals}",
                    f"  • Total readings: {len(df_long)}",
                    f"  • Time range: {time_range['min']} to {time_range['max']}",
                    tz_display,
                ])

                status = "\n".join(status_lines)
                # Show first 10 rows, excluding series_id for cleaner preview
                preview_cols = [col for col in df_long.columns if col != "series_id"]
                preview = df_long[preview_cols].head(10)

                # Get available signals for plotting
                signals = sorted(df_long["signal"].unique().tolist())

                # Update all signal selectors
                raw_signal_selector_update = gr.CheckboxGroup(
                    choices=signals,
                    value=signals[:3] if len(signals) >= 3 else signals,  # Select first 3 by default
                    label="Select Signals to Plot",
                )

                detection_signal_selector_update = gr.CheckboxGroup(
                    choices=signals,
                    value=[],  # None selected by default for detection
                    label="Select Signals for Detection",
                    info="Choose one or more signals to analyze"
                )

                viz_signal_selector_update = gr.Dropdown(
                    choices=signals,
                    value=signals[0] if signals else None,  # Select first signal by default
                    label="Select Signal to Visualize",
                    info="Choose one signal for detailed analysis"
                )

                return status, preview, df_long, raw_signal_selector_update, detection_signal_selector_update, viz_signal_selector_update, signals

            except Exception as e:
                return (
                    f"❌ Error processing CSV: {str(e)}",
                    None,
                    None,
                    gr.CheckboxGroup(choices=[], value=[]),
                    gr.CheckboxGroup(choices=[], value=[]),
                    gr.Dropdown(choices=[], value=None),
                    []
                )

        def run_detection(
            df_long: pd.DataFrame | None,
            selected_signals: list[str],
            selected_algos: list[str],
        ) -> tuple[str, pd.DataFrame | None, pd.DataFrame | None, gr.CheckboxGroup, gr.Dropdown, gr.Slider]:
            """
            Run anomaly detection algorithms.

            Parameters
            ----------
            df_long : pd.DataFrame or None
                Processed long-format dataframe.
            selected_signals : list[str]
                Selected signal names for detection.
            selected_algos : list[str]
                Selected algorithm ids.

            Returns
            -------
            tuple[str, pd.DataFrame | None, pd.DataFrame | None, gr.CheckboxGroup, gr.Dropdown, gr.Slider]
                Detection results message, detected anomalies dataframe for display,
                full results, updated algorithm filter, updated viz signal dropdown, and updated slider.
            """
            if df_long is None:
                return (
                    "❌ Please process a CSV file first",
                    None,
                    None,
                    gr.CheckboxGroup(choices=[], value=[]),
                    gr.Dropdown(choices=[], value=None),
                    gr.Slider(minimum=0.0, maximum=10.0, value=0.0, step=0.1, label="Minimum Anomaly Score")
                )

            if not selected_signals:
                return (
                    "❌ Please select at least one signal for detection",
                    None,
                    None,
                    gr.CheckboxGroup(choices=[], value=[]),
                    gr.Dropdown(choices=[], value=None),
                    gr.Slider(minimum=0.0, maximum=10.0, value=0.0, step=0.1, label="Minimum Anomaly Score")
                )

            if not selected_algos:
                return (
                    "❌ Please select at least one algorithm",
                    None,
                    None,
                    gr.CheckboxGroup(choices=[], value=[]),
                    gr.Dropdown(choices=[], value=None),
                    gr.Slider(minimum=0.0, maximum=10.0, value=0.0, step=0.1, label="Minimum Anomaly Score")
                )

            # Filter data to selected signals only
            # Ensure selected_signals are strings (they might come as indices)
            selected_signals = [str(sig) for sig in selected_signals]
            df_filtered = df_long[df_long["signal"].isin(selected_signals)].copy()

            # Check if filtered data is empty
            if len(df_filtered) == 0:
                available_signals = sorted(df_long["signal"].unique().tolist())
                return (
                    f"❌ No data found for selected signals: {', '.join(selected_signals)}\n\n"
                    f"Available signals in dataset: {', '.join(available_signals)}\n\n"
                    f"This usually happens due to a type mismatch. Please select different signals.",
                    None,
                    None,
                    gr.CheckboxGroup(choices=[], value=[]),
                    gr.Dropdown(choices=[], value=None),
                    gr.Slider(minimum=0.0, maximum=10.0, value=0.0, step=0.1, label="Minimum Anomaly Score")
                )

            # Run each selected algorithm and collect results
            all_results = []
            summary_lines = [
                "🎯 Detection Results:",
                f"\nSignals analyzed: {', '.join(selected_signals)}",
                f"Dataset: {len(df_filtered)} rows",
                ""
            ]

            for algo_id in selected_algos:
                try:
                    # Run algorithm on filtered data
                    result_df = run_algorithm(algo_id, df_filtered)

                    # Add algorithm id to the results
                    result_df["algorithm"] = algo_id

                    # Count anomalies
                    if "is_anomaly" in result_df.columns:
                        n_anomalies = result_df["is_anomaly"].sum()
                        pct_anomalies = (n_anomalies / len(result_df) * 100) if len(result_df) > 0 else 0
                    else:
                        n_anomalies = 0
                        pct_anomalies = 0

                    all_results.append(result_df)
                    summary_lines.append(
                        f"✅ {algo_id}: {n_anomalies:,} anomalies detected ({pct_anomalies:.2f}%)"
                    )

                except Exception as e:
                    summary_lines.append(f"❌ {algo_id}: Error - {e!s}")

            # Combine all results
            if all_results:
                combined_results = pd.concat(all_results, ignore_index=True)

                # Filter to only anomalies for display
                if "is_anomaly" in combined_results.columns:
                    anomalies_only = combined_results[combined_results["is_anomaly"]].copy()
                else:
                    anomalies_only = combined_results.copy()

                # Get top 100 anomalies per algorithm, sorted by score
                top_anomalies_per_algo = []
                if "score" in anomalies_only.columns:
                    for algo in selected_algos:
                        algo_anomalies = anomalies_only[anomalies_only["algorithm"] == algo].copy()
                        top_algo_anomalies = algo_anomalies.sort_values("score", ascending=False).head(100)
                        # Only append non-empty dataframes
                        if len(top_algo_anomalies) > 0:
                            top_anomalies_per_algo.append(top_algo_anomalies)

                    if top_anomalies_per_algo:
                        display_df = pd.concat(top_anomalies_per_algo, ignore_index=True)
                        # Sort by algorithm first, then by score within each algorithm
                        display_df = display_df.sort_values(["algorithm", "score"], ascending=[True, False])
                    else:
                        display_df = anomalies_only.head(100)
                else:
                    display_df = anomalies_only.head(100)

                # Format for display - only select columns if they exist and dataframe is not empty
                if len(display_df) > 0 and all(col in display_df.columns for col in ["algorithm", "series_id", "signal", "timestamp_start", "score", "is_anomaly"]):
                    display_df = display_df[
                        ["algorithm", "series_id", "signal", "timestamp_start", "score", "is_anomaly"]
                    ].copy()
                elif len(display_df) == 0:
                    # Create empty dataframe with proper columns
                    display_df = pd.DataFrame(columns=["algorithm", "series_id", "signal", "timestamp_start", "score", "is_anomaly"])

                total_anomalies = len(anomalies_only)
                shown_anomalies = len(display_df)

                if total_anomalies == 0:
                    summary_lines.append(f"\n⚠️ No anomalies detected by any algorithm. All data points appear normal.")
                    summary_lines.append("You can still visualize the results, but no anomalous regions will be highlighted.")
                else:
                    summary_lines.append(f"\n📊 Total anomalies: {total_anomalies:,} (showing top {shown_anomalies:,}: up to 100 per algorithm by score)")

                # Update algorithm filter with detected algorithms
                algo_filter_update = gr.CheckboxGroup(
                    choices=selected_algos,
                    value=selected_algos,  # All selected by default
                    label="Display Algorithms",
                    info="Select which algorithms to show on the plot"
                )

                # Update viz signal dropdown with detected signals
                viz_signal_dropdown_update = gr.Dropdown(
                    choices=selected_signals,
                    value=selected_signals[0] if selected_signals else None,
                    label="Select Signal to Visualize",
                    info="Choose one signal for detailed analysis"
                )

                # Calculate slider parameters - max capped at 100, default as median of anomalies
                if "score" in anomalies_only.columns and len(anomalies_only) > 0:
                    max_score_val = anomalies_only["score"].max()
                    median_score_val = anomalies_only["score"].median()

                    if pd.isna(max_score_val) or max_score_val == 0:
                        slider_max = 10.0
                        slider_default = 0.0
                    else:
                        # Cap max at 100, round up to nearest integer
                        slider_max = min(100.0, max(10.0, float(int(max_score_val) + 1)))
                        # Use median as default, or 0 if no valid median
                        slider_default = float(median_score_val) if not pd.isna(median_score_val) else 0.0
                else:
                    slider_max = 10.0
                    slider_default = 0.0

                slider_update = gr.Slider(
                    minimum=0.0,
                    maximum=slider_max,
                    value=slider_default,
                    step=0.1,
                    label="Minimum Anomaly Score",
                    info=f"Median anomaly score: {slider_default:.2f}" if len(anomalies_only) > 0 else "No anomalies detected"
                )

                return "\n".join(summary_lines), display_df, combined_results, algo_filter_update, viz_signal_dropdown_update, slider_update
            else:
                return (
                    "\n".join(summary_lines) + "\n\n⚠️ No results generated",
                    None,
                    None,
                    gr.CheckboxGroup(choices=[], value=[]),
                    gr.Dropdown(choices=[], value=None),
                    gr.Slider(minimum=0.0, maximum=10.0, value=0.0, step=0.1, label="Minimum Anomaly Score")
                )

        def plot_raw_data(
            df_long: pd.DataFrame | None,
            selected_signals: list[str],
            sample_rate: int,
        ) -> go.Figure:
            """
            Generate raw time series plot for selected signals (no anomalies).

            Parameters
            ----------
            df_long : pd.DataFrame or None
                Processed long-format dataframe.
            selected_signals : list[str]
                List of signal names to plot.
            sample_rate : int
                Sampling rate (plot 1 out of K points).

            Returns
            -------
            plotly.graph_objects.Figure
                Interactive plotly figure.
            """
            if df_long is None:
                fig = go.Figure()
                fig.add_annotation(
                    text="Please process a CSV file first",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=16),
                )
                fig.update_layout(xaxis_visible=False, yaxis_visible=False)
                return fig

            if not selected_signals:
                fig = go.Figure()
                fig.add_annotation(
                    text="Please select at least one signal to plot",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=16),
                )
                fig.update_layout(xaxis_visible=False, yaxis_visible=False)
                return fig

            # Convert sample_rate to int and ensure it's at least 1
            k = max(1, int(sample_rate))

            # Filter data for selected signals
            df_plot = df_long[df_long["signal"].isin(selected_signals)].copy()

            # Apply sampling
            if k > 1:
                df_plot = df_plot.iloc[::k]

            # Create figure
            fig = go.Figure()

            # Add a trace for each signal
            for signal in selected_signals:
                signal_data = df_plot[df_plot["signal"] == signal]

                fig.add_trace(go.Scatter(
                    x=signal_data["timestamp"],
                    y=signal_data["value"],
                    mode="lines",
                    name=signal,
                    line=dict(width=2),
                ))

            # Update layout
            n_points = len(df_plot)
            sampling_info = f" (sampling: 1/{k})" if k > 1 else ""

            fig.update_layout(
                title=f"Raw Data Visualization - {len(selected_signals)} signal(s), {n_points} points{sampling_info}",
                xaxis_title="Timestamp",
                yaxis_title="Value",
                hovermode="x unified",
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02,
                ),
                height=600,
            )

            return fig

        def plot_anomaly_visualization(
            df_long: pd.DataFrame | None,
            selected_signal: str | None,
            detection_results: pd.DataFrame | None,
            algorithm_filter: list[str] | None,
            min_score: float,
            sample_rate: int,
        ) -> go.Figure:
            """
            Generate anomaly visualization plot for a single signal with background shading.

            Parameters
            ----------
            df_long : pd.DataFrame or None
                Processed long-format dataframe.
            selected_signal : str or None
                Signal name to visualize.
            detection_results : pd.DataFrame or None
                Detection results with anomalies from different algorithms.
            algorithm_filter : list[str] or None
                List of algorithm names to display. If None, show all.
            min_score : float
                Minimum anomaly score to display.
            sample_rate : int
                Sampling rate (plot 1 out of K points) for display only.

            Returns
            -------
            plotly.graph_objects.Figure
                Interactive plotly figure with background shading.
            """
            if df_long is None:
                fig = go.Figure()
                fig.add_annotation(
                    text="Please process a CSV file first",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=16),
                )
                fig.update_layout(xaxis_visible=False, yaxis_visible=False)
                return fig

            if detection_results is None:
                fig = go.Figure()
                fig.add_annotation(
                    text="Please run anomaly detection first",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=16),
                )
                fig.update_layout(xaxis_visible=False, yaxis_visible=False)
                return fig

            if not selected_signal:
                fig = go.Figure()
                fig.add_annotation(
                    text="Please select a signal to visualize",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=16),
                )
                fig.update_layout(xaxis_visible=False, yaxis_visible=False)
                return fig

            # Convert sample_rate to int and ensure it's at least 1
            k = max(1, int(sample_rate))

            # Filter data for selected signal
            df_plot = df_long[df_long["signal"] == selected_signal].copy()

            # Apply sampling for display
            if k > 1:
                df_plot = df_plot.iloc[::k]

            # Create figure
            fig = go.Figure()

            # Plot signal as line
            fig.add_trace(go.Scatter(
                x=df_plot["timestamp"],
                y=df_plot["value"],
                mode="lines",
                name=selected_signal,
                line=dict(width=2, color="blue"),
            ))

            # Add anomaly scatter points if available
            if "is_anomaly" in detection_results.columns:
                # Filter anomalies for this signal
                anomalies = detection_results[
                    (detection_results["is_anomaly"]) &
                    (detection_results["signal"] == selected_signal)
                ].copy()

                # Check if there are any anomalies for this signal
                if len(anomalies) == 0:
                    # Add annotation when no anomalies detected
                    fig.add_annotation(
                        text="No anomalies detected for this signal",
                        xref="paper", yref="paper",
                        x=0.5, y=0.95,
                        showarrow=False,
                        font=dict(size=14, color="green"),
                        bgcolor="rgba(144, 238, 144, 0.3)",
                        bordercolor="green",
                        borderwidth=2,
                        borderpad=10,
                    )
                else:
                    # Apply score threshold filter
                    if "score" in anomalies.columns:
                        anomalies = anomalies[anomalies["score"] >= min_score].copy()

                    # Apply algorithm filter
                    if algorithm_filter and "algorithm" in anomalies.columns:
                        anomalies = anomalies[anomalies["algorithm"].isin(algorithm_filter)].copy()

                    # Define styles for algorithms (markers and colors)
                    algo_styles = {
                        "zscore": {"color": "red", "symbol": "diamond", "name": "Z-Score"},
                        "iqr": {"color": "orange", "symbol": "circle", "name": "IQR"},
                        "isolation_forest": {"color": "purple", "symbol": "square", "name": "Isolation Forest"},
                    }

                    # Get unique algorithms in the filtered results
                    algorithms_present = anomalies["algorithm"].unique() if "algorithm" in anomalies.columns else []

                    # Add scatter points for each algorithm
                    for algo in algorithms_present:
                        algo_anomalies = anomalies[anomalies["algorithm"] == algo].copy()

                        if len(algo_anomalies) == 0:
                            continue

                        # Merge with df_plot to get y-values for the anomalies
                        # Both should now have same timezone since detection preserves input timezone
                        merged = algo_anomalies.merge(
                            df_plot[["timestamp", "value"]],
                            left_on="timestamp_start",
                            right_on="timestamp",
                            how="inner"
                        )

                        if len(merged) == 0:
                            continue

                        style = algo_styles.get(algo, {"color": "red", "symbol": "circle", "name": algo})

                        # Add scatter trace for anomalies
                        fig.add_trace(go.Scatter(
                            x=merged["timestamp_start"],
                            y=merged["value"],
                            mode="markers",
                            name=f"🚨 {style['name']}",
                            marker=dict(
                                size=10,
                                color=style["color"],
                                symbol=style["symbol"],
                                line=dict(width=1, color="white")
                            ),
                            hovertemplate=f"<b>{style['name']}</b><br>" +
                                        "Time: %{x}<br>" +
                                        "Value: %{y:.2f}<br>" +
                                        "Score: %{customdata:.2f}<extra></extra>",
                            customdata=merged["score"],
                            showlegend=True,
                        ))

            # Update layout
            n_points = len(df_plot)
            sampling_info = f" (sampling: 1/{k})" if k > 1 else ""
            anomaly_info = ""
            if detection_results is not None and "is_anomaly" in detection_results.columns:
                total_anomalies = len(detection_results[
                    (detection_results["is_anomaly"]) &
                    (detection_results["signal"] == selected_signal)
                ])
                if total_anomalies > 0:
                    filtered_count = len(anomalies) if 'anomalies' in locals() else 0
                    filter_info = ""
                    if min_score > 0:
                        filter_info += f", score≥{min_score:.1f}"
                    if algorithm_filter:
                        filter_info += f", {len(algorithm_filter)} algo(s)"
                    anomaly_info = f" | {filtered_count}/{total_anomalies} anomalies shown{filter_info}"

            fig.update_layout(
                title=f"Anomaly Detection Results - {selected_signal}, {n_points} points{sampling_info}{anomaly_info}",
                xaxis_title="Timestamp",
                yaxis_title="Value",
                hovermode="x unified",
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02,
                ),
                height=600,
            )

            return fig

        def detect_periods(
            detection_results: pd.DataFrame | None,
            k: int,
            n: int,
            threshold: float,
        ) -> tuple[str, pd.DataFrame | None, pd.DataFrame | None, gr.CheckboxGroup, gr.Dropdown]:
            """
            Detect anomaly periods from point-by-point results.

            Parameters
            ----------
            detection_results : pd.DataFrame or None
                Point-by-point detection results
            k : int
                Minimum anomalous points in window
            n : int
                Window size
            threshold : float
                Score threshold

            Returns
            -------
            tuple[str, pd.DataFrame | None, pd.DataFrame | None, gr.CheckboxGroup, gr.Dropdown]
                Summary message, display dataframe, full results, algorithm selector update, signal dropdown update
            """
            # Import here to avoid circular imports
            from sensors_anomalies.algorithms.periods import detect_anomaly_periods

            if detection_results is None or detection_results.empty:
                return (
                    "❌ Please run point-by-point detection first (use 'Run Detection' button above)",
                    None,
                    None,
                    gr.CheckboxGroup(choices=[], value=[]),
                    gr.Dropdown(choices=[], value=None),
                )

            # Validate parameters
            k = int(k)
            n = int(n)

            if k > n:
                return (
                    f"❌ Invalid parameters: K ({k}) must be <= N ({n})",
                    None,
                    None,
                    gr.CheckboxGroup(choices=[], value=[]),
                    gr.Dropdown(choices=[], value=None),
                )

            if k < 1 or n < 1:
                return (
                    f"❌ Invalid parameters: K ({k}) and N ({n}) must be >= 1",
                    None,
                    None,
                    gr.CheckboxGroup(choices=[], value=[]),
                    gr.Dropdown(choices=[], value=None),
                )

            try:
                # Filter to only anomalous points before period detection
                if "is_anomaly" in detection_results.columns:
                    anomalous_points = detection_results[detection_results["is_anomaly"]].copy()
                else:
                    anomalous_points = detection_results.copy()

                if anomalous_points.empty:
                    summary = (
                        f"⚠️ No anomalies found in point detection results\n\n"
                        f"Parameters: K={k}, N={n}, Threshold={threshold}\n\n"
                        f"Cannot detect periods without anomalous points.\n"
                        f"Try running point detection with different algorithms or parameters first."
                    )
                    return (
                        summary,
                        None,
                        None,
                        gr.CheckboxGroup(choices=[], value=[]),
                        gr.Dropdown(choices=[], value=None),
                    )

                # Run period detection on anomalous points only
                period_results = detect_anomaly_periods(
                    anomalous_points,
                    k=k,
                    n=n,
                    score_threshold=threshold,
                )

                if period_results.empty:
                    summary = (
                        f"⚠️ No anomaly periods detected\n\n"
                        f"Parameters: K={k}, N={n}, Threshold={threshold}\n"
                        f"Analyzed {len(detection_results)} points\n\n"
                        f"Try:\n"
                        f"  • Lowering the score threshold\n"
                        f"  • Decreasing K (fewer anomalous points required)\n"
                        f"  • Increasing N (larger window size)"
                    )
                    return (
                        summary,
                        None,
                        None,
                        gr.CheckboxGroup(choices=[], value=[]),
                        gr.Dropdown(choices=[], value=None),
                    )

                # Calculate statistics per algorithm
                summary_lines = [
                    "✅ Period Detection Complete",
                    f"\nParameters: K={k} out of N={n}, Score Threshold={threshold}",
                    f"Input: {len(detection_results)} points",
                    f"\n📊 Results by Algorithm:",
                ]

                algorithms = sorted(period_results["algorithm"].unique())
                signals = sorted(period_results["signal"].unique())

                for algo in algorithms:
                    algo_periods = period_results[period_results["algorithm"] == algo]
                    n_periods = len(algo_periods)

                    if n_periods > 0:
                        longest = algo_periods["n_points"].max()
                        shortest = algo_periods["n_points"].min()
                        mean_duration = algo_periods["n_points"].mean()

                        summary_lines.extend([
                            f"\n{algo}:",
                            f"  • Periods detected: {n_periods}",
                            f"  • Longest period: {longest} timestamps",
                            f"  • Shortest period: {shortest} timestamps",
                            f"  • Mean duration: {mean_duration:.1f} timestamps",
                        ])

                # Create display dataframe - first 10 periods per algorithm
                display_rows = []
                for algo in algorithms:
                    algo_periods = period_results[period_results["algorithm"] == algo].head(10)
                    display_rows.append(algo_periods)

                if display_rows:
                    display_df = pd.concat(display_rows, ignore_index=True)
                    # Sort by algorithm, then by timestamp
                    display_df = display_df.sort_values(["algorithm", "timestamp_start"])
                else:
                    display_df = period_results.head(10)

                summary_lines.append(f"\n📋 Showing first 10 periods per algorithm in table below")
                summary_lines.append(f"Total periods: {len(period_results)}")

                # Update UI components
                algo_selector_update = gr.CheckboxGroup(
                    choices=algorithms,
                    value=algorithms,
                    label="Display Algorithms",
                    info="Select which algorithms' periods to show"
                )

                signal_dropdown_update = gr.Dropdown(
                    choices=signals,
                    value=signals[0] if signals else None,
                    label="Select Signal to Visualize",
                    info="Choose one signal for period visualization"
                )

                return (
                    "\n".join(summary_lines),
                    display_df,
                    period_results,
                    algo_selector_update,
                    signal_dropdown_update,
                )

            except Exception as e:
                return (
                    f"❌ Error during period detection: {e!s}",
                    None,
                    None,
                    gr.CheckboxGroup(choices=[], value=[]),
                    gr.Dropdown(choices=[], value=None),
                )

        def plot_period_visualization(
            df_long: pd.DataFrame | None,
            selected_signal: str | None,
            period_results: pd.DataFrame | None,
            algorithm_filter: list[str] | None,
            sample_rate: int,
        ) -> go.Figure:
            """
            Generate period visualization with horizontal bands.

            Parameters
            ----------
            df_long : pd.DataFrame or None
                Processed long-format dataframe
            selected_signal : str or None
                Signal to visualize
            period_results : pd.DataFrame or None
                Period detection results
            algorithm_filter : list[str] or None
                Algorithms to display
            sample_rate : int
                Sampling rate for display

            Returns
            -------
            go.Figure
                Plotly figure with period bands
            """
            if df_long is None:
                fig = go.Figure()
                fig.add_annotation(
                    text="Please process a CSV file first",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=16),
                )
                fig.update_layout(xaxis_visible=False, yaxis_visible=False)
                return fig

            if period_results is None or period_results.empty:
                fig = go.Figure()
                fig.add_annotation(
                    text="Please run period detection first",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=16),
                )
                fig.update_layout(xaxis_visible=False, yaxis_visible=False)
                return fig

            if not selected_signal:
                fig = go.Figure()
                fig.add_annotation(
                    text="Please select a signal to visualize",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=16),
                )
                fig.update_layout(xaxis_visible=False, yaxis_visible=False)
                return fig

            # Filter data for selected signal
            k = max(1, int(sample_rate))
            df_plot = df_long[df_long["signal"] == selected_signal].copy()

            if k > 1:
                df_plot = df_plot.iloc[::k]

            # Create figure
            fig = go.Figure()

            # Plot signal line first
            fig.add_trace(go.Scatter(
                x=df_plot["timestamp"],
                y=df_plot["value"],
                mode="lines",
                name=selected_signal,
                line=dict(width=2, color="blue"),
                showlegend=True,
            ))

            # Filter periods for selected signal
            signal_periods = period_results[period_results["signal"] == selected_signal].copy()

            if algorithm_filter:
                signal_periods = signal_periods[signal_periods["algorithm"].isin(algorithm_filter)]

            if len(signal_periods) == 0:
                fig.add_annotation(
                    text="No anomaly periods detected for this signal with selected algorithms",
                    xref="paper", yref="paper",
                    x=0.5, y=0.95,
                    showarrow=False,
                    font=dict(size=14, color="green"),
                    bgcolor="rgba(144, 238, 144, 0.3)",
                    bordercolor="green",
                    borderwidth=2,
                    borderpad=10,
                )
            else:
                # Define colors for algorithms (with transparency)
                algo_colors = {
                    "zscore": "rgba(255, 0, 0, 0.3)",  # Red with transparency
                    "iqr": "rgba(255, 165, 0, 0.3)",  # Orange with transparency
                    "isolation_forest": "rgba(128, 0, 128, 0.3)",  # Purple with transparency
                }

                algo_names = {
                    "zscore": "Z-Score",
                    "iqr": "IQR",
                    "isolation_forest": "Isolation Forest",
                }

                # Get y-axis range for bands
                y_min = df_plot["value"].min()
                y_max = df_plot["value"].max()
                y_range = y_max - y_min
                margin = y_range * 0.05 if y_range > 0 else 1
                y_lower = y_min - margin
                y_upper = y_max + margin

                # Group periods by algorithm for better legend control
                algorithms_with_periods = signal_periods["algorithm"].unique()

                for algo in algorithms_with_periods:
                    algo_periods = signal_periods[signal_periods["algorithm"] == algo]
                    color = algo_colors.get(algo, "rgba(128, 128, 128, 0.3)")
                    algo_name = algo_names.get(algo, algo)

                    # Create one trace per algorithm with all its periods
                    # Use scatter with fill to create toggleable rectangles
                    x_coords = []
                    y_coords = []

                    for _, period in algo_periods.iterrows():
                        # Create a rectangle by adding 5 points (bottom-left, top-left, top-right, bottom-right, close)
                        x_start = period["timestamp_start"]
                        x_end = period["timestamp_end"]

                        # Add rectangle coordinates (closed path)
                        x_coords.extend([x_start, x_start, x_end, x_end, x_start, None])
                        y_coords.extend([y_lower, y_upper, y_upper, y_lower, y_lower, None])

                    # Add single trace for this algorithm with all its periods
                    fig.add_trace(go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        fill="toself",
                        fillcolor=color,
                        line=dict(width=0),
                        mode="lines",
                        name=f"Period: {algo_name}",
                        showlegend=True,
                        hoverinfo="skip",
                        legendgroup=algo,
                    ))

            # Update layout
            n_points = len(df_plot)
            sampling_info = f" (sampling: 1/{k})" if k > 1 else ""
            n_periods = len(signal_periods)
            period_info = f" | {n_periods} period(s) detected" if n_periods > 0 else " | No periods"

            fig.update_layout(
                title=f"Anomaly Period Visualization - {selected_signal}, {n_points} points{sampling_info}{period_info}",
                xaxis_title="Timestamp",
                yaxis_title="Value",
                hovermode="x unified",
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02,
                ),
                height=600,
            )

            return fig

        # Wire up callbacks
        process_btn.click(
            process_csv,
            inputs=[file_upload],
            outputs=[status_output, data_preview, processed_df_state, raw_signal_selector, detection_signal_selector, viz_signal_selector, available_signals_state],
        )

        raw_plot_btn.click(
            plot_raw_data,
            inputs=[processed_df_state, raw_signal_selector, raw_sample_rate],
            outputs=[raw_signal_plot],
        )

        run_btn.click(
            run_detection,
            inputs=[processed_df_state, detection_signal_selector, algos],
            outputs=[detection_output, detection_results_table, detection_results_state, viz_algorithm_filter, viz_signal_selector, viz_score_threshold],
        )

        viz_plot_btn.click(
            plot_anomaly_visualization,
            inputs=[processed_df_state, viz_signal_selector, detection_results_state, viz_algorithm_filter, viz_score_threshold, viz_sample_rate],
            outputs=[anomaly_plot],
        )

        period_detect_btn.click(
            detect_periods,
            inputs=[detection_results_state, period_k, period_n, period_threshold],
            outputs=[period_detection_output, period_results_table, period_results_state, period_viz_algorithms, period_viz_signal],
        )

        period_viz_btn.click(
            plot_period_visualization,
            inputs=[processed_df_state, period_viz_signal, period_results_state, period_viz_algorithms, period_viz_sample_rate],
            outputs=[period_viz_plot],
        )

    return demo


def main() -> None:
    """
    Run the Gradio app locally.
    """
    app = build_app()
    app.launch()


if __name__ == "__main__":
    main()
