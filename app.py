
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
            "Many scientific or industrial applications rely on sensor data. Hence, the importance to detect sensor anomalies before to build datasets for applications. "
            "This application allows to test several approaches for such detections. "
            "Upload a CSV file with sensor time-series data. "
            "The app supports three formats: **long format**, **semi-long format**, and **wide format**."
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

        # Store the processed dataframe in state
        processed_df_state = gr.State(value=None)
        available_signals_state = gr.State(value=[])
        detection_results_state = gr.State(value=None)

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

    return demo


def main() -> None:
    """
    Run the Gradio app locally.
    """
    app = build_app()
    app.launch()


if __name__ == "__main__":
    main()
