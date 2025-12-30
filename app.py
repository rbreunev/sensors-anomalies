
"""
Gradio entrypoint for the Sensors Anomalies demo.

Users upload a CSV file at runtime, which is either:
1. Already in long format (series_id, timestamp, signal, value, [label])
2. In wide format and can be transformed using a registered transformer
3. In wide format and will be auto-transformed using generic wide-to-long conversion

This design works both locally and on Hugging Face Spaces without requiring
dataset files in the repository or Kaggle credentials.
"""

from __future__ import annotations

from typing import Any

import gradio as gr
import pandas as pd

from sensors_anomalies.algorithms.registry import list_algorithms
from sensors_anomalies.datasets.registry import list_transformers
from sensors_anomalies.utils.csv_utils import (
    is_long_format,
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
    transformers = ["auto"] + list_transformers()
    algorithms = list_algorithms()

    with gr.Blocks(title="Sensors Anomalies Demo") as demo:
        gr.Markdown("# 🔍 Sensors Anomalies Detection")
        gr.Markdown(
            "Upload a CSV file with sensor time-series data. "
            "The CSV can be in **long format** (ready to use) or **wide format** (will be transformed)."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Data Upload")

                file_upload = gr.File(
                    label="Upload CSV",
                    file_types=[".csv"],
                    type="binary",
                )

                dataset_type = gr.Dropdown(
                    choices=transformers,
                    label="Dataset Type",
                    value="auto",
                    info="Select 'auto' for automatic detection or choose a specific transformer",
                )

                process_btn = gr.Button("📊 Process CSV", variant="primary")

                gr.Markdown("### 🎯 Detection")

                algos = gr.CheckboxGroup(
                    choices=algorithms,
                    label="Algorithms",
                    value=algorithms[:1] if algorithms else [],
                )

                run_btn = gr.Button("🚀 Run Detection", variant="secondary")

            with gr.Column(scale=2):
                gr.Markdown("### 📈 Results")

                status_output = gr.Textbox(
                    label="Status",
                    lines=5,
                    placeholder="Upload a CSV to get started...",
                )

                data_preview = gr.Dataframe(
                    label="Data Preview (Long Format)",
                    interactive=False,
                    wrap=True,
                )

        # Store the processed dataframe in state
        processed_df_state = gr.State(value=None)

        def process_csv(
            file_bytes: bytes | None,
            dataset_type: str,
        ) -> tuple[str, pd.DataFrame | None, pd.DataFrame | None]:
            """
            Process uploaded CSV file.

            Parameters
            ----------
            file_bytes : bytes or None
                Uploaded file bytes.
            dataset_type : str
                Dataset type for transformation.

            Returns
            -------
            tuple[str, pd.DataFrame | None, pd.DataFrame | None]
                Status message, preview dataframe, and full dataframe for state.
            """
            if file_bytes is None:
                return "❌ Please upload a CSV file", None, None

            try:
                # Read CSV
                df = read_uploaded_csv(file_bytes)
                status_lines = [f"✅ CSV loaded: {len(df)} rows, {len(df.columns)} columns"]

                # Check if already in long format
                if is_long_format(df):
                    status_lines.append("✅ Data is already in long format")
                    df_long = validate_and_normalize_long_format(df)
                else:
                    status_lines.append("⚙️ Data is in wide format, transforming...")

                    if dataset_type == "auto":
                        # Generic wide-to-long transformation
                        df_long = transform_wide_to_long(df)
                        status_lines.append("✅ Auto-transformed to long format")
                    else:
                        # Use specific transformer
                        from sensors_anomalies.datasets.registry import apply_transformer
                        df_long = apply_transformer(dataset_type, df)
                        status_lines.append(f"✅ Transformed using '{dataset_type}' transformer")

                # Get statistics
                n_series = df_long["series_id"].nunique()
                n_signals = df_long["signal"].nunique()
                has_labels = "label" in df_long.columns
                time_range = df_long["timestamp"].agg(["min", "max"])

                status_lines.extend([
                    f"\n📊 Dataset Statistics:",
                    f"  • Series: {n_series}",
                    f"  • Signals: {n_signals}",
                    f"  • Total readings: {len(df_long)}",
                    f"  • Time range: {time_range['min']} to {time_range['max']}",
                    f"  • Has labels: {'Yes' if has_labels else 'No'}",
                ])

                status = "\n".join(status_lines)
                preview = df_long.head(100)  # Show first 100 rows

                return status, preview, df_long

            except Exception as e:
                return f"❌ Error processing CSV: {str(e)}", None, None

        def run_detection(
            df_long: pd.DataFrame | None,
            selected_algos: list[str],
        ) -> str:
            """
            Run anomaly detection algorithms.

            Parameters
            ----------
            df_long : pd.DataFrame or None
                Processed long-format dataframe.
            selected_algos : list[str]
                Selected algorithm ids.

            Returns
            -------
            str
                Detection results message.
            """
            if df_long is None:
                return "❌ Please process a CSV file first"

            if not selected_algos:
                return "❌ Please select at least one algorithm"

            # Placeholder - actual algorithm execution will be implemented in Step 3
            results = [
                "🎯 Detection Results:",
                f"\nDataset: {len(df_long)} rows",
                f"Algorithms: {', '.join(selected_algos)}",
                "\n⚙️ Algorithm execution not yet implemented.",
                "This will be added in Step 3 (Anomaly Detection Algorithms).",
            ]

            return "\n".join(results)

        # Wire up callbacks
        process_btn.click(
            process_csv,
            inputs=[file_upload, dataset_type],
            outputs=[status_output, data_preview, processed_df_state],
        )

        run_btn.click(
            run_detection,
            inputs=[processed_df_state, algos],
            outputs=[status_output],
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
