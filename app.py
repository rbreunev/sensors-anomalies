
"""
Gradio entrypoint for the Sensors Anomalies demo.

The first commit provides a project skeleton that is ready for Hugging Face Spaces.
Real dataset loaders and detection algorithms are added in subsequent steps.
"""

from __future__ import annotations

from typing import List

import gradio as gr

from sensors_anomalies.algorithms.registry import list_algorithms
from sensors_anomalies.datasets.registry import list_datasets


def build_app() -> gr.Blocks:
    """
    Build the Gradio application.

    Returns
    -------
    gradio.Blocks
        The Gradio Blocks app.
    """
    datasets: List[str] = list_datasets()
    algorithms: List[str] = list_algorithms()

    with gr.Blocks() as demo:
        gr.Markdown("# Sensors Anomalies Demo")
        gr.Markdown("Project skeleton – datasets and algorithms will be added next.")

        dataset = gr.Dropdown(
            choices=datasets,
            label="Dataset",
            value=datasets[0],
        )

        algos = gr.CheckboxGroup(
            choices=algorithms,
            label="Algorithms",
            value=algorithms,
        )

        output = gr.Textbox(label="Status", lines=5)
        run_btn = gr.Button("Run")

        def _run(ds: str, alg: List[str]) -> str:
            """
            Placeholder callback for the initial project skeleton.

            Parameters
            ----------
            ds : str
                Selected dataset id.
            alg : list[str]
                Selected algorithm ids.

            Returns
            -------
            str
                Status message.
            """
            return f"Dataset: {ds}\nAlgorithms: {', '.join(alg)}\n\nImplementation coming next."

        run_btn.click(_run, inputs=[dataset, algos], outputs=output)

    return demo


def main() -> None:
    """
    Run the Gradio app locally.
    """
    app = build_app()
    app.launch()


if __name__ == "__main__":
    main()
