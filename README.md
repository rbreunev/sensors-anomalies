# Sensors Anomalies

End-to-end demo for sensor anomaly/fault detection:
- dataset registry
- algorithm registry
- interactive Gradio app (Hugging Face Spaces-ready)

## Local development (uv)
```bash
uv venv venv
uv sync --extra dev
uv run pytest
uv run tox -e lint,type,pylint,tests
uv run python app.py
