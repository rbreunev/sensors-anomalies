# Sensors Anomalies

End-to-end demo for sensor anomaly/fault detection on time-series data:
- **Dataset transformers** for normalizing wide-format CSVs to canonical long format
- **Algorithm registry** for comparing multiple detection approaches
- **Interactive Gradio app** deployable on Hugging Face Spaces

## 🎯 Key Features

- **Runtime CSV upload** - No local data storage, works on HF Spaces without credentials
- **Canonical long format** - All pipelines operate on standardized `(series_id, timestamp, signal, value, [label])` schema
- **Flexible input** - Auto-detect and transform wide-format CSVs or accept pre-formatted long-format data
- **Production-grade tooling** - Full type checking, testing, linting with modern Python tools

## 🚀 Quick Start

### Local development (uv)
```bash
# Setup environment
uv venv
uv sync --extra dev
uv pip install -e .

# Run tests
uv run pytest
uv run tox -e lint,type,pylint,tests

# Launch Gradio app
uv run python app.py
```

### Using the App

1. **Upload a CSV** - Either:
   - Long format: Already has `series_id`, `timestamp`, `signal`, `value` columns
   - Wide format: Has `timestamp` and multiple sensor columns (e.g., `sensor_00`, `sensor_01`, ...)

2. **Select Dataset Type**:
   - `auto` - Automatically detect format and transform if needed
   - `sensor_fault` - Use Kaggle Sensor Fault Detection transformer
   - Future transformers will be added here

3. **Process CSV** - Validates and normalizes to long format

4. **Run Detection** - Select algorithms and detect anomalies (coming in Step 3)

## 📊 Data Format

### Canonical Long Format (Required)

All datasets must be normalized to this schema:

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `series_id` | str | ✅ | Time series identifier (e.g., sensor unit, run) |
| `timestamp` | datetime | ✅ | Observation timestamp |
| `signal` | str | ✅ | Signal/sensor name |
| `value` | float | ✅ | Numeric measurement |
| `label` | int | ❌ | Anomaly label (0=normal, 1=fault) |

**Example:**
```csv
series_id,timestamp,signal,value,label
s1,2024-01-01 00:00:00,sensor_00,1.23,0
s1,2024-01-01 00:00:00,sensor_01,2.45,0
s1,2024-01-01 00:01:00,sensor_00,1.25,0
s1,2024-01-01 00:01:00,sensor_01,2.50,1
```

### Wide Format (Auto-Transformed)

If you upload a wide-format CSV, it will be automatically transformed:

**Input (Wide):**
```csv
timestamp,sensor_00,sensor_01,target
2024-01-01 00:00:00,1.23,2.45,0
2024-01-01 00:01:00,1.25,2.50,1
```

**Output (Long):**
```csv
series_id,timestamp,signal,value,label
default_series,2024-01-01 00:00:00,sensor_00,1.23,0
default_series,2024-01-01 00:00:00,sensor_01,2.45,0
default_series,2024-01-01 00:01:00,sensor_00,1.25,1
default_series,2024-01-01 00:01:00,sensor_01,2.50,1
```

## 📦 Supported Datasets

### Sensor Fault Detection (Kaggle)

**Source:** https://www.kaggle.com/datasets/arashnic/sensor-fault-detection-data

**How to use:**
1. Download `sensor.csv` from Kaggle (requires Kaggle account)
2. Upload to the Gradio app
3. Select `sensor_fault` transformer or use `auto`

**Format:** Wide format with ~50 sensor columns and binary fault labels

**License:** Check Kaggle dataset page for terms

### Gas Sensor Array Drift (UCI)

**Source:** https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset

**Status:** Coming in future update

## 🏗️ Project Structure

```
sensors-anomalies/
├── src/sensors_anomalies/
│   ├── types.py                    # Core data types and validation
│   ├── constants.py                # Project constants
│   ├── datasets/
│   │   ├── registry.py             # Transformer registry
│   │   └── sensor_fault.py         # Kaggle sensor fault transformer
│   ├── algorithms/
│   │   └── registry.py             # Algorithm registry (Step 3)
│   └── utils/
│       ├── time.py                 # Time utilities
│       └── csv_utils.py            # CSV processing and validation
├── tests/
│   ├── test_imports.py
│   ├── test_types.py
│   ├── test_csv_utils.py           # CSV utilities tests
│   └── test_transformers.py        # Transformer tests
├── app.py                          # Gradio application
├── pyproject.toml                  # Dependencies and tool config
└── README.md
```

## 🔧 Development

### Code Quality

All code follows strict standards:
- ✅ Full type annotations (`mypy --strict`)
- ✅ NumPy-style docstrings
- ✅ Ruff formatting and linting
- ✅ Pylint checks
- ✅ 100% test coverage for core modules

### Adding a New Transformer

1. Create `src/sensors_anomalies/datasets/your_dataset.py`:

```python
from sensors_anomalies.datasets.registry import register_transformer
from sensors_anomalies.types import DatasetSpec

def transform_your_dataset(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Transform your dataset to long format."""
    # Implementation here
    return df_long

def register_your_dataset_transformer() -> None:
    register_transformer(
        spec=DatasetSpec("your_id", "Your Name", "https://source.url"),
        transformer=transform_your_dataset,
    )
```

2. Register in `datasets/registry.py`:

```python
def _register_defaults() -> None:
    from sensors_anomalies.datasets.your_dataset import register_your_dataset_transformer
    register_your_dataset_transformer()
```

3. Add tests in `tests/test_transformers.py`

## 📝 Citations

### Sensor Fault Detection Dataset
If using this dataset, please cite according to the [Kaggle dataset page](https://www.kaggle.com/datasets/arashnic/sensor-fault-detection-data).

## 🎓 Project Goals

This project demonstrates:
- **Dataset-centric design** - Data quality, normalization, validation
- **Scalable patterns** - Long format for distributed processing
- **Engineering rigor** - Types, docs, tests, linting
- **Practical ML** - Comparing multiple anomaly detection approaches
- **Deployment-ready** - Gradio app for HF Spaces

## 📄 License

See [LICENSE](LICENSE) file.

**Note:** Dataset licenses vary - check individual dataset sources for usage terms.
