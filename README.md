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

1. **Upload a CSV** - The app supports three formats:
   - **Long format:** Already has `series_id`, `timestamp`, `signal`, `value` columns
   - **Semi-long format:** Has `timestamp`, signal/sensor identifier, `value` columns (but no `series_id`)
   - **Wide format:** Has `timestamp` and multiple sensor columns (e.g., `sensor_00`, `sensor_01`, ...)

2. **Process CSV** - Automatically detects format and validates/transforms to long format

3. **Run Detection** - Select algorithms and detect anomalies

4. **Visualize Results** - View point-by-point anomalies on interactive plots

5. **Detect Periods** - Identify sustained anomaly periods (optional)

### Period-Based Anomaly Detection

In addition to point-by-point detection, the app supports **period-based anomaly detection** to identify sustained anomalies:

**Workflow:**

1. **Run Point Detection** - First, run the standard anomaly detection to get scores for each timestamp
2. **Configure Period Parameters**:
   - **K**: Minimum number of anomalous points required in the window (e.g., 3)
   - **N**: Size of the sliding window to check (e.g., 5)
   - **Score Threshold**: Minimum anomaly score for a point to be considered anomalous (e.g., 2.0)
3. **Detect Periods** - The algorithm uses a sliding window approach to identify continuous anomaly periods
4. **View Statistics** - See number of periods, longest/shortest periods, mean duration
5. **Visualize Periods** - View periods as colored horizontal bands on the time series plot

**How Period Detection Works:**

The period detection algorithm uses a forward-looking window approach. For each timestamp in the data, it looks at the next N consecutive points in the timeline. If K or more of those N points exceed the score threshold, that timestamp is marked as part of an anomaly period. All consecutive marked timestamps are then merged into continuous periods.

**Example:** With K=3, N=5, threshold=2.0:
- At timestamp t=100, look at next 5 points: [t100, t101, t102, t103, t104]
- If at least 3 of these 5 have anomaly score ≥ 2.0, mark t100 as part of a period
- Move to t=101 and repeat
- All consecutive marked timestamps form a period with timestamp_start to timestamp_end
- Each period shows: start time, end time, number of points, mean score, max score

**Typical Use Case:**
```
Point detection: Identifies 47 individual anomalous timestamps with high scores
Period detection (K=3, N=5, threshold=2.0): Consolidates these into 3 meaningful periods:
  - Period 1: 2024-01-01 10:00 to 2024-01-01 13:00 (15 points, mean score: 3.2)
  - Period 2: 2024-01-01 18:00 to 2024-01-01 19:00 (8 points, mean score: 2.8)
  - Period 3: 2024-01-02 09:00 to 2024-01-02 11:00 (24 points, mean score: 4.1)
```

**Why Use Period Detection:**

Period detection is useful for identifying sustained anomalies rather than isolated spikes, which is important for many sensor monitoring applications:
- **Equipment degradation** - Detect when sensors show prolonged abnormal behavior
- **Process drift** - Identify when industrial processes deviate from normal operation
- **Fault diagnosis** - Distinguish between transient noise and persistent faults
- **Maintenance scheduling** - Prioritize interventions based on sustained anomalies

**Parameter Selection Tips:**

Use the point-by-point detection visualization to explore score distributions before setting period parameters:
- High threshold + low K = Only detect very strong, concentrated anomalies
- Low threshold + high K = Detect weaker but more sustained anomalies
- Adjust N based on your temporal resolution (larger N for high-frequency data)

**Acceptable Column Names:**
- **Timestamp:** `timestamp`, `Timestamp`, `time`, `Time`, `datetime`, `Datetime`, `date`, `Date`, `Horodatage`, `capture_date`, `measurement_date`
- **Signal/Sensor (semi-long):** `signal`, `Signal`, `sensor`, `Sensor`, `SensorId`, `sensor_id`, `SensorID`, `sensor_name`, `SensorName`
- **Value (semi-long):** `value`, `Value`, `measurement`, `Measurement`, `reading`, `Reading`
- **Label (optional):** `label`, `Label`, `target`, `Target`, `fault`, `Fault`, `anomaly`, `Anomaly`
- **Sensors (wide):** Any numeric columns not matching the above

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

### Semi-Long Format (Auto-Normalized)

If you upload a semi-long format CSV (has timestamp, signal, and value columns but no series_id), it will be normalized:

**Input (Semi-Long):**
```csv
timestamp,SensorId,Value,target
2024-01-01 00:00:00,sensor_00,1.23,0
2024-01-01 00:00:00,sensor_01,2.45,0
2024-01-01 00:01:00,sensor_00,1.25,1
2024-01-01 00:01:00,sensor_01,2.50,1
```

**Output (Long):**
```csv
series_id,timestamp,signal,value,label
default_series,2024-01-01 00:00:00,sensor_00,1.23,0
default_series,2024-01-01 00:00:00,sensor_01,2.45,0
default_series,2024-01-01 00:01:00,sensor_00,1.25,1
default_series,2024-01-01 00:01:00,sensor_01,2.50,1
```

### Wide Format (Auto-Transformed)

If you upload a wide-format CSV (has timestamp and multiple sensor columns), it will be melted to long format:

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
│   │   └── registry.py             # Transformer registry + transformers
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

Add the transformer function directly to `src/sensors_anomalies/datasets/registry.py`:

```python
def transform_your_dataset(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Transform your dataset to long format."""
    # Implementation here
    return df_long
```

Then register it in `_register_defaults()`:

```python
def _register_defaults() -> None:
    """Register default transformers."""
    # Existing transformers...

    # Add your new transformer
    register_transformer(
        spec=DatasetSpec("your_id", "Your Name", "https://source.url"),
        transformer=transform_your_dataset,
    )
```

Add tests in `tests/test_transformers.py` to validate the transformation.

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
