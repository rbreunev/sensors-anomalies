---
title: sensors-anomalies
emoji: 🧪
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.44.1"
python_version: "3.11"
app_file: app.py
pinned: false
---

# Sensors Anomalies

This repository contains tooling for sensor anomaly detection.
The whole can be deployed as a gradio application, representing the full workflow :

- Data loading (cna handle different kind of CSV)
- Data exploration
- Point based unsupervised anomaly detection algorithms test and set up
- Post filtering rules test and set up to define period of anomalies

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

### Deployment (Hugging Face Spaces)

For deploying to Hugging Face Spaces, a `requirements.txt` file is included. To regenerate it from the project dependencies:

```bash
uv pip compile pyproject.toml -o requirements.txt
```

This command creates a pinned requirements file with all transitive dependencies resolved, ensuring reproducible deployments.

### Using the App

**Main Workflow:**

1. **Upload CSV** - Upload your sensor data file
2. **Process CSV** - App automatically detects format and normalizes to long format
3. **Run Detection** - Apply anomaly detection algorithms to identify suspicious points
4. **Visualize Results** - Explore point-by-point anomalies on interactive plots
5. **Detect Periods** *(optional)* - Identify sustained anomaly periods using sliding window rules

---

#### Upload CSV

The app supports three CSV formats and automatically normalizes them to the canonical long format:

**Supported Formats:**

1. **Long format:** Already has `series_id`, `timestamp`, `signal`, `value` columns
2. **Semi-long format:** Has `timestamp`, signal/sensor identifier, `value` columns (but no `series_id`)
3. **Wide format:** Has `timestamp` and multiple sensor columns (e.g., `sensor_00`, `sensor_01`, ...)

**Acceptable Column Names:**

- **Timestamp:** `timestamp`, `Timestamp`, `time`, `Time`, `datetime`, `Datetime`, `date`, `Date`, `Horodatage`, `capture_date`, `measurement_date`
- **Signal/Sensor (semi-long):** `signal`, `Signal`, `sensor`, `Sensor`, `SensorId`, `sensor_id`, `SensorID`, `sensor_name`, `SensorName`
- **Value (semi-long):** `value`, `Value`, `measurement`, `Measurement`, `reading`, `Reading`
- **Label (optional):** `label`, `Label`, `target`, `Target`, `fault`, `Fault`, `anomaly`, `Anomaly`
- **Sensors (wide):** Any numeric columns not matching the above

The app automatically detects your format and validates/transforms it to the canonical long format for processing.

---

#### Run Detection

Select one or more unsupervised anomaly detection algorithms to identify suspicious data points. The app will:

1. Apply the selected algorithms to your time series data
2. Generate anomaly scores for each timestamp
3. Display results on interactive plots showing point-by-point anomalies

Use the visualization to explore score distributions and identify patterns in the detected anomalies.

---

#### Detect Periods

In addition to point-by-point detection, you can identify **sustained anomaly periods** to consolidate isolated anomalies into meaningful time ranges. This is particularly useful since sensor faults can last long and impact data quality over extended periods.

**How It Works:**

The period detection algorithm uses a forward-looking sliding window approach:

1. For each timestamp, examine the next **N** consecutive points
2. If **K** or more of those N points exceed the **score threshold**, mark that timestamp as part of an anomaly period
3. All consecutive marked timestamps are merged into continuous periods

**Parameters:**

- **K**: Minimum number of anomalous points required in the window (e.g., 3)
- **N**: Size of the sliding window to check (e.g., 5)
- **Score Threshold**: Minimum anomaly score for a point to be considered anomalous (e.g., 2.0)

**Example:** With K=3, N=5, threshold=2.0:
- At timestamp t=100, look at next 5 points: [t100, t101, t102, t103, t104]
- If at least 3 of these 5 have anomaly score ≥ 2.0, mark t100 as part of a period
- Move to t=101 and repeat
- All consecutive marked timestamps form a period with start/end times
- Each period shows: start time, end time, number of points, mean score, max score

**Typical Use Case:**
```
Point detection: Identifies 47 individual anomalous timestamps with high scores
Period detection (K=3, N=5, threshold=2.0): Consolidates these into 3 meaningful periods:
  - Period 1: 2024-01-01 10:00 to 2024-01-01 13:00 (15 points, mean score: 3.2)
  - Period 2: 2024-01-01 18:00 to 2024-01-01 19:00 (8 points, mean score: 2.8)
  - Period 3: 2024-01-02 09:00 to 2024-01-02 11:00 (24 points, mean score: 4.1)
```

**Parameter Selection Tips:**

Use the point-by-point detection visualization to explore score distributions before setting period parameters:
- **High threshold + low K** = Only detect very strong, concentrated anomalies
- **Low threshold + high K** = Detect weaker but more sustained anomalies
- **Adjust N** based on your temporal resolution (larger N for high-frequency data)

**Results:**

The app displays:
- Number of detected periods
- Period statistics (longest/shortest, mean duration)
- Visualization with periods shown as colored horizontal bands on the time series plot

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



## 🔧 Development

### Code Quality

All code follows strict standards:
- ✅ Full type annotations (`mypy --strict`)
- ✅ NumPy-style docstrings
- ✅ Ruff formatting and linting
- ✅ Pylint checks
- ✅ 100% test coverage for core modules

## 📄 License

See [LICENSE](LICENSE) file.

**Note:** Dataset licenses vary - check individual dataset sources for usage terms.
