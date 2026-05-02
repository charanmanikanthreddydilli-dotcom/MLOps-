# MLOps Batch Signal Pipeline — MetaStackerBandit T0

A minimal, production-style MLOps batch job that computes a rolling-mean
binary signal on OHLCV close prices. Designed for **reproducibility**,
**observability**, and **deployment readiness**.

---

## Project structure

```
.
├── run.py           # Main pipeline script
├── config.yaml      # Job configuration (seed, window, version)
├── data.csv         # 10,000-row OHLCV dataset
├── requirements.txt # Python dependencies
├── Dockerfile       # Container definition
├── metrics.json     # Sample output from a successful run
├── run.log          # Sample log from a successful run
└── README.md        # This file
```

---

## How it works

| Step | Detail |
|------|--------|
| **1 — Config** | Loads `config.yaml`; validates `seed`, `window`, `version` |
| **2 — Dataset** | Reads CSV; validates existence, non-emptiness, `close` column |
| **3 — Rolling mean** | `pandas.rolling(window, min_periods=window)` on `close`; first `window-1` rows are NaN and excluded |
| **4 — Signal** | `signal = 1 if close > rolling_mean else 0` (NaN rows excluded from rate) |
| **5 — Metrics** | Writes `metrics.json` with `rows_processed`, `signal_rate`, `latency_ms` |
| **6 — Logs** | Structured log written to `run.log` and echoed to stdout |

---

## Local run

### Prerequisites

- Python 3.9+
- pip

### Setup

```bash
# 1. Clone / download the repo, then enter it
cd mlops-task

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the pipeline
python run.py \
  --input    data.csv \
  --config   config.yaml \
  --output   metrics.json \
  --log-file run.log
```

On success the final `metrics.json` is printed to stdout and written to disk.

### Verify outputs

```bash
cat metrics.json   # structured metrics
cat run.log        # detailed execution log
```

---

## Docker build & run

```bash
# Build the image
docker build -t mlops-task .

# Run the container (prints metrics JSON to stdout, exits 0 on success)
docker run --rm mlops-task
```

The container:
- Bundles `data.csv` and `config.yaml` at build time
- Runs the pipeline with no hard-coded paths
- Writes `metrics.json` and `run.log` inside the container
- Prints the final metrics JSON to stdout
- Exits **0** on success, **non-zero** on failure

To retrieve output files from the container:

```bash
# Mount a host directory to /app/out and write outputs there
docker run --rm \
  -v "$(pwd)/output:/app/out" \
  mlops-task \
  python run.py \
    --input    data.csv \
    --config   config.yaml \
    --output   out/metrics.json \
    --log-file out/run.log
```

---

## Example `metrics.json`

```json
{
  "version": "v1",
  "rows_processed": 10000,
  "metric": "signal_rate",
  "value": 0.5036,
  "latency_ms": 34,
  "seed": 42,
  "status": "success"
}
```

### Error example

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Required column 'close' not found. Available columns: ['open', 'high']"
}
```

---

## Config reference (`config.yaml`)

| Key | Type | Description |
|-----|------|-------------|
| `seed` | int | NumPy random seed for reproducibility |
| `window` | int | Rolling mean window size (rows) |
| `version` | str | Pipeline version tag written to metrics output |

---

## Reproducibility guarantee

Running the pipeline twice on the same `data.csv` + `config.yaml` always
produces **identical** `metrics.json` values. The `seed` field controls
`numpy.random.seed()`, and the signal logic is purely deterministic
(`pandas.rolling` + comparison).

---

## Error handling

The script handles and logs all of the following, writing an error
`metrics.json` in every case:

- Missing input CSV or config YAML
- Unparseable YAML or CSV
- Empty CSV
- Missing required config keys (`seed`, `window`, `version`)
- Wrong types (e.g. non-integer seed)
- Missing `close` column
- Any unexpected runtime exception
