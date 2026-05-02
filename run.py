"""
MLOps Batch Job — MetaStackerBandit Technical Assessment
Computes rolling-mean signal on OHLCV close prices.

Usage:
    python run.py --input data.csv --config config.yaml \
                  --output metrics.json --log-file run.log
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rolling-mean signal pipeline (MLOps assessment)"
    )
    parser.add_argument("--input",    required=True, help="Path to input OHLCV CSV")
    parser.add_argument("--config",   required=True, help="Path to config YAML")
    parser.add_argument("--output",   required=True, help="Path to output metrics JSON")
    parser.add_argument("--log-file", required=True, dest="log_file",
                        help="Path to log file")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_file: str) -> logging.Logger:
    """Configure root logger to write to both file and stdout."""
    logger = logging.getLogger("mlops_job")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"
    )

    # File handler
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler (info+)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REQUIRED_CONFIG_KEYS = {"seed", "window", "version"}


def load_config(config_path: str) -> dict:
    """Load and validate YAML config. Raises ValueError on bad config."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must be a mapping (key: value) at the top level")

    missing = REQUIRED_CONFIG_KEYS - cfg.keys()
    if missing:
        raise ValueError(f"Config missing required fields: {sorted(missing)}")

    # Type checks
    if not isinstance(cfg["seed"], int):
        raise ValueError(f"Config 'seed' must be an integer, got: {type(cfg['seed']).__name__}")
    if not isinstance(cfg["window"], int) or cfg["window"] < 1:
        raise ValueError(f"Config 'window' must be a positive integer, got: {cfg['window']}")
    if not isinstance(cfg["version"], str) or not cfg["version"].strip():
        raise ValueError("Config 'version' must be a non-empty string")

    return cfg


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_dataset(input_path: str) -> pd.DataFrame:
    """Load and validate CSV. Raises on any structural problem."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV: {exc}") from exc

    if df.empty:
        raise ValueError("Input CSV is empty (zero rows)")

    df.columns = [c.strip().lower() for c in df.columns]  # normalise column names

    if "close" not in df.columns:
        raise ValueError(
            f"Required column 'close' not found. Available columns: {list(df.columns)}"
        )

    if not pd.api.types.is_numeric_dtype(df["close"]):
        raise ValueError("Column 'close' must contain numeric values")

    if df["close"].isna().all():
        raise ValueError("Column 'close' contains only NaN values")

    return df


# ---------------------------------------------------------------------------
# Signal pipeline
# ---------------------------------------------------------------------------

def compute_rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """
    Compute rolling mean with min_periods=window so the first (window-1)
    rows produce NaN. Those rows are excluded from signal computation.
    """
    return series.rolling(window=window, min_periods=window).mean()


def compute_signal(close: pd.Series, rolling_mean: pd.Series) -> pd.Series:
    """
    Binary signal:
        1  if close > rolling_mean
        0  otherwise (including NaN rolling-mean rows)
    NaN positions are assigned 0 and excluded from signal_rate calculation.
    """
    signal = (close > rolling_mean).astype(float)
    signal[rolling_mean.isna()] = np.nan
    return signal


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def build_success_metrics(version: str, rows_processed: int,
                           signal_rate: float, latency_ms: float,
                           seed: int) -> dict:
    return {
        "version":        version,
        "rows_processed": rows_processed,
        "metric":         "signal_rate",
        "value":          round(signal_rate, 4),
        "latency_ms":     round(latency_ms),
        "seed":           seed,
        "status":         "success",
    }


def build_error_metrics(version: str, error_message: str) -> dict:
    return {
        "version":       version,
        "status":        "error",
        "error_message": error_message,
    }


def write_metrics(metrics: dict, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Entry point. Returns exit code (0 = success, 1 = failure)."""
    args = parse_args()

    # Bootstrap logger early so every branch gets logged
    logger = setup_logging(args.log_file)
    logger.info("=" * 60)
    logger.info("MLOps batch job STARTED")
    logger.info("=" * 60)
    logger.info("CLI args: input=%s  config=%s  output=%s  log=%s",
                args.input, args.config, args.output, args.log_file)

    t_start = time.perf_counter()

    # Partial config for error output (before full load)
    version = "unknown"

    try:
        # ── 1. Load & validate config ──────────────────────────────────────
        logger.info("Loading config: %s", args.config)
        cfg = load_config(args.config)
        version = cfg["version"]
        seed    = cfg["seed"]
        window  = cfg["window"]
        logger.info("Config OK  →  version=%s  seed=%d  window=%d",
                    version, seed, window)

        # Set global seed for reproducibility
        np.random.seed(seed)
        logger.info("NumPy random seed set to %d", seed)

        # ── 2. Load & validate dataset ─────────────────────────────────────
        logger.info("Loading dataset: %s", args.input)
        df = load_dataset(args.input)
        logger.info("Dataset loaded: %d rows, columns=%s",
                    len(df), list(df.columns))

        # ── 3. Rolling mean ────────────────────────────────────────────────
        logger.info("Computing rolling mean  (window=%d) on 'close'", window)
        df["rolling_mean"] = compute_rolling_mean(df["close"], window)
        n_warmup = window - 1
        logger.info("Rolling mean computed. First %d row(s) excluded (warm-up NaNs)", n_warmup)

        # ── 4. Signal ──────────────────────────────────────────────────────
        logger.info("Generating binary signal (1 if close > rolling_mean)")
        df["signal"] = compute_signal(df["close"], df["rolling_mean"])
        valid_mask   = df["signal"].notna()
        rows_valid   = valid_mask.sum()
        signal_rate  = float(df.loc[valid_mask, "signal"].mean())
        logger.info("Signal generated: %d valid rows, signal_rate=%.6f",
                    rows_valid, signal_rate)

        # ── 5. Metrics & timing ────────────────────────────────────────────
        t_end      = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        metrics = build_success_metrics(
            version=version,
            rows_processed=len(df),
            signal_rate=signal_rate,
            latency_ms=latency_ms,
            seed=seed,
        )

        write_metrics(metrics, args.output)
        logger.info("Metrics written to: %s", args.output)
        logger.info("Metrics summary: rows_processed=%d  signal_rate=%.4f  latency_ms=%.0f",
                    metrics["rows_processed"], metrics["value"], metrics["latency_ms"])

        # Print final JSON to stdout (Docker requirement)
        print(json.dumps(metrics, indent=2))

        logger.info("=" * 60)
        logger.info("MLOps batch job COMPLETED successfully")
        logger.info("=" * 60)
        return 0

    except (FileNotFoundError, ValueError) as exc:
        logger.error("Validation / IO error: %s", exc, exc_info=False)
        _write_error(version, str(exc), args.output, logger)
        return 1

    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        _write_error(version, f"Unexpected error: {exc}", args.output, logger)
        return 1


def _write_error(version: str, message: str, output_path: str,
                 logger: logging.Logger) -> None:
    metrics = build_error_metrics(version, message)
    try:
        write_metrics(metrics, output_path)
        logger.info("Error metrics written to: %s", output_path)
    except Exception as write_exc:
        logger.error("Could not write error metrics: %s", write_exc)
    logger.error("=" * 60)
    logger.error("MLOps batch job FAILED — %s", message)
    logger.error("=" * 60)
    # Also print error JSON to stdout
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    sys.exit(main())
