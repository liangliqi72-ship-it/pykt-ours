#!/usr/bin/env python
"""Aggregate pyKT printed result lines from official DUGP experiment logs.

Usage:
  python my_scripts/aggregate_dugp_results.py --log_dir saved_model_dugp_official --out my_logs/dugp_summary.csv

The script scans config.json and stdout-like text files under the log dir when
available. If your training output is redirected, put *.log files inside the
save dir; otherwise config parsing still records the planned runs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

RESULT_RE = re.compile(
    r"^(?P<fold>\d+)\t(?P<modelname>[^\t]+)\t(?P<embtype>[^\t]+)\t"
    r"(?P<testauc>-?\d+(?:\.\d+)?)\t(?P<testacc>-?\d+(?:\.\d+)?)\t"
    r"(?P<window_testauc>-?\d+(?:\.\d+)?)\t(?P<window_testacc>-?\d+(?:\.\d+)?)\t"
    r"(?P<validauc>-?\d+(?:\.\d+)?)\t(?P<validacc>-?\d+(?:\.\d+)?)\t(?P<best_epoch>-?\d+)"
)


def parse_result_lines(root: Path):
    rows = []
    for log_file in root.rglob("*.log"):
        for line in log_file.read_text(errors="ignore").splitlines():
            m = RESULT_RE.match(line.strip())
            if m:
                row = m.groupdict()
                row["source"] = str(log_file)
                rows.append(row)
    return rows


def parse_configs(root: Path):
    rows = []
    for cfg_path in root.rglob("config.json"):
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            continue
        params = cfg.get("params", {})
        rows.append({
            "fold": params.get("fold"),
            "modelname": params.get("model_name"),
            "embtype": params.get("emb_type"),
            "dataset": params.get("dataset_name"),
            "seed": params.get("seed"),
            "dugp_mode": params.get("dugp_mode", "baseline"),
            "config_path": str(cfg_path),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--out", type=str, default="my_logs/dugp_summary.csv")
    args = parser.parse_args()

    root = Path(args.log_dir)
    result_rows = parse_result_lines(root)
    cfg_rows = parse_configs(root)

    df_results = pd.DataFrame(result_rows)
    df_cfg = pd.DataFrame(cfg_rows)
    if not df_results.empty:
        for col in ["fold", "testauc", "testacc", "window_testauc", "window_testacc", "validauc", "validacc", "best_epoch"]:
            df_results[col] = pd.to_numeric(df_results[col], errors="coerce")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not df_results.empty:
        df_results.to_csv(out_path, index=False)
        summary = df_results.groupby(["modelname", "embtype"], dropna=False).agg(
            testauc_mean=("testauc", "mean"), testauc_std=("testauc", "std"),
            testacc_mean=("testacc", "mean"), testacc_std=("testacc", "std"),
            validauc_mean=("validauc", "mean"), validauc_std=("validauc", "std"),
            runs=("testauc", "count"),
        ).reset_index()
        summary.to_csv(out_path.with_name(out_path.stem + "_mean_std.csv"), index=False)
        print(summary)
    else:
        cfg_out = out_path.with_name(out_path.stem + "_configs_only.csv")
        df_cfg.to_csv(cfg_out, index=False)
        print(f"No *.log result lines found. Wrote config inventory to {cfg_out}")


if __name__ == "__main__":
    main()
