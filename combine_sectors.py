#!/usr/bin/env python3
"""
combine_sectors.py
------------------
Stack per-sector metrics into one table.

Example
-------
    python combine_sectors.py --in tables/metrics --out tables/all_sectors_metrics.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description="Combine Sector_*_metrics tables")
    p.add_argument("--in", dest="indir", default="tables/metrics")
    p.add_argument("--out", default="tables/all_sectors_metrics.parquet")
    args = p.parse_args()

    indir = Path(args.indir)
    files = sorted(indir.glob("Sector_*_metrics.parquet"))
    if not files:
        files = sorted(indir.glob("Sector_*_metrics.csv"))
    if not files:
        raise SystemExit(f"No Sector_*_metrics.parquet/csv in {indir}")

    frames = []
    for f in files:
        df = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
        df["source_table"] = f.name
        frames.append(df)
        print(f"  {f.name}: {len(df):,} rows")

    out = pd.concat(frames, ignore_index=True)
    if "status" in out.columns:
        print("status counts:\n", out["status"].value_counts(dropna=False).to_string())

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(outp, index=False)
    out.to_csv(outp.with_suffix(".csv"), index=False)
    print(f"Wrote {len(out):,} rows → {outp}")


if __name__ == "__main__":
    main()