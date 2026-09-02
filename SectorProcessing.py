#!/usr/bin/env python3
"""
process_local.py
----------------
Process locally stored TESS 2-minute SPOC light curves.

Reads sector folders of *_lc.fits (default: D:\\sector1, D:\\sector2, ...)
Writes one metrics table per sector for later classification.

Does not download data. Does not clip outliers (that removes real eclipses).

Example
-------
    python process_local.py --sectors 1-22 --fits-root D:\\ --out tables/metrics
    python process_local.py --sectors 23,24,25 --fits-root D:\\ --force
"""
from __future__ import annotations

import argparse
import gc
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from lightkurve import read

MIN_POINTS = 1000
MIN_PERIOD = 0.05
MAX_PERIOD = 8.0
PERIOD_CEILING = 7.95
TIC_RE = re.compile(r"-s(\d+)-(\d+)-", re.IGNORECASE)


def parse_sectors(text: str) -> list[int]:
    """'1-22' or '1,2,5-7' -> [1, 2, 5, 6, 7]."""
    out: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def sector_dir(root: Path, sector: int) -> Path:
    for name in (
        f"sector{sector}",
        f"sector{sector:02d}",
        f"Sector{sector}",
        f"Sector{sector:02d}",
    ):
        p = root / name
        if p.exists():
            return p
    return root / f"sector{sector}"


def find_lc_files(root: Path, sector: int, log: logging.Logger) -> list[Path]:
    folder = sector_dir(root, sector)
    if not folder.exists():
        log.warning("Missing folder: %s", folder)
        return []
    return [p for p in folder.rglob("*_lc.fits") if p.is_file()]


def tic_from_name(path: Path):
    m = TIC_RE.search(path.name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def as_float(x):
    if hasattr(x, "value"):
        x = x.value
    return np.asarray(x, dtype=float)


def fold_phase(time, t0, period):
    return ((time - t0) / period + 0.5) % 1.0 - 0.5


def sine_r2(phase, flux) -> float:
    phi = (phase + 0.5) % 1.0
    y = flux.astype(float)
    if len(y) < 50 or np.nanstd(y) == 0:
        return 0.0
    A = np.column_stack(
        [np.ones(len(phi)), np.sin(2 * np.pi * phi), np.cos(2 * np.pi * phi)]
    )
    good = np.isfinite(y)
    if good.sum() < 50:
        return 0.0
    coef, *_ = np.linalg.lstsq(A[good], y[good], rcond=None)
    pred = A[good] @ coef
    ss_res = np.sum((y[good] - pred) ** 2)
    ss_tot = np.sum((y[good] - np.mean(y[good])) ** 2)
    if ss_tot <= 0:
        return 0.0
    return float(max(0.0, min(1.0, 1.0 - ss_res / ss_tot)))


def robust_level(vals, pct=5) -> float:
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan
    if len(vals) < 8:
        return float(np.nanmedian(vals))
    return float(np.nanpercentile(vals, pct))


def process_file(path: Path, sector: int) -> dict | None:
    _sec_in_name, tic = tic_from_name(path)
    if tic is None:
        return None

    lc = read(str(path))
    lc = lc.remove_nans()
    # Do not remove_outliers(sigma=5): that clips real eclipses.

    time = as_float(lc.time)
    flux = as_float(lc.flux)
    good = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[good], flux[good]
    n = len(flux)
    if n < MIN_POINTS:
        return {
            "tic_id": tic,
            "sector": sector,
            "status": "too_short",
            "num_points": n,
            "source_file": path.name,
        }

    mean_flux = float(np.nanmean(flux))
    if mean_flux <= 0:
        return {
            "tic_id": tic,
            "sector": sector,
            "status": "neg_flux",
            "mean_flux": mean_flux,
            "num_points": n,
            "source_file": path.name,
        }

    pg = lc.to_periodogram(
        method="lombscargle",
        minimum_period=MIN_PERIOD,
        maximum_period=MAX_PERIOD,
    )
    best_period = float(pg.period_at_max_power.value)
    peak_power = (
        float(pg.max_power.value) if hasattr(pg.max_power, "value") else float(pg.max_power)
    )

    t0 = float(time[int(np.nanargmin(flux))])
    phase = fold_phase(time, t0, best_period)

    prim_m = np.abs(phase) <= 0.08
    sec_m = np.abs(np.abs(phase) - 0.50) <= 0.08
    ooe_m = (~prim_m) & (~sec_m)

    baseline = (
        float(np.nanmedian(flux[ooe_m])) if ooe_m.sum() >= 30 else float(np.nanmedian(flux))
    )
    if baseline <= 0:
        baseline = float(np.nanmedian(flux[flux > 0])) if np.any(flux > 0) else np.nan

    prim_level = robust_level(flux[prim_m], 8) if prim_m.sum() else robust_level(flux, 5)
    sec_level = robust_level(flux[sec_m], 8) if sec_m.sum() else baseline

    primary_depth_ppt = (
        float((baseline - prim_level) / baseline * 1000)
        if baseline == baseline and baseline > 0
        else 0.0
    )
    secondary_depth_ppt = (
        float((baseline - sec_level) / baseline * 1000)
        if baseline == baseline and baseline > 0
        else 0.0
    )
    primary_depth_ppt = max(0.0, primary_depth_ppt)
    secondary_depth_ppt = max(0.0, secondary_depth_ppt)

    std_flux = float(np.nanstd(flux))
    variability_ppt = float(std_flux / mean_flux * 1000) if mean_flux > 0 else 0.0
    s_r2 = sine_r2(phase, flux)

    dip = baseline - prim_level if baseline == baseline else 0.0
    thresh = baseline - 0.30 * dip if dip > 0 else baseline
    duty_cycle = float(np.mean(flux < thresh)) if dip > 0 else 0.0

    ooe_rms = float(np.nanstd(flux[ooe_m])) if ooe_m.sum() >= 30 else std_flux
    ooe_rms_ratio = float(ooe_rms / std_flux) if std_flux > 0 else 1.0
    depth_ratio = (
        float(secondary_depth_ppt / primary_depth_ppt) if primary_depth_ppt > 1e-6 else 0.0
    )
    var_vs_depth = (
        float(variability_ppt / primary_depth_ppt) if primary_depth_ppt > 1e-6 else np.nan
    )

    return {
        "tic_id": int(tic),
        "sector": int(sector),
        "status": "ok",
        "best_period_days": best_period,
        "peak_power": peak_power,
        "primary_depth_ppt": primary_depth_ppt,
        "secondary_depth_ppt": secondary_depth_ppt,
        "variability_ppt": variability_ppt,
        "mean_flux": mean_flux,
        "num_points": int(n),
        "t0_btjd": t0,
        "period_at_ceiling": bool(best_period >= PERIOD_CEILING),
        "extreme_depth": bool(primary_depth_ppt > 800 and mean_flux < 100),
        "sine_r2": s_r2,
        "duty_cycle": duty_cycle,
        "ooe_rms_ratio": ooe_rms_ratio,
        "depth_ratio": depth_ratio,
        "var_vs_depth": var_vs_depth,
        "n_primary": int(prim_m.sum()),
        "n_secondary": int(sec_m.sum()),
        "n_ooe": int(ooe_m.sum()),
        "source_file": path.name,
    }


def run_sector(
    sector: int,
    fits_root: Path,
    out_dir: Path,
    log: logging.Logger,
    force: bool,
    checkpoint_every: int,
    sleep_between: float,
) -> None:
    parquet = out_dir / f"Sector_{sector:03d}_metrics.parquet"
    csv = out_dir / f"Sector_{sector:03d}_metrics.csv"

    done: set[int] = set()
    rows: list[dict] = []
    if parquet.exists() and not force:
        old = pd.read_parquet(parquet)
        rows = old.to_dict("records")
        done = {int(x) for x in old["tic_id"].dropna()}
        log.info("Sector %03d: resume with %d TICs already done", sector, len(done))

    files = find_lc_files(fits_root, sector, log)
    log.info("Sector %03d: %d FITS under %s", sector, len(files), sector_dir(fits_root, sector))
    if not files:
        log.warning("Sector %03d: no *_lc.fits — skip", sector)
        return

    ok = fail = skip = 0
    t0s = time.time()
    for i, path in enumerate(files, 1):
        _, tic = tic_from_name(path)
        if tic is not None and tic in done and not force:
            skip += 1
            continue
        try:
            rec = process_file(path, sector)
            if rec is None:
                fail += 1
                continue
            rows.append(rec)
            done.add(int(rec["tic_id"]))
            if rec.get("status") == "ok":
                ok += 1
            else:
                fail += 1
        except Exception as exc:
            fail += 1
            log.warning("  TIC %s %s: %s", tic, path.name, exc)

        if i % checkpoint_every == 0:
            df = pd.DataFrame(rows)
            df.to_parquet(parquet, index=False)
            df.to_csv(csv, index=False)
            log.info(
                "  S%03d %d/%d  ok=%d fail=%d skip=%d  %.1f min",
                sector,
                i,
                len(files),
                ok,
                fail,
                skip,
                (time.time() - t0s) / 60,
            )
            gc.collect()
        if sleep_between:
            time.sleep(sleep_between)

    if not rows:
        log.warning("Sector %03d: no rows written", sector)
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(["tic_id", "num_points"], ascending=[True, False])
    df = df.drop_duplicates("tic_id", keep="first")
    df.to_parquet(parquet, index=False)
    df.to_csv(csv, index=False)
    n_ok = int((df["status"] == "ok").sum()) if "status" in df.columns else len(df)
    log.info("Sector %03d finished  rows=%d ok=%d  → %s", sector, len(df), n_ok, parquet.name)


def build_logger(out_dir: Path) -> logging.Logger:
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"process_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    log = logging.getLogger("tess_process")
    log.info("Log file: %s", log_file)
    return log


def main() -> None:
    p = argparse.ArgumentParser(
        description="Process local TESS 2-minute SPOC light curves into per-sector metrics."
    )
    p.add_argument("--fits-root", default=r"D:\\", help="Parent of sector1, sector2, ...")
    p.add_argument("--out", default="tables/metrics", help="Directory for parquet/csv metrics")
    p.add_argument("--sectors", default="1-22", help="e.g. 1-22 or 23,24,25")
    p.add_argument("--force", action="store_true", help="Ignore existing sector tables")
    p.add_argument("--checkpoint-every", type=int, default=200)
    p.add_argument("--sleep", type=float, default=0.02, help="Pause between files (seconds)")
    args = p.parse_args()

    fits_root = Path(args.fits_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sectors = parse_sectors(args.sectors)
    log = build_logger(out_dir)

    log.info("TESS local light-curve processor")
    log.info("SECTORS_TO_RUN = %s", sectors)
    log.info("FITS root = %s", fits_root)
    log.info("Output    = %s", out_dir.resolve())

    overall = time.time()
    for sector in sectors:
        try:
            run_sector(
                sector,
                fits_root,
                out_dir,
                log,
                force=args.force,
                checkpoint_every=args.checkpoint_every,
                sleep_between=args.sleep,
            )
        except KeyboardInterrupt:
            log.warning("Interrupted — checkpoints are safe to resume")
            break
        except Exception as exc:
            log.exception("Sector %s crashed: %s", sector, exc)

    log.info("All done in %.1f min", (time.time() - overall) / 60)


if __name__ == "__main__":
    main()