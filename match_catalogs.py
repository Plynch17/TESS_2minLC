#!/usr/bin/env python3
"""
match_catalogs.py
-----------------
Cross-match processed TICs to VSX (VizieR) and SIMBAD.

Looks up TIC coordinates from the MAST TIC catalog, then cone-searches
VSX and queries SIMBAD as TIC <id>. Writes match_class used by classify.py.

This step hits public services. Use --sleep and the default prefilter.

Example
-------
    python match_catalogs.py --in tables/all_sectors_metrics.parquet --out tables/match.csv
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
from astroquery.mast import Catalogs
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier

VSX_CATALOG = "B/vsx/vsx"
VSX_RADIUS = 5 * u.arcsec

EB_TOKENS = ("EA", "EB", "EW", "ECL")
PULS_TOKENS = (
    "DCEP", "DCEPS", "DSCT", "SXPHE", "GDOR", "RRAB", "RRC",
    "MIRA", "SRB", "SRA", "SRC", "LB", "LPV", "SPB", "BCEP",
)


def s(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def type_looks_eb(text: str) -> bool:
    u = text.upper()
    return any(tok in u for tok in EB_TOKENS) or u in ("E", "E:", "E/")


def type_looks_puls(text: str) -> bool:
    u = text.upper()
    return any(tok in u for tok in PULS_TOKENS)


def period_alias(tess_p: float, cat_p: float) -> str:
    if not (np.isfinite(tess_p) and np.isfinite(cat_p) and tess_p > 0 and cat_p > 0):
        return ""
    r = cat_p / tess_p
    if 0.92 <= r <= 1.08:
        return "P"
    if 1.85 <= r <= 2.15:
        return "P/2"
    if 2.75 <= r <= 3.30:
        return "P/3"
    if 0.45 <= r <= 0.55:
        return "2P"
    return "MISMATCH"


def tic_coords(tic_ids: list[int], sleep: float) -> pd.DataFrame:
    rows = []
    chunk = 200
    for i in range(0, len(tic_ids), chunk):
        part = tic_ids[i : i + chunk]
        try:
            tab = Catalogs.query_criteria(catalog="Tic", ID=part)
            for r in tab:
                rows.append(
                    {
                        "tic_id": int(r["ID"]),
                        "ra": float(r["ra"]),
                        "dec": float(r["dec"]),
                    }
                )
        except Exception as exc:
            print(f"  TIC batch {i} failed ({exc}); retrying one-by-one")
            for tic in part:
                try:
                    tab = Catalogs.query_criteria(catalog="Tic", ID=tic)
                    if len(tab):
                        rows.append(
                            {
                                "tic_id": int(tic),
                                "ra": float(tab[0]["ra"]),
                                "dec": float(tab[0]["dec"]),
                            }
                        )
                except Exception as e2:
                    print(f"    TIC {tic}: {e2}")
                time.sleep(sleep)
        time.sleep(sleep)
        print(f"  coords {min(i + chunk, len(tic_ids))}/{len(tic_ids)}")
    return pd.DataFrame(rows).drop_duplicates("tic_id")


def query_vsx(ra: float, dec: float):
    viz = Vizier(columns=["Name", "Type", "Period", "RAJ2000", "DEJ2000"])
    viz.ROW_LIMIT = 5
    try:
        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        tabs = viz.query_region(coord, radius=VSX_RADIUS, catalog=VSX_CATALOG)
    except Exception:
        return "", "", np.nan
    if not tabs:
        return "", "", np.nan
    t = tabs[0]
    name = s(t["Name"][0]) if "Name" in t.colnames else ""
    vtype = s(t["Type"][0]) if "Type" in t.colnames else ""
    per = np.nan
    if "Period" in t.colnames:
        try:
            per = float(t["Period"][0])
        except Exception:
            per = np.nan
    return name, vtype, per


def query_simbad(tic: int, custom: Simbad):
    try:
        tab = custom.query_object(f"TIC {tic}")
    except Exception:
        return "", "", ""
    if tab is None or len(tab) == 0:
        return "", "", ""
    r = tab[0]
    oid = s(r["MAIN_ID"]) if "MAIN_ID" in tab.colnames else ""
    otype = s(r["OTYPE"]) if "OTYPE" in tab.colnames else ""
    sptype = s(r["SP_TYPE"]) if "SP_TYPE" in tab.colnames else ""
    return oid, otype, sptype


def match_class_of(row) -> str:
    vsx_eb = bool(row["vsx_eb"])
    sim_eb = bool(row["simbad_eb"])
    vsx_puls = bool(row["vsx_puls"])
    sim_puls = bool(row["simbad_puls"])
    if vsx_eb or sim_eb:
        return "catalog_EB"
    if vsx_puls or sim_puls:
        return "catalog_nonEB"
    if s(row["vsx_type"]) or s(row["simbad_type"]):
        return "catalog_other"
    return "uncatalogued"


def main() -> None:
    p = argparse.ArgumentParser(description="Cross-match TICs to VSX and SIMBAD")
    p.add_argument("--in", dest="infile", required=True)
    p.add_argument("--out", default="tables/match.csv")
    p.add_argument("--sleep", type=float, default=0.25)
    p.add_argument("--limit", type=int, default=0, help="Debug: first N unique TICs")
    p.add_argument("--min-flux", type=float, default=200.0)
    p.add_argument("--min-depth", type=float, default=15.0)
    p.add_argument("--min-power", type=float, default=5.0)
    p.add_argument("--no-prefilter", action="store_true")
    p.add_argument("--progress", default="tables/match_progress.csv")
    args = p.parse_args()

    src = Path(args.infile)
    df = pd.read_parquet(src) if src.suffix == ".parquet" else pd.read_csv(src)
    df["tic_id"] = pd.to_numeric(df["tic_id"], errors="coerce")
    df = df.dropna(subset=["tic_id"])
    df["tic_id"] = df["tic_id"].astype(int)

    if not args.no_prefilter:
        mask = pd.Series(True, index=df.index)
        if "status" in df.columns:
            mask &= df["status"].eq("ok") | df["status"].isna()
        if "mean_flux" in df.columns:
            mask &= df["mean_flux"].fillna(0) >= args.min_flux
        if "primary_depth_ppt" in df.columns:
            mask &= df["primary_depth_ppt"].fillna(0) >= args.min_depth
        if "peak_power" in df.columns:
            mask &= df["peak_power"].fillna(0) >= args.min_power
        work = df.loc[mask].copy()
        print(f"Prefilter: {len(work):,} / {len(df):,} rows")
    else:
        work = df.copy()
        print(f"No prefilter: {len(work):,} rows")

    unique = work.drop_duplicates("tic_id")
    if args.limit:
        unique = unique.head(args.limit)
    tics = unique["tic_id"].astype(int).tolist()
    print(f"Unique TICs to match: {len(tics)}")

    print("Fetching TIC coordinates from MAST ...")
    coords = tic_coords(tics, args.sleep)
    unique = unique.merge(coords, on="tic_id", how="left")

    custom = Simbad()
    custom.add_votable_fields("otype", "sptype")

    prog = Path(args.progress)
    prog.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if prog.exists():
        prev = pd.read_csv(prog)
        done = {int(r.tic_id): r.to_dict() for r in prev.itertuples(index=False)}
        print(f"Resume: {len(done)} already matched")

    rows = []
    for i, rec in enumerate(unique.itertuples(index=False), 1):
        tic = int(rec.tic_id)
        if tic in done:
            rows.append(done[tic])
            continue
        ra, dec = getattr(rec, "ra", np.nan), getattr(rec, "dec", np.nan)
        vsx_name = vsx_type = sim_id = sim_type = sim_sp = ""
        vsx_period = np.nan
        if np.isfinite(ra) and np.isfinite(dec):
            vsx_name, vsx_type, vsx_period = query_vsx(ra, dec)
        sim_id, sim_type, sim_sp = query_simbad(tic, custom)
        tess_p = float(getattr(rec, "best_period_days", np.nan))
        row = {
            "tic_id": tic,
            "ra": ra,
            "dec": dec,
            "vsx_name": vsx_name,
            "vsx_type": vsx_type,
            "vsx_period": vsx_period,
            "simbad_id": sim_id,
            "simbad_type": sim_type,
            "simbad_sptype": sim_sp,
            "vsx_eb": type_looks_eb(vsx_type),
            "simbad_eb": type_looks_eb(sim_type),
            "vsx_puls": type_looks_puls(vsx_type),
            "simbad_puls": type_looks_puls(sim_type),
            "period_alias": period_alias(tess_p, vsx_period),
        }
        rows.append(row)
        done[tic] = row
        if i % 25 == 0:
            pd.DataFrame(rows).to_csv(prog, index=False)
            print(f"  matched {i}/{len(unique)}")
        time.sleep(args.sleep)

    hits = pd.DataFrame(rows)
    hits["match_class"] = hits.apply(match_class_of, axis=1)
    hits.to_csv(prog, index=False)

    merged = work.merge(hits, on="tic_id", how="left")
    merged["match_class"] = merged["match_class"].fillna("uncatalogued")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    merged.to_parquet(out.with_suffix(".parquet"), index=False)
    print("match_class:\n", merged.drop_duplicates("tic_id")["match_class"].value_counts().to_string())
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()