#!/usr/bin/env python3
"""
classify.py
-----------
Apply the frozen eclipsing-binary classification rules to a catalog-matched table.

Input must already contain match_class, vsx_type, simbad_type, and the
photometric columns from process_local.py.

Example
-------
    python classify.py --input tables/match.csv --out tables/classified
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE_VERSION = "v4"


def s(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def add_reason(m: pd.DataFrame, mask: pd.Series, label: str) -> None:
    empty = m["veto_reason"] == ""
    m.loc[mask & empty, "veto_reason"] = label
    m.loc[mask & ~empty & (m["veto_reason"] != label), "veto_reason"] = (
        m.loc[mask & ~empty & (m["veto_reason"] != label), "veto_reason"] + ";" + label
    )


def other_reject(t) -> bool:
    if pd.isna(t) or str(t).strip() in ("", "nan"):
        return False
    u = str(t).upper()
    if any(k in u for k in ("EA", "EB", "EW", "ELL", "E/")):
        return False
    other = (
        "ROT", "RS", "CTTS", "YSO", "BY", "TTS", "NL", "UG",
        "GCAS", "BE", "WR", "ACV", "SXARI",
    )
    return any(k in u for k in other)


def is_generic_eclipse(t) -> bool:
    u = str(t).strip().upper()
    if u in ("", "NAN"):
        return False
    return u == "E" or u.startswith("E/") or u.startswith("E:") or u == "E:"


def type_has(type_col: pd.Series, pat: str) -> pd.Series:
    return type_col.str.contains(pat, regex=True, na=False)


def period_quality_row(row) -> str:
    alias = s(row.get("period_alias", "")).upper()
    r = row.get("period_ratio", np.nan)
    if pd.isna(r) or s(row.get("match_class", "")) in ("", "uncatalogued"):
        if alias in ("P/2",):
            return "P/2_ok"
        if alias in ("P/3",):
            return "P/3_check"
        if alias in ("MISMATCH",):
            return "mismatch"
        return "no_catalog_P"
    if 0.92 <= r <= 1.08 or alias == "P":
        return "P_match"
    if 1.85 <= r <= 2.15 or alias == "P/2":
        return "P/2_ok"
    if 2.75 <= r <= 3.30 or alias == "P/3":
        return "P/3_check"
    return "mismatch"


def load_input(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)
    df["tic_id"] = pd.to_numeric(df["tic_id"], errors="coerce").astype("Int64")
    for c in [
        "best_period_days", "primary_depth_ppt", "secondary_depth_ppt",
        "sine_r2", "mean_flux", "confidence", "depth_ratio", "vsx_period",
        "catalog_period", "period_ratio",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in [
        "vsx_type", "vsx_name", "simbad_type", "match_class", "period_alias",
        "veto_reason", "note", "morphology", "clean_class", "exception_note",
        "period_quality", "period_note",
    ]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].map(s)
    if "is_vetoed" not in df.columns:
        df["is_vetoed"] = False
    if "confidence" not in df.columns:
        df["confidence"] = 0.0
    if "depth_ratio" not in df.columns or df["depth_ratio"].isna().all():
        prim = df["primary_depth_ppt"].replace(0, np.nan)
        df["depth_ratio"] = df["secondary_depth_ppt"] / prim
        df["depth_ratio"] = df["depth_ratio"].fillna(0.0)
    if "morphology" not in df.columns or df["morphology"].eq("").all():
        df["morphology"] = np.where(
            (df["depth_ratio"].fillna(1) < 0.20) & (df["sine_r2"].fillna(1) < 0.35),
            "likely_EA",
            np.where(df["sine_r2"].fillna(0) >= 0.40, "sine_like", "likely_EB"),
        )
    return df


def apply_v1(m: pd.DataFrame, known_non_eb: set[int]) -> pd.DataFrame:
    m = m.copy()
    type_join = (m["vsx_type"].fillna("") + "|" + m["simbad_type"].fillna("")).str.upper()
    eb_tag = (
        m["match_class"].eq("catalog_EB")
        | type_join.str.contains(r"(?:EA|EB|EW|ECL)", regex=True, na=False)
    )
    sector_window = (
        (m["best_period_days"] >= 5.5)
        & (m["sine_r2"].fillna(0) >= 0.20)
        & ~eb_tag
    )
    long_two_min = (
        (m["best_period_days"] >= 5.0)
        & (m["depth_ratio"].fillna(0) >= 0.25)
        & ~eb_tag
    )
    sine_like = (
        (m["sine_r2"].fillna(0) >= 0.40)
        & (m["depth_ratio"].fillna(0) >= 0.50)
        & ~eb_tag
    )
    faint_shallow = (
        (m["mean_flux"].fillna(np.inf) < 200)
        & (m["primary_depth_ppt"].fillna(0) < 30)
        & (m["sine_r2"].fillna(0) > 0.35)
    )
    long_cat_short_obs = (
        (m["vsx_period"] >= 20)
        & (m["best_period_days"] >= 4.5)
        & ~eb_tag
    )
    known_hit = m["tic_id"].astype("Int64").isin(known_non_eb)

    m["veto_reason"] = ""
    add_reason(m, known_hit, "known_nonEB")
    add_reason(m, m["match_class"] == "catalog_nonEB", "catalog_pulsator")
    add_reason(m, sector_window, "sector_window_pulsator")
    add_reason(m, long_two_min, "longP_two_minima")
    add_reason(m, sine_like, "sine_like")
    add_reason(m, faint_shallow, "faint_shallow")
    add_reason(m, long_cat_short_obs, "long_catalog_period")
    add_reason(m, m["vsx_type"].map(other_reject), "catalog_other_nonEB")
    m["is_vetoed"] = m["veto_reason"] != ""

    def final_bin(row):
        vr = str(row["veto_reason"])
        if row["match_class"] == "catalog_EB" and not (
            "known_nonEB" in vr or "catalog_pulsator" in vr
        ):
            return "science_EB"
        if row["is_vetoed"]:
            return "rejected"
        if row["match_class"] == "uncatalogued":
            if (row["best_period_days"] >= 5.5) and (row["depth_ratio"] >= 0.25):
                return "manual_review"
            if row["morphology"] == "likely_EA" and row["secondary_depth_ppt"] < 8:
                return "science_new"
            return "manual_review"
        return "manual_review"

    m["clean_class"] = m.apply(final_bin, axis=1)
    return m


def apply_v2(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "note" not in df.columns:
        df["note"] = ""
    promote = (
        (df["clean_class"] == "manual_review")
        & (df["match_class"] == "catalog_other")
        & (df["morphology"] == "likely_EA")
        & df["vsx_type"].map(is_generic_eclipse)
    )
    df.loc[promote, "clean_class"] = "science_EB"
    df.loc[promote, "note"] = "promoted_generic_E"
    demote = (
        (df["clean_class"] == "science_new")
        & (df["match_class"] == "uncatalogued")
        & (df["best_period_days"] >= 5.5)
    )
    df.loc[demote, "clean_class"] = "manual_review"
    df.loc[demote, "note"] = "demoted_longP_uncatalogued"
    return df


def apply_v3(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    TYPE = (df["vsx_type"] + "|" + df["simbad_type"]).str.upper()
    HAS_ECLIPSE = type_has(TYPE, r"(EA|EB|EW|ECL|\bE/|\+E\b|\bE\+)")
    HARD_NON_EB = (
        type_has(TYPE, r"\b(SPB|BCEP|DCEP|DCEPS|DSCT|SXPHE|GDOR|RRAB|RRC|MIRA|\bM\b|SRB|SRA|SRC|LB|LPV|SR\b)")
        | type_has(TYPE, r"\b(UXOR|BLLAC|BLZAR|CTTS|INST|INT:|YSO|TTS|UG|NL|CBSS|GCAS(?!\+)|BE(?!\+)|ROT|BY|RS(?!\+)|ACV|ROAP|ROAM)")
    )
    HARD_NON_EB = HARD_NON_EB & ~HAS_ECLIPSE
    GENERIC_E = df["vsx_type"].str.match(r"^E[:/]?$", case=False, na=False) | (
        df["vsx_type"].str.upper() == "E"
    )

    df["catalog_period"] = df["vsx_period"]
    df["period_ratio"] = np.where(
        (df["catalog_period"] > 0) & (df["best_period_days"] > 0),
        df["catalog_period"] / df["best_period_days"],
        np.nan,
    )
    df["period_quality"] = df.apply(period_quality_row, axis=1)
    df["orbital_period_days"] = df["best_period_days"]
    df.loc[df["period_quality"] == "P/2_ok", "orbital_period_days"] = (
        df.loc[df["period_quality"] == "P/2_ok", "best_period_days"] * 2.0
    )
    m_cat = (df["period_quality"] == "P_match") & df["catalog_period"].notna()
    df.loc[m_cat, "orbital_period_days"] = df.loc[m_cat, "catalog_period"]

    df["saturated"] = df["mean_flux"] >= 1.0e7
    df["extreme_depth"] = df["primary_depth_ppt"] >= 600
    df["high_sine_r2"] = df["sine_r2"] >= 0.35
    df["long_catalog_mismatch"] = (
        (df["period_quality"] == "mismatch")
        & df["catalog_period"].notna()
        & (df["catalog_period"] >= 15)
    )

    ELL_ONLY = type_has(TYPE, r"\bELL") & ~type_has(TYPE, r"\bEA|\bEB|\bEW")
    COMPOSITE = (
        type_has(TYPE, r"\+|GCAS|WR|SPB|BCEP|HB|GS|ELL")
        | df["high_sine_r2"]
        | ELL_ONLY
    )

    df["exception_note"] = df.get("exception_note", "").map(s) if "exception_note" in df.columns else ""

    move_rej = HARD_NON_EB & df["clean_class"].isin(["science_EB", "science_new", "manual_review"])
    df.loc[move_rej, "clean_class"] = "rejected"
    df.loc[move_rej, "is_vetoed"] = True
    df.loc[move_rej, "veto_reason"] = df.loc[move_rej, "veto_reason"].map(
        lambda x: (x + ";" if x else "") + "hard_nonEB_type"
    )

    vetoed_sci = (
        df["is_vetoed"].astype(str).str.lower().isin(["true", "1"])
        & df["clean_class"].isin(["science_EB", "science_new"])
    )
    keep_comp = vetoed_sci & HAS_ECLIPSE
    demote_veto = vetoed_sci & ~HAS_ECLIPSE
    df.loc[keep_comp, "clean_class"] = "science_composite"
    df.loc[keep_comp, "exception_note"] = "catalog_eclipse_plus_extra;cleared_as_veto_conflict"
    df.loc[keep_comp, "is_vetoed"] = False
    df.loc[demote_veto, "clean_class"] = "manual_review"
    df.loc[demote_veto, "exception_note"] = "veto_conflict_demoted"

    sci = df["clean_class"].isin(["science_EB", "science_new"])
    df.loc[sci & COMPOSITE, "clean_class"] = "science_composite"

    p3 = df["clean_class"].isin(["science_EB", "science_new"]) & (df["period_quality"] == "P/3_check")
    df.loc[p3, "clean_class"] = "science_composite"
    mismatch_big = df["clean_class"].isin(["science_EB", "science_new"]) & df["long_catalog_mismatch"]
    df.loc[mismatch_big, "clean_class"] = "science_composite"

    promote_E = (
        (df["clean_class"] == "manual_review")
        & GENERIC_E
        & (df["morphology"] == "likely_EA")
        & (df["depth_ratio"].fillna(1) < 0.15)
        & (df["sine_r2"].fillna(1) < 0.35)
        & (df["period_quality"].isin(["P_match", "P/2_ok", "no_catalog_P"]))
        & (df["mean_flux"] >= 400)
        & (df["primary_depth_ppt"] >= 20)
        & ~HARD_NON_EB
    )
    df.loc[promote_E, "clean_class"] = "science_EB"
    df.loc[promote_E, "note"] = "promoted_generic_E"

    bad_E = (
        GENERIC_E
        & df["clean_class"].isin(["science_EB", "science_new"])
        & ~(
            (df["depth_ratio"].fillna(1) < 0.15)
            & (df["sine_r2"].fillna(1) < 0.35)
            & (df["period_quality"].isin(["P_match", "P/2_ok", "no_catalog_P"]))
        )
    )
    df.loc[bad_E, "clean_class"] = "manual_review"
    df.loc[bad_E, "exception_note"] = "generic_E_failed_rule"

    ext_new = (
        df["extreme_depth"]
        & (df["match_class"] == "uncatalogued")
        & df["clean_class"].isin(["science_EB", "science_new"])
    )
    df.loc[ext_new, "clean_class"] = "manual_review"

    conflict = (
        df["is_vetoed"].astype(str).str.lower().isin(["true", "1"])
        & df["clean_class"].str.startswith("science")
    )
    df.loc[conflict, "is_vetoed"] = False
    return df


def apply_period_policy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tess_search_period_days"] = df["best_period_days"]
    df["two_p_days"] = df["best_period_days"] * 2.0
    df["period_note"] = df.get("period_note", "")

    mismatch_known = (
        (df["match_class"] == "catalog_EB")
        & (df["period_quality"] == "mismatch")
        & df["catalog_period"].notna()
        & (df["catalog_period"] > 0)
    )
    df.loc[mismatch_known, "orbital_period_days"] = df.loc[mismatch_known, "catalog_period"]
    df.loc[mismatch_known, "period_note"] = "catalog_P_used;tess_peak_is_harmonic_or_window"

    r = df["period_ratio"]
    df["harmonic_or_window"] = False
    for n in (3, 4, 5, 6, 7, 8):
        df.loc[r.between(n - 0.12, n + 0.12), "harmonic_or_window"] = True
    df.loc[r.between(0.45, 0.55), "harmonic_or_window"] = True

    detached = (df["morphology"] == "likely_EA") & (df["depth_ratio"].fillna(1) < 0.20)
    df["orbital_period_candidate_days"] = np.where(
        detached, df["two_p_days"], df["orbital_period_days"]
    )
    p2 = df["period_quality"] == "P/2_ok"
    df.loc[p2, "orbital_period_candidate_days"] = df.loc[p2, "orbital_period_days"]
    df.loc[mismatch_known, "orbital_period_candidate_days"] = df.loc[mismatch_known, "catalog_period"]

    move = (
        (df["clean_class"] == "science_new")
        & (
            (df["sine_r2"] > 0.28)
            | (df["best_period_days"] > 5.5)
            | (df["primary_depth_ppt"] < 40)
        )
    )
    df.loc[move, "clean_class"] = "manual_review"
    df.loc[move, "exception_note"] = (
        df.loc[move, "exception_note"].fillna("").astype(str) + ";new_grey_zone"
    )
    return df


def uniq(sub: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ("confidence", "primary_depth_ppt", "mean_flux") if c in sub.columns]
    return sub.sort_values(cols, ascending=[False] * len(cols)).drop_duplicates("tic_id")


def write_products(df: pd.DataFrame, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["pipeline_version"] = PIPELINE_VERSION

    known_ok = uniq(df[
        (df["clean_class"] == "science_EB")
        & (df["match_class"] == "catalog_EB")
        & (df["period_quality"].isin(["P_match", "P/2_ok"]))
    ].copy())
    new_ok = uniq(df[
        (df["clean_class"] == "science_new")
        & (df["morphology"] == "likely_EA")
        & (df["primary_depth_ppt"] >= 40)
        & (df["sine_r2"] <= 0.25)
        & (df["best_period_days"] <= 5.5)
        & (df["extreme_depth"] != True)
    ].copy())
    comp = uniq(df[df["clean_class"] == "science_composite"].copy())
    review = df[df["clean_class"] == "manual_review"].copy()
    rej = df[df["clean_class"] == "rejected"].copy()

    df.to_csv(out / "all_classified.csv", index=False)
    known_ok.to_csv(out / "paper_known_EB.csv", index=False)
    new_ok.to_csv(out / "paper_new_EA_candidates.csv", index=False)
    comp.to_csv(out / "paper_composite.csv", index=False)
    review.to_csv(out / "manual_review.csv", index=False)
    rej.to_csv(out / "rejected.csv", index=False)
    return {
        "known": int(known_ok["tic_id"].nunique()) if len(known_ok) else 0,
        "new": int(new_ok["tic_id"].nunique()) if len(new_ok) else 0,
        "composite": int(comp["tic_id"].nunique()) if len(comp) else 0,
        "review": len(review),
        "rejected": len(rej),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Classify catalog-matched TESS EB candidates")
    p.add_argument("--input", required=True)
    p.add_argument("--out", default="tables/classified")
    p.add_argument("--known-non-eb", default="")
    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.out)
    df = load_input(inp)

    knp = Path(args.known_non_eb) if args.known_non_eb else inp.parent / "KNOWN_NON_EB.csv"
    if knp.exists():
        kdf = pd.read_csv(knp)
        col = "tic_id" if "tic_id" in kdf.columns else kdf.columns[0]
        known = set(pd.to_numeric(kdf[col], errors="coerce").dropna().astype(int))
        print(f"KNOWN_NON_EB: {len(known)} from {knp}")
    else:
        known = set(
            df.loc[df["match_class"] == "catalog_nonEB", "tic_id"]
            .dropna()
            .astype(int)
        )
        print(f"KNOWN_NON_EB: {len(known)} inferred from catalog_nonEB")

    df = apply_v1(df, known)
    df = apply_v2(df)
    df = apply_v3(df)
    df = apply_period_policy(df)
    stats = write_products(df, out)

    print(df["clean_class"].value_counts().to_string())
    print(f"Paper known EA/EB: {stats['known']}")
    print(f"Paper new EA:      {stats['new']}")
    print(f"Composite:         {stats['composite']}")
    print(f"Review / rejected: {stats['review']} / {stats['rejected']}")
    print("Wrote", out.resolve())


if __name__ == "__main__":
    main()