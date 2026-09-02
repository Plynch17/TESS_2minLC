# TESS_2minLC
A pipeline for 2min Lightcurve processing and classification from MAST

# TESS 2-minute EA/EB search

A reproducible pipeline that measures TESS 2-minute SPOC light curves stored on disk, cross-matches survivors to VSX and SIMBAD, and writes a high-purity eclipsing-binary catalogue.

This is a starting-point experiment, not a complete TESS EB census. The search uses official 2-minute target light curves only, one sector at a time, with the period search capped near 8 days. Long-period and very shallow systems are incomplete by design.

## What you get

<!-- TABLE 1: scripts -->
Script | Role
--- | ---
SectorProcessing.py | Read local *_lc.fits and write per-sector metrics
combine_sectors.py | Stack those tables
match_catalogs.py | VSX + SIMBAD cross-match
TessClassificationv4.py | Frozen rules to paper tables

Paper-facing files written by classify.py:

- paper_known_EB.csv — catalogued EA/EB recovered at P or P/2
- paper_new_EA_candidates.csv — uncatalogued detached-looking systems that pass the purity cuts
- paper_composite.csv — hybrids, ellipsoidal variables, and large period mismatches
- manual_review.csv and rejected.csv — not science tables

best_period_days is the Lomb-Scargle peak in that sector. It is not automatically the orbital period. Use orbital_period_days for known systems.

## Requirements

- Python 3.10+
- A local copy of the 2-minute SPOC light curves (about 30-40 GB per sector)
- Network access only for match_catalogs.py

    pip install numpy pandas astropy astroquery lightkurve pyarrow

Expected FITS layout:

    D:\sector1\... *_lc.fits
    D:\sector2\...

sector01 and Sector1 are also accepted.

## Run order

    python SectorProcessing.py --fits-root D:\ --out tables/metrics --sectors 1-22
    python combine_sectors.py --in tables/metrics --out tables/all_sectors_metrics.parquet
    python match_catalogs.py --in tables/all_sectors_metrics.parquet --out tables/match.csv
    python TessClassificationv4.py --input tables/match.csv --out tables/classified

Resume is built in. Interrupted sector runs pick up from the existing parquet. Catalog matching writes tables/match_progress.csv and can be restarted.

Do not run match_catalogs.py on every raw TIC. The default prefilter keeps only status=ok rows with mean_flux >= 200, primary_depth_ppt >= 15, and peak_power >= 5.

## What the processor measures

For each light curve the script:

1. Drops NaNs but does not sigma-clip (clipping removes real eclipses)
2. Finds the strongest Lomb-Scargle period between 0.05 and 8 days
3. Sets phase zero at the time of minimum flux
4. Measures primary and secondary depths with a percentile estimator
5. Fits a single sine in phase and stores sine_r2

Useful columns: best_period_days, primary_depth_ppt, secondary_depth_ppt, sine_r2, depth_ratio, mean_flux, peak_power, period_at_ceiling, t0_btjd, source_file.

## Classification in one paragraph

Known EA/EB stay in the science table if the catalog type is an eclipse class and the TESS peak is P or P/2. Uncatalogued objects need a detached morphology, primary depth of at least 40 ppt, sine_r2 of at most 0.25, and period of at most 5.5 days. Pulsators, rotators, YSOs, and most 5-8 day window artefacts are rejected or sent to review. Composite types (EA+SPB, ELL, WR+E, large catalog-period mismatch) get their own table so they are not counted as clean detached binaries.

## Limits to state if you use the tables

- 2-minute SPOC targets only
- One ~27-day sector is the time baseline; periods near the 8-day cap are usually window artefacts
- TESS pixels are large; a deep dip can belong to a neighbour in the aperture
- Saturated stars (mean_flux above about 1e7 electrons per second) validate detection, not a physical depth
- "New" means not listed as an eclipsing binary in VSX/SIMBAD, not "never published in any TESS EB paper"
- confidence is a morphology score, not a probability

<!-- TABLE 2: column dictionary -->
Column | Meaning
--- | ---
best_period_days / tess_search_period_days | Lomb-Scargle peak in this sector
catalog_period | VSX period when present
period_quality | P_match, P/2_ok, P/3_check, mismatch, or no_catalog_P
orbital_period_days | published orbit when known
orbital_period_candidate_days | suggested orbit for detached morphology
sine_r2 | sine-fit score; high values are usually not boxy detached eclipses
saturated | mean flux at or above 1e7; detection OK, depth not physical
extreme_depth | primary depth at or above 600 ppt; do not use depth in plots

## Data you should not commit

- FITS light curves
- MAST download trees
- Multi-gigabyte metric dumps unless you use Git LFS or Zenodo

## Citation

TESS data: Ricker et al. 2015; SPOC 2-minute products via MAST.
VSX: Watson, Henden and Price.
SIMBAD: Wenger et al. 2000.


