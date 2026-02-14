"""
PJM Data Center Siting Analysis
================================
Pulls Day-Ahead and Real-Time LMP data plus system/zonal load from PJM
via the gridstatus library, computes location-level summary metrics,
ranks locations on cost / stability / congestion / basis criteria, and
produces CSV exports and an interactive HTML dashboard.

Metrics produced per location:
  - Average DA and RT LMP
  - Average congestion component and congestion share
  - LMP volatility (std dev) and 95th-percentile
  - DA-RT basis (mean and volatility)
  - Correlation of RT LMP with load (hourly Pearson r)

Rankings:
  1. Cheap + Stable   – low average RT LMP and low volatility
  2. Low Congestion   – low congestion component and share
  3. Low Basis Risk   – low absolute basis and basis volatility

Prerequisites
-------------
  export PJM_API_KEY="your-pjm-api-key"
  pip install -r requirements.txt

Usage
-----
  python pjm_siting_analysis.py

Outputs are written to ./out/ (CSVs, interactive HTML dashboard, and
console executive summary).

Timezone Handling
-----------------
gridstatus returns PJM timestamps in US/Eastern (EPT).  All internal
processing preserves this timezone.
If timestamps arrive timezone-naive, they are localized to US/Eastern.
"""

import os
import sys
import json
import math
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import gridstatus
from gridstatus import Markets

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ===================================================================
# SECTION 1 – CONFIGURATION  (edit these before running)
# ===================================================================

# Date range (inclusive).  Strings in "YYYY-MM-DD" format.
START_DATE: str = "2024-06-01"
END_DATE: str = "2024-08-31"

# Markets to pull.  Must be valid gridstatus.Markets members.
MARKETS_TO_PULL = [Markets.DAY_AHEAD_HOURLY, Markets.REAL_TIME_HOURLY]

# Location mode: "zonal" or "nodal"
#   "zonal" – pulls all PJM pricing zones (locations="zones")
#   "nodal" – pulls specific pnode IDs listed in LOCATIONS
LOCATION_MODE: str = "zonal"

# Locations – only used when LOCATION_MODE == "nodal".
# Provide a list of integer pnode IDs.  Leave empty to trigger a
# discovery pull that prints available nodes and exits.
LOCATIONS: list = []

# Output directory (relative to this script).
OUT_DIR: Path = Path(__file__).resolve().parent / "out"

# Number of top locations to show on the map and in tables.
TOP_N: int = 15

# Approximate geographic centers for PJM pricing zones (lat, lon).
# Used to place markers on the interactive Leaflet map.
ZONE_COORDS: dict = {
    "AE":     (39.36, -74.42),   # Atlantic City Electric — southern NJ
    "AEP":    (39.96, -82.99),   # American Electric Power — Columbus OH
    "APS":    (41.10, -80.65),   # Allegheny Power — western PA / OH border
    "ATSI":   (41.50, -81.69),   # FirstEnergy — Cleveland OH
    "BC":     (39.29, -76.61),   # BGE — Baltimore MD
    "COMED":  (41.88, -87.63),   # ComEd — Chicago IL
    "DAYTON": (39.76, -84.20),   # Dayton P&L — Dayton OH
    "DEOK":   (39.10, -84.51),   # Duke Energy OH/KY — Cincinnati
    "DOM":    (37.54, -77.44),   # Dominion — Richmond VA
    "DPL":    (39.74, -75.55),   # Delmarva Power — Wilmington DE
    "DUQ":    (40.44, -79.99),   # Duquesne Light — Pittsburgh PA
    "EKPC":   (38.05, -84.50),   # Eastern KY Power — Lexington KY
    "JC":     (40.49, -74.45),   # Jersey Central — central NJ
    "ME":     (40.33, -75.93),   # Met-Ed — Reading PA
    "PE":     (39.95, -75.17),   # PECO — Philadelphia PA
    "PEP":    (38.91, -77.04),   # Pepco — Washington DC
    "PL":     (40.61, -75.49),   # PPL — Allentown PA
    "PN":     (41.24, -78.73),   # Penelec — north-central PA
    "PS":     (40.74, -74.17),   # PSE&G — Newark NJ
    "RECO":   (41.05, -74.13),   # Rockland Electric — northern NJ
}


# ===================================================================
# SECTION 2 – COLUMN CONSTANTS & MAPPING
# ===================================================================

# gridstatus get_lmp() returns:
#   Time, Interval Start, Interval End, Market, Location Id,
#   Location Name, Location Short Name, Location Type,
#   LMP, Energy, Congestion, Loss
#
# We rename to lowercase snake_case for internal consistency.

LMP_COL_MAP: dict = {
    "Interval Start": "time",
    "Location Id": "location_id",
    "Location Name": "location",
    "Location Short Name": "location_short",
    "Location Type": "location_type",
    "LMP": "lmp_total",
    "Energy": "lmp_energy",
    "Congestion": "lmp_congestion",
    "Loss": "lmp_loss",
    "Market": "market",
}

# Columns retained after renaming.
LMP_KEEP_COLS: list = list(LMP_COL_MAP.values())

# Load data column references.
LOAD_TIME_COL: str = "Interval Start"
LOAD_SYSTEM_COL: str = "Load"

# Columns that are NOT zone-level load in the raw load DataFrame.
LOAD_META_COLS: set = {"Time", "Interval Start", "Interval End", "Load"}


# ===================================================================
# SECTION 3 – HELPER FUNCTIONS
# ===================================================================

def ensure_output_dir() -> Path:
    """Create the output directory if it does not exist."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", OUT_DIR)
    return OUT_DIR


def pull_lmp(
    pjm: gridstatus.PJM,
    market: Markets,
    start: str,
    end: str,
    location_mode: str,
    locations: list,
) -> pd.DataFrame:
    """
    Pull LMP data for one market and return a cleaned DataFrame.

    Parameters
    ----------
    pjm : gridstatus.PJM
        Initialized PJM client.
    market : Markets
        e.g. Markets.DAY_AHEAD_HOURLY
    start, end : str
        Date strings "YYYY-MM-DD".
    location_mode : str
        "zonal" or "nodal".
    locations : list
        Pnode IDs (used only when location_mode == "nodal").

    Returns
    -------
    pd.DataFrame with columns defined by LMP_KEEP_COLS.
    """
    log.info("Pulling %s LMP data (%s to %s) ...", market, start, end)

    try:
        if location_mode == "zonal":
            df = pjm.get_lmp(
                date=start,
                market=market,
                end=end,
                locations="zones",
                verbose=False,
            )
        elif location_mode == "nodal" and locations:
            df = pjm.get_lmp(
                date=start,
                market=market,
                end=end,
                locations=locations,
                verbose=False,
            )
        else:
            # Discovery mode: pull one day to list available nodes.
            log.info(
                "No LOCATIONS specified.  Running discovery pull for %s ...",
                start,
            )
            df = pjm.get_lmp(
                date=start,
                market=market,
                end=start,
                locations="ALL",
                verbose=False,
            )
            unique_locs = (
                df[["Location Id", "Location Name", "Location Type"]]
                .drop_duplicates()
                .sort_values("Location Name")
            )
            print("\n=== Available PJM Nodes (sample) ===")
            print(unique_locs.to_string(index=False, max_rows=100))
            print(
                f"\nTotal unique locations: {len(unique_locs)}"
                "\nPopulate the LOCATIONS list with desired pnode IDs and re-run."
            )
            sys.exit(0)
    except Exception as exc:
        log.error("Failed to pull %s LMP data: %s", market, exc)
        raise SystemExit(
            f"Data pull failed for {market}.  "
            "Check PJM_API_KEY, date range, and network connectivity."
        ) from exc

    if df is None or df.empty:
        log.warning("No data returned for %s.", market)
        raise SystemExit(f"Empty dataset for {market}.  Cannot proceed.")

    # Detect available columns and map them.
    # gridstatus may include extra columns; we only keep what we need.
    available = set(df.columns)
    rename_map = {k: v for k, v in LMP_COL_MAP.items() if k in available}
    df = df.rename(columns=rename_map)

    # Keep only the columns we need (that actually exist after rename).
    keep = [c for c in LMP_KEEP_COLS if c in df.columns]
    df = df[keep].copy()

    # Enforce numeric dtypes on price columns.
    for col in ["lmp_total", "lmp_energy", "lmp_congestion", "lmp_loss"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    log.info(
        "  %s: %d rows, %d locations, %s to %s",
        market,
        len(df),
        df["location"].nunique() if "location" in df.columns else "?",
        df["time"].min() if "time" in df.columns else "?",
        df["time"].max() if "time" in df.columns else "?",
    )
    return df


def pull_load(
    pjm: gridstatus.PJM,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Pull PJM system and zonal load, resample from 5-min to hourly,
    and return in long format with columns: time, zone, load_mw.
    """
    log.info("Pulling load data (%s to %s) ...", start, end)

    try:
        df = pjm.get_load(date=start, end=end, verbose=False)
    except Exception as exc:
        log.error("Failed to pull load data: %s", exc)
        raise SystemExit(
            "Load data pull failed.  "
            "Check PJM_API_KEY, date range, and network connectivity."
        ) from exc

    if df is None or df.empty:
        log.warning("No load data returned.")
        raise SystemExit("Empty load dataset.  Cannot proceed.")

    # Identify zone columns dynamically.
    zone_cols = [c for c in df.columns if c not in LOAD_META_COLS]
    log.info("  Detected %d zone columns: %s", len(zone_cols), zone_cols)

    # Use Interval Start as the time index.
    if LOAD_TIME_COL not in df.columns:
        # Fallback: try "Time"
        if "Time" in df.columns:
            df = df.rename(columns={"Time": LOAD_TIME_COL})
        else:
            raise SystemExit(
                f"Expected column '{LOAD_TIME_COL}' not found in load data.  "
                f"Available: {list(df.columns)}"
            )

    df[LOAD_TIME_COL] = pd.to_datetime(df[LOAD_TIME_COL])
    df = df.set_index(LOAD_TIME_COL)

    # Resample 5-min intervals to hourly means.
    # Mean is appropriate because load is an instantaneous power (MW) reading.
    cols_to_resample = [LOAD_SYSTEM_COL] + zone_cols
    cols_to_resample = [c for c in cols_to_resample if c in df.columns]
    df_hourly = df[cols_to_resample].resample("h").mean()

    # Melt zone columns into long format.
    df_hourly = df_hourly.reset_index()
    df_hourly = df_hourly.rename(columns={LOAD_TIME_COL: "time"})

    # Build long-form zone load.
    zone_cols_present = [c for c in zone_cols if c in df_hourly.columns]
    load_long = df_hourly[["time"] + zone_cols_present].melt(
        id_vars=["time"],
        value_vars=zone_cols_present,
        var_name="zone",
        value_name="load_mw",
    )

    # Append system-wide load as a pseudo-zone "PJM_RTO".
    if LOAD_SYSTEM_COL in df_hourly.columns:
        sys_load = df_hourly[["time", LOAD_SYSTEM_COL]].copy()
        sys_load = sys_load.rename(columns={LOAD_SYSTEM_COL: "load_mw"})
        sys_load["zone"] = "PJM_RTO"
        load_long = pd.concat([load_long, sys_load], ignore_index=True)

    load_long["load_mw"] = pd.to_numeric(load_long["load_mw"], errors="coerce")
    log.info(
        "  Load: %d rows (hourly x zones), %s to %s",
        len(load_long),
        load_long["time"].min(),
        load_long["time"].max(),
    )
    return load_long


def map_location_to_zone(location_short: str) -> str:
    """
    Map an LMP Location Short Name (e.g. 'COMED', 'AE')
    to the load zone abbreviation used in get_load() output.

    The load DataFrame uses bare abbreviations (AE, AEP, COMED …).
    LMP Location Short Names for zones are typically the same abbreviation
    or may have a ' Zone' suffix.  We strip that suffix and uppercase.
    """
    if not isinstance(location_short, str):
        return str(location_short)
    name = location_short.upper().replace(" ZONE", "").strip()
    return name


# ===================================================================
# SECTION 4 – DATA PULL ORCHESTRATION
# ===================================================================

def fetch_all_data() -> tuple:
    """
    Orchestrate all data pulls.

    Returns
    -------
    (df_da, df_rt, df_load) – three DataFrames.
    """
    # Validate API key.
    api_key = os.getenv("PJM_API_KEY")
    if not api_key:
        log.error(
            "PJM_API_KEY environment variable is not set.  "
            "Register at https://www.pjm.com/ and export PJM_API_KEY."
        )
        sys.exit(1)

    pjm = gridstatus.PJM(api_key=api_key)

    # Pull DA LMPs.
    df_da = pull_lmp(
        pjm,
        market=Markets.DAY_AHEAD_HOURLY,
        start=START_DATE,
        end=END_DATE,
        location_mode=LOCATION_MODE,
        locations=LOCATIONS,
    )

    # Pull RT LMPs.
    df_rt = pull_lmp(
        pjm,
        market=Markets.REAL_TIME_HOURLY,
        start=START_DATE,
        end=END_DATE,
        location_mode=LOCATION_MODE,
        locations=LOCATIONS,
    )

    # Pull load.
    df_load = pull_load(pjm, start=START_DATE, end=END_DATE)

    # Save raw CSVs immediately.
    df_da.to_csv(OUT_DIR / "lmp_da_raw.csv", index=False)
    log.info("Saved %s", OUT_DIR / "lmp_da_raw.csv")

    df_rt.to_csv(OUT_DIR / "lmp_rt_raw.csv", index=False)
    log.info("Saved %s", OUT_DIR / "lmp_rt_raw.csv")

    df_load.to_csv(OUT_DIR / "load_raw.csv", index=False)
    log.info("Saved %s", OUT_DIR / "load_raw.csv")

    return df_da, df_rt, df_load


# ===================================================================
# SECTION 5 – CLEANING & ALIGNMENT
# ===================================================================

def merge_and_align(
    df_da: pd.DataFrame,
    df_rt: pd.DataFrame,
    df_load: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge DA and RT LMP data on (time, location), compute derived
    columns (basis, congestion share), and join load.

    Returns a single DataFrame with one row per (hour, location).
    """
    log.info("Merging DA and RT data ...")

    # Suffix price columns for DA and RT.
    da_price_cols = {
        "lmp_total": "lmp_total_da",
        "lmp_energy": "lmp_energy_da",
        "lmp_congestion": "lmp_congestion_da",
        "lmp_loss": "lmp_loss_da",
    }
    rt_price_cols = {
        "lmp_total": "lmp_total_rt",
        "lmp_energy": "lmp_energy_rt",
        "lmp_congestion": "lmp_congestion_rt",
        "lmp_loss": "lmp_loss_rt",
    }

    df_da_r = df_da.rename(columns=da_price_cols).copy()
    df_rt_r = df_rt.rename(columns=rt_price_cols).copy()

    # Drop the 'market' column before merging (it differs between DA/RT).
    for frame in (df_da_r, df_rt_r):
        if "market" in frame.columns:
            frame.drop(columns=["market"], inplace=True)

    # Define merge keys.
    merge_keys = ["time", "location", "location_id", "location_short", "location_type"]
    # Keep only existing keys.
    merge_keys = [k for k in merge_keys if k in df_da_r.columns and k in df_rt_r.columns]

    da_val_cols = [c for c in da_price_cols.values() if c in df_da_r.columns]
    rt_val_cols = [c for c in rt_price_cols.values() if c in df_rt_r.columns]

    df = pd.merge(
        df_da_r[merge_keys + da_val_cols],
        df_rt_r[merge_keys + rt_val_cols],
        on=merge_keys,
        how="outer",
    )
    log.info("  Merged shape: %s", df.shape)

    # Compute DA-RT basis.
    if "lmp_total_da" in df.columns and "lmp_total_rt" in df.columns:
        df["basis"] = df["lmp_total_da"] - df["lmp_total_rt"]

    # Compute congestion share (fraction of total LMP from congestion).
    # Guard against divide-by-zero with a small threshold.
    for suffix in ("da", "rt"):
        total_col = f"lmp_total_{suffix}"
        cong_col = f"lmp_congestion_{suffix}"
        share_col = f"cong_share_{suffix}"
        if total_col in df.columns and cong_col in df.columns:
            df[share_col] = np.where(
                df[total_col].abs() < 0.01,
                0.0,
                df[cong_col] / df[total_col],
            )

    # Map LMP location to load zone abbreviation for the join.
    if "location_short" in df.columns:
        df["zone"] = df["location_short"].apply(map_location_to_zone)
    elif "location" in df.columns:
        df["zone"] = df["location"].apply(map_location_to_zone)
    else:
        df["zone"] = "PJM_RTO"

    # Ensure both sides have compatible time dtypes for the join.
    df["time"] = pd.to_datetime(df["time"])
    df_load["time"] = pd.to_datetime(df_load["time"])

    # Strip timezone from both sides if mixed (tz-aware vs naive).
    if df["time"].dt.tz is not None:
        df["time"] = df["time"].dt.tz_localize(None)
    if df_load["time"].dt.tz is not None:
        df_load["time"] = df_load["time"].dt.tz_localize(None)

    # Left-join load on (time, zone).
    df = pd.merge(df, df_load, on=["time", "zone"], how="left")

    load_matched = df["load_mw"].notna().sum()
    log.info(
        "  Load matched on %d / %d rows (%.1f%%)",
        load_matched,
        len(df),
        100.0 * load_matched / max(len(df), 1),
    )

    df = df.sort_values(["location", "time"]).reset_index(drop=True)

    # Save merged data.
    df.to_csv(OUT_DIR / "lmp_merged.csv", index=False)
    log.info("Saved %s", OUT_DIR / "lmp_merged.csv")

    return df


# ===================================================================
# SECTION 6 – SUMMARY METRICS
# ===================================================================

def compute_summary(df_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-location summary statistics.

    Returns one row per location with metrics for DA, RT, basis,
    congestion, and load correlation.
    """
    log.info("Computing summary metrics ...")

    def _agg(group: pd.DataFrame) -> pd.Series:
        """Aggregate a single location's hourly data."""
        result = {}

        # Average LMPs.
        result["avg_lmp_da"] = group["lmp_total_da"].mean() if "lmp_total_da" in group else np.nan
        result["avg_lmp_rt"] = group["lmp_total_rt"].mean() if "lmp_total_rt" in group else np.nan

        # Average congestion component.
        result["avg_congestion_da"] = group["lmp_congestion_da"].mean() if "lmp_congestion_da" in group else np.nan
        result["avg_congestion_rt"] = group["lmp_congestion_rt"].mean() if "lmp_congestion_rt" in group else np.nan

        # Average congestion share.
        result["avg_cong_share_da"] = group["cong_share_da"].mean() if "cong_share_da" in group else np.nan
        result["avg_cong_share_rt"] = group["cong_share_rt"].mean() if "cong_share_rt" in group else np.nan

        # Volatility (standard deviation).
        result["vol_lmp_da"] = group["lmp_total_da"].std() if "lmp_total_da" in group else np.nan
        result["vol_lmp_rt"] = group["lmp_total_rt"].std() if "lmp_total_rt" in group else np.nan

        # 95th percentile.
        result["p95_lmp_da"] = group["lmp_total_da"].quantile(0.95) if "lmp_total_da" in group else np.nan
        result["p95_lmp_rt"] = group["lmp_total_rt"].quantile(0.95) if "lmp_total_rt" in group else np.nan

        # Basis (DA - RT).
        result["avg_basis"] = group["basis"].mean() if "basis" in group else np.nan
        result["vol_basis"] = group["basis"].std() if "basis" in group else np.nan

        # Correlation of RT LMP with load (hourly Pearson r).
        if "load_mw" in group and "lmp_total_rt" in group:
            valid = group[["lmp_total_rt", "load_mw"]].dropna()
            if len(valid) > 10:
                result["corr_load_rt"] = valid["lmp_total_rt"].corr(valid["load_mw"])
            else:
                result["corr_load_rt"] = np.nan
        else:
            result["corr_load_rt"] = np.nan

        # Data coverage (count of hours with RT data).
        result["count_hours"] = int(
            group["lmp_total_rt"].notna().sum() if "lmp_total_rt" in group else 0
        )

        return pd.Series(result)

    # Determine groupby keys based on available columns.
    group_keys = []
    for col in ["location", "location_id", "location_short"]:
        if col in df_merged.columns:
            group_keys.append(col)
    if not group_keys:
        raise SystemExit("No location columns found in merged data.")

    summary = df_merged.groupby(group_keys, group_keys=False).apply(_agg).reset_index()

    # Save.
    summary.to_csv(OUT_DIR / "location_summary.csv", index=False)
    log.info("Saved %s  (%d locations)", OUT_DIR / "location_summary.csv", len(summary))

    return summary


# ===================================================================
# SECTION 7 – RANKINGS
# ===================================================================

def _min_max(series: pd.Series) -> pd.Series:
    """Normalize a Series to [0, 1] using min-max scaling."""
    smin, smax = series.min(), series.max()
    if (smax - smin) < 1e-9:
        return pd.Series(0.0, index=series.index)
    return (series - smin) / (smax - smin)


def compute_rankings(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Rank locations on three composite criteria.
    Uses min-max normalization so each component is on [0, 1].
    Lower composite score = better location.

    Criteria
    --------
    1. cheap_stable    : 50% norm(avg_lmp_rt) + 50% norm(vol_lmp_rt)
    2. low_congestion  : 50% norm(|avg_congestion_rt|) + 50% norm(|avg_cong_share_rt|)
    3. low_basis       : 50% norm(|avg_basis|) + 50% norm(vol_basis)
    """
    log.info("Computing rankings ...")
    s = summary.copy()

    # --- Cheap + Stable ---
    s["cheap_stable_score"] = (
        0.5 * _min_max(s["avg_lmp_rt"])
        + 0.5 * _min_max(s["vol_lmp_rt"])
    )
    s["cheap_stable_rank"] = (
        s["cheap_stable_score"].rank(method="min").astype(int)
    )

    # --- Low Congestion Risk ---
    s["low_congestion_score"] = (
        0.5 * _min_max(s["avg_congestion_rt"].abs())
        + 0.5 * _min_max(s["avg_cong_share_rt"].abs())
    )
    s["low_congestion_rank"] = (
        s["low_congestion_score"].rank(method="min").astype(int)
    )

    # --- Low Basis Risk ---
    s["low_basis_score"] = (
        0.5 * _min_max(s["avg_basis"].abs())
        + 0.5 * _min_max(s["vol_basis"])
    )
    s["low_basis_rank"] = (
        s["low_basis_score"].rank(method="min").astype(int)
    )

    # Save.
    s.to_csv(OUT_DIR / "rankings.csv", index=False)
    log.info("Saved %s", OUT_DIR / "rankings.csv")

    return s


# ===================================================================
# SECTION 8 – INTERACTIVE HTML DASHBOARD
# ===================================================================

# The template uses __PLACEHOLDER__ tokens (not {braces}) to avoid
# conflicts with CSS and JavaScript curly braces.

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PJM Data Center Siting Analysis</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
      integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin="" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
        integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
<style>
*,*::before,*::after{box-sizing:border-box}
body{
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  margin:0; padding:0; background:#f4f5f7; color:#333;
}
header{
  background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);
  color:#fff; padding:24px 32px; margin-bottom:20px;
}
header h1{margin:0; font-size:1.7em; font-weight:700}
header p{margin:6px 0 0; opacity:.8; font-size:.92em}
.container{max-width:1280px; margin:0 auto; padding:0 20px 40px}
#controls{
  display:flex; align-items:center; gap:12px;
  margin-bottom:16px; flex-wrap:wrap;
}
#controls label{font-weight:600; font-size:.95em}
#controls select{
  padding:8px 14px; border-radius:6px; border:1px solid #c0c0c0;
  font-size:.93em; background:#fff; cursor:pointer;
}
#map{
  height:520px; width:100%; border-radius:10px;
  box-shadow:0 2px 12px rgba(0,0,0,.12); margin-bottom:24px;
}
#table-container{
  overflow-x:auto; background:#fff; border-radius:10px;
  box-shadow:0 2px 12px rgba(0,0,0,.12);
}
table{width:100%; border-collapse:collapse; font-size:.88em}
thead{position:sticky; top:0; z-index:2}
th{
  background:#1a1a2e; color:#fff; padding:11px 14px;
  cursor:pointer; white-space:nowrap; user-select:none;
  font-weight:600; text-align:right; border-bottom:2px solid #0f3460;
}
th:first-child,th:nth-child(2){text-align:left}
th:hover{background:#2a2a4e}
th .arrow{font-size:.7em; margin-left:4px; opacity:.6}
th .arrow.active{opacity:1}
td{padding:9px 14px; border-bottom:1px solid #eaeaea; text-align:right}
td:first-child,td:nth-child(2){text-align:left; font-weight:600}
tr:nth-child(even){background:#f8f8fc}
tr:hover{background:#e6eaf4}
tr.highlighted{background:#fff3cd !important; transition:background .3s}
.legend-box{
  background:#fff; padding:10px 14px; border-radius:6px;
  box-shadow:0 1px 5px rgba(0,0,0,.2); font-size:.85em; line-height:1.6;
}
.legend-box i{display:inline-block; width:14px; height:14px; margin-right:6px;
  vertical-align:middle; border-radius:50%}
.note{
  margin-top:16px; font-size:.82em; color:#777; text-align:center;
}
</style>
</head>
<body>

<header>
  <h1>PJM Data Center Siting Analysis</h1>
  <p>__START_DATE__ to __END_DATE__&ensp;|&ensp;__N_LOCATIONS__ zones analyzed
     &ensp;|&ensp;Mode: __LOCATION_MODE__</p>
</header>

<div class="container">

<div id="controls">
  <label for="ranking-select">Rank by:</label>
  <select id="ranking-select">
    <option value="cheap_stable">Cheap + Stable</option>
    <option value="low_congestion">Low Congestion Risk</option>
    <option value="low_basis">Low Basis Risk</option>
  </select>
</div>

<div id="map"></div>

<div id="table-container">
<table id="data-table">
  <thead>
    <tr>
      <th data-col="rank" data-type="num">Rank<span class="arrow"></span></th>
      <th data-col="location" data-type="str">Location<span class="arrow"></span></th>
      <th data-col="avg_lmp_rt" data-type="num">Avg RT LMP<span class="arrow"></span></th>
      <th data-col="avg_lmp_da" data-type="num">Avg DA LMP<span class="arrow"></span></th>
      <th data-col="avg_congestion_rt" data-type="num">Congestion RT<span class="arrow"></span></th>
      <th data-col="avg_cong_share_rt" data-type="num">Cong Share<span class="arrow"></span></th>
      <th data-col="vol_lmp_rt" data-type="num">Vol RT<span class="arrow"></span></th>
      <th data-col="p95_lmp_rt" data-type="num">P95 RT<span class="arrow"></span></th>
      <th data-col="avg_basis" data-type="num">Avg Basis<span class="arrow"></span></th>
      <th data-col="vol_basis" data-type="num">Vol Basis<span class="arrow"></span></th>
      <th data-col="corr_load_rt" data-type="num">Load Corr<span class="arrow"></span></th>
      <th data-col="count_hours" data-type="num">Hours<span class="arrow"></span></th>
    </tr>
  </thead>
  <tbody></tbody>
</table>
</div>

<p class="note">
  Map tiles &copy; <a href="https://openstreetmap.org">OpenStreetMap</a> contributors.
  Data from PJM via gridstatus.  Open <code>out/rankings.csv</code> for full data.
</p>

</div><!-- .container -->

<script>
// ---- Embedded data (injected by Python) ----
const DATA = __DATA_JSON__;

// ---- Ranking configuration ----
const RANKINGS = {
  cheap_stable:   {rankCol:'cheap_stable_rank',   scoreCol:'cheap_stable_score',   label:'Cheap + Stable'},
  low_congestion: {rankCol:'low_congestion_rank',  scoreCol:'low_congestion_score',  label:'Low Congestion Risk'},
  low_basis:      {rankCol:'low_basis_rank',       scoreCol:'low_basis_score',       label:'Low Basis Risk'},
};
const TOP_N = __TOP_N__;

// ---- Color scale: score 0 (best/green) → 1 (worst/red) ----
function getColor(score){
  if(score===null||score===undefined||isNaN(score)) return '#999';
  const s = Math.max(0, Math.min(1, score));
  const hue = 120*(1-s);          // 120=green, 0=red
  return 'hsl('+hue+',70%,42%)';
}

// ---- Format helpers ----
function fmt$(v){ return v===null||v===undefined?'N/A':'$'+v.toFixed(2); }
function fmt4(v){ return v===null||v===undefined?'N/A':v.toFixed(4); }
function fmt3(v){ return v===null||v===undefined?'N/A':v.toFixed(3); }
function fmtInt(v){ return v===null||v===undefined?'N/A':Math.round(v).toLocaleString(); }

// ---- Initialize Leaflet map ----
const map = L.map('map').setView([39.8, -79.0], 6);
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{
  attribution:'&copy; <a href="https://openstreetmap.org">OpenStreetMap</a> contributors',
  maxZoom:18,
}).addTo(map);

// Layer group for markers (cleared on each update).
let markerLayer = L.layerGroup().addTo(map);
let markerMap = {};  // location_short → marker

// ---- Legend control ----
const legend = L.control({position:'bottomright'});
legend.onAdd = function(){
  const div = L.DomUtil.create('div','legend-box');
  div.innerHTML =
    '<strong>Score</strong><br>'+
    '<i style="background:hsl(120,70%,42%)"></i> Best (low)<br>'+
    '<i style="background:hsl(60,70%,42%)"></i> Mid<br>'+
    '<i style="background:hsl(0,70%,42%)"></i> Worst (high)';
  return div;
};
legend.addTo(map);

// ---- State ----
let currentRows = [];         // top-N rows for the active ranking
let tableSortCol = null;
let tableSortAsc = true;

// ---- Update map markers ----
function updateMap(rankKey){
  markerLayer.clearLayers();
  markerMap = {};
  const cfg = RANKINGS[rankKey];

  // Sort all data by the ranking column, take top N.
  const sorted = DATA.slice().sort((a,b)=>(a[cfg.rankCol]||999)-(b[cfg.rankCol]||999));
  currentRows = sorted.slice(0, TOP_N);

  // Compute score range for color mapping within the displayed set.
  const scores = currentRows.map(r=>r[cfg.scoreCol]).filter(v=>v!==null&&!isNaN(v));
  const sMin = Math.min(...scores);
  const sMax = Math.max(...scores);
  const sRange = sMax-sMin || 1;

  currentRows.forEach(function(r){
    if(r.lat===null||r.lon===null||r.lat===undefined||r.lon===undefined) return;
    const normScore = (r[cfg.scoreCol]-sMin)/sRange;
    const color = getColor(normScore);
    const marker = L.circleMarker([r.lat, r.lon],{
      radius:11, fillColor:color, color:'#333', weight:1.5,
      fillOpacity:0.85, opacity:1,
    }).addTo(markerLayer);

    marker.bindPopup(
      '<div style="font-size:.9em;line-height:1.5">'+
      '<strong>'+r.location_short+'</strong><br>'+
      cfg.label+' Rank: <b>#'+r[cfg.rankCol]+'</b><br>'+
      'Avg RT LMP: <b>'+fmt$(r.avg_lmp_rt)+'</b>/MWh<br>'+
      'Volatility RT: '+fmt$(r.vol_lmp_rt)+'<br>'+
      'Congestion RT: '+fmt$(r.avg_congestion_rt)+'<br>'+
      'Avg Basis: '+fmt$(r.avg_basis)+'<br>'+
      'Score: '+fmt4(r[cfg.scoreCol])+
      '</div>'
    );

    // Click marker → highlight table row.
    marker.on('click', function(){
      highlightTableRow(r.location_short);
    });

    markerMap[r.location_short] = marker;
  });
}

// ---- Update table ----
function updateTable(rankKey){
  const cfg = RANKINGS[rankKey];
  // Reset sort state when ranking changes.
  tableSortCol = cfg.rankCol;
  tableSortAsc = true;
  renderTable();
}

function renderTable(){
  const tbody = document.querySelector('#data-table tbody');
  tbody.innerHTML = '';

  // Sort currentRows by tableSortCol.
  const rows = currentRows.slice();
  rows.sort(function(a,b){
    let va = a[tableSortCol], vb = b[tableSortCol];
    if(va===null||va===undefined) va = Infinity;
    if(vb===null||vb===undefined) vb = Infinity;
    if(typeof va==='string') return tableSortAsc?va.localeCompare(vb):vb.localeCompare(va);
    return tableSortAsc?(va-vb):(vb-va);
  });

  rows.forEach(function(r, idx){
    const tr = document.createElement('tr');
    tr.dataset.loc = r.location_short||'';
    tr.innerHTML =
      '<td>'+(idx+1)+'</td>'+
      '<td>'+((r.location_short||r.location)||'')+'</td>'+
      '<td>'+fmt$(r.avg_lmp_rt)+'</td>'+
      '<td>'+fmt$(r.avg_lmp_da)+'</td>'+
      '<td>'+fmt$(r.avg_congestion_rt)+'</td>'+
      '<td>'+fmt4(r.avg_cong_share_rt)+'</td>'+
      '<td>'+fmt$(r.vol_lmp_rt)+'</td>'+
      '<td>'+fmt$(r.p95_lmp_rt)+'</td>'+
      '<td>'+fmt$(r.avg_basis)+'</td>'+
      '<td>'+fmt$(r.vol_basis)+'</td>'+
      '<td>'+fmt3(r.corr_load_rt)+'</td>'+
      '<td>'+fmtInt(r.count_hours)+'</td>';

    // Hover table row → highlight map marker.
    tr.addEventListener('mouseenter', function(){
      const m = markerMap[r.location_short];
      if(m){ m.setStyle({radius:16, weight:3}); m.bringToFront(); }
    });
    tr.addEventListener('mouseleave', function(){
      const m = markerMap[r.location_short];
      if(m) m.setStyle({radius:11, weight:1.5});
    });

    tbody.appendChild(tr);
  });

  // Update sort arrows.
  document.querySelectorAll('#data-table th .arrow').forEach(function(el){
    el.textContent = '';
    el.classList.remove('active');
  });
  const activeHeader = document.querySelector('#data-table th[data-col="'+tableSortCol+'"] .arrow');
  if(activeHeader){
    activeHeader.textContent = tableSortAsc?' \u25B2':' \u25BC';
    activeHeader.classList.add('active');
  }
}

// ---- Highlight a table row by location_short ----
function highlightTableRow(locShort){
  // Remove previous highlights.
  document.querySelectorAll('#data-table tr.highlighted').forEach(function(el){
    el.classList.remove('highlighted');
  });
  const row = document.querySelector('#data-table tr[data-loc="'+locShort+'"]');
  if(row){
    row.classList.add('highlighted');
    row.scrollIntoView({behavior:'smooth', block:'center'});
    setTimeout(function(){ row.classList.remove('highlighted'); }, 3000);
  }
}

// ---- Column header click → sort ----
document.querySelectorAll('#data-table th').forEach(function(th){
  th.addEventListener('click', function(){
    const col = th.dataset.col;
    if(!col) return;
    if(tableSortCol===col){ tableSortAsc=!tableSortAsc; }
    else{ tableSortCol=col; tableSortAsc=true; }
    renderTable();
  });
});

// ---- Dropdown change handler ----
document.getElementById('ranking-select').addEventListener('change', function(){
  const key = this.value;
  updateMap(key);
  updateTable(key);
});

// ---- Initial render ----
updateMap('cheap_stable');
updateTable('cheap_stable');
</script>
</body>
</html>"""


def _clean_for_json(value):
    """Convert NaN / Infinity to None for valid JSON serialization."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def generate_html(rankings: pd.DataFrame) -> None:
    """
    Generate a self-contained HTML dashboard with an interactive
    Leaflet map and sortable data table.  Writes out/index.html.
    """
    all_data = rankings.copy()

    # Add lat/lon coordinates by mapping location_short through
    # map_location_to_zone() to get the normalized zone key, then
    # looking up in ZONE_COORDS.
    def _get_coord(loc_short, idx):
        zone_key = map_location_to_zone(loc_short)
        coords = ZONE_COORDS.get(zone_key)
        return coords[idx] if coords else None

    all_data["lat"] = all_data["location_short"].apply(lambda x: _get_coord(x, 0))
    all_data["lon"] = all_data["location_short"].apply(lambda x: _get_coord(x, 1))

    # Warn about unmatched locations (table only, no map dot).
    unmatched = all_data[all_data["lat"].isna()]["location"].tolist()
    if unmatched:
        log.warning(
            "No coordinates for: %s (will appear in table but not on map)",
            unmatched,
        )

    # Convert to JSON-safe records (NaN → null).
    records = [
        {k: _clean_for_json(v) for k, v in row.items()}
        for row in all_data.to_dict(orient="records")
    ]
    json_str = json.dumps(records, default=str, indent=2)

    # Build HTML from template using safe token replacement.
    html = HTML_TEMPLATE
    html = html.replace("__DATA_JSON__", json_str)
    html = html.replace("__START_DATE__", START_DATE)
    html = html.replace("__END_DATE__", END_DATE)
    html = html.replace("__N_LOCATIONS__", str(len(all_data)))
    html = html.replace("__LOCATION_MODE__", LOCATION_MODE)
    html = html.replace("__TOP_N__", str(TOP_N))

    # Write to file.
    path = OUT_DIR / "index.html"
    path.write_text(html, encoding="utf-8")
    log.info("Saved interactive dashboard: %s", path)


# ===================================================================
# SECTION 9 – EXECUTIVE SUMMARY
# ===================================================================

def print_executive_summary(
    summary: pd.DataFrame,
    rankings: pd.DataFrame,
) -> None:
    """Print a formatted executive summary to the console."""
    sep = "=" * 78
    print(f"\n{sep}")
    print("PJM DATA CENTER SITING ANALYSIS — EXECUTIVE SUMMARY")
    print(f"Date range : {START_DATE}  to  {END_DATE}")
    print(f"Mode       : {LOCATION_MODE}")
    print(f"Locations  : {len(summary)}")
    print(sep)

    # Helper to format a ranking table.
    def _print_ranking(title, rank_col, score_col, metric_cols, headers):
        print(f"\n--- {title} ---")
        top = rankings.nsmallest(TOP_N, rank_col)
        # Print header.
        row_fmt = "  {:<5s}  {:<28s}" + "  {:>12s}" * len(headers)
        print(row_fmt.format("Rank", "Location", *headers))
        print("  " + "-" * (5 + 2 + 28 + (12 + 2) * len(headers)))
        for _, r in top.iterrows():
            vals = [f"{r[mc]:.4f}" if isinstance(r[mc], float) else str(r[mc]) for mc in metric_cols]
            print(row_fmt.format(
                str(int(r[rank_col])),
                str(r["location"])[:28],
                *vals,
            ))

    _print_ranking(
        "Top Locations: Cheap + Stable (low avg RT LMP, low volatility)",
        "cheap_stable_rank",
        "cheap_stable_score",
        ["avg_lmp_rt", "vol_lmp_rt", "cheap_stable_score"],
        ["Avg RT LMP", "Vol RT LMP", "Score"],
    )

    _print_ranking(
        "Top Locations: Low Congestion Risk",
        "low_congestion_rank",
        "low_congestion_score",
        ["avg_congestion_rt", "avg_cong_share_rt", "low_congestion_score"],
        ["Avg Cong RT", "Avg CShare", "Score"],
    )

    _print_ranking(
        "Top Locations: Low Basis Risk (DA-RT)",
        "low_basis_rank",
        "low_basis_score",
        ["avg_basis", "vol_basis", "low_basis_score"],
        ["Avg Basis", "Vol Basis", "Score"],
    )

    # Data coverage warnings.
    if "count_hours" in summary.columns:
        max_hours = summary["count_hours"].max()
        threshold = 0.9 * max_hours
        low_coverage = summary[summary["count_hours"] < threshold]
        if not low_coverage.empty:
            print(f"\n--- Data Coverage Warnings (<90% of max {max_hours} hours) ---")
            for _, r in low_coverage.iterrows():
                pct = 100.0 * r["count_hours"] / max(max_hours, 1)
                print(f"  {r['location']:<28s}  {int(r['count_hours']):>6d} hours  ({pct:.1f}%)")

    # Notable high-congestion locations.
    if "avg_congestion_rt" in rankings.columns:
        # Sort by absolute congestion to find the most impacted locations.
        high_cong = rankings.reindex(
            rankings["avg_congestion_rt"].abs().nlargest(3).index
        )
        if not high_cong.empty:
            print("\n--- Notable High Congestion Locations ---")
            for _, r in high_cong.iterrows():
                print(
                    f"  {r['location']:<28s}  "
                    f"Avg Cong RT = ${r['avg_congestion_rt']:>8.2f}  "
                    f"Share = {r['avg_cong_share_rt']:.4f}"
                )

    print(f"\nOutput files: {OUT_DIR}/")
    print(f"Interactive dashboard: {OUT_DIR / 'index.html'}")
    print(sep)
    print()


# ===================================================================
# SECTION 10 – MAIN ENTRY POINT
# ===================================================================

def main() -> None:
    """Run the full PJM siting analysis pipeline."""
    log.info("Starting PJM siting analysis")
    log.info("Date range: %s to %s", START_DATE, END_DATE)
    log.info("Location mode: %s", LOCATION_MODE)

    # Step 0: Create output directory.
    ensure_output_dir()

    # Step 1: Pull data.
    log.info("=== STEP 1: Data Pull ===")
    df_da, df_rt, df_load = fetch_all_data()

    # Step 2: Merge and align.
    log.info("=== STEP 2: Cleaning & Alignment ===")
    df_merged = merge_and_align(df_da, df_rt, df_load)

    # Step 3: Summary metrics.
    log.info("=== STEP 3: Summary Metrics ===")
    summary = compute_summary(df_merged)

    # Step 4: Rankings.
    log.info("=== STEP 4: Rankings ===")
    rankings = compute_rankings(summary)

    # Step 5: Generate interactive HTML dashboard.
    log.info("=== STEP 5: Generating Interactive Dashboard ===")
    try:
        generate_html(rankings)
    except Exception as exc:
        log.warning("HTML dashboard generation failed: %s", exc)

    # Step 6: Executive summary.
    log.info("=== STEP 6: Executive Summary ===")
    print_executive_summary(summary, rankings)

    log.info("Analysis complete.  All outputs in %s", OUT_DIR)


if __name__ == "__main__":
    main()


# ===================================================================
# HOW TO RUN
# ===================================================================
#
# 1. Install dependencies:
#        pip install -r requirements.txt
#
# 2. Set your PJM API key (register at https://www.pjm.com/):
#        export PJM_API_KEY="your-api-key-here"
#
# 3. (Optional) Edit the CONFIGURATION section at the top of this
#    script to set START_DATE, END_DATE, LOCATION_MODE, etc.
#
# 4. Run the script:
#        python pjm_siting_analysis.py
#
# 5. Outputs are written to ./out/:
#        out/lmp_da_raw.csv           Raw day-ahead LMP data
#        out/lmp_rt_raw.csv           Raw real-time LMP data
#        out/load_raw.csv             Hourly load (long format)
#        out/lmp_merged.csv           Aligned DA+RT with basis & load
#        out/location_summary.csv     Per-location summary metrics
#        out/rankings.csv             Summary + composite scores & ranks
#        out/index.html               Interactive map + data table dashboard
#
# 6. Open out/index.html in a web browser to view the interactive
#    dashboard with Leaflet map and sortable data table.
#
# 7. A textual executive summary is also printed to the console.
