"""
PJM Data Center Siting Analysis
================================
Pulls Day-Ahead and Real-Time LMP data plus system/zonal load from PJM
via the gridstatus library, computes location-level summary metrics,
ranks locations on cost / stability / congestion / basis criteria, and
produces CSV exports and diagnostic plots.

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

Outputs are written to ./out/ (CSVs, PNGs, and console executive summary).

Timezone Handling
-----------------
gridstatus returns PJM timestamps in US/Eastern (EPT).  All internal
processing preserves this timezone.  Plot axes are labelled as ET.
If timestamps arrive timezone-naive, they are localized to US/Eastern.
"""

import os
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (server / CI safe)
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402

import gridstatus  # noqa: E402
from gridstatus import Markets  # noqa: E402

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

# Resample frequency for time-series plots ("D" = daily, "W" = weekly).
# Summary statistics always use the native hourly granularity.
RESAMPLE_FREQ: str = "D"

# Output directory (relative to this script).
OUT_DIR: Path = Path(__file__).resolve().parent / "out"

# Number of top locations to show in time-series plots.
TOP_N: int = 5


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
# SECTION 8 – PLOTS
# ===================================================================

def plot_daily_rt_lmp(
    df_merged: pd.DataFrame,
    rankings: pd.DataFrame,
) -> None:
    """
    Daily average RT LMP time series for the TOP_N cheapest+stable
    locations.  Saves plot_daily_rt_lmp.png.
    """
    top_locs = (
        rankings.nsmallest(TOP_N, "cheap_stable_rank")["location"].tolist()
    )
    subset = df_merged[df_merged["location"].isin(top_locs)].copy()
    if subset.empty:
        log.warning("plot_daily_rt_lmp: no data for top locations.")
        return

    subset["time"] = pd.to_datetime(subset["time"])
    daily = (
        subset
        .set_index("time")
        .groupby("location")["lmp_total_rt"]
        .resample(RESAMPLE_FREQ)
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    for loc in top_locs:
        loc_data = daily[daily["location"] == loc]
        ax.plot(loc_data["time"], loc_data["lmp_total_rt"], label=loc, linewidth=1.2)

    ax.set_xlabel("Date (ET)")
    ax.set_ylabel("RT LMP ($/MWh)")
    ax.set_title(f"Daily Avg RT LMP — Top {TOP_N} Cheapest + Stable Locations")
    ax.legend(loc="upper right", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=30)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = OUT_DIR / "plot_daily_rt_lmp.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)


def plot_lmp_histogram(
    df_merged: pd.DataFrame,
    location_name: str,
) -> None:
    """
    Histogram of hourly RT LMP for a single location.
    Saves plot_lmp_histogram.png.
    """
    data = df_merged.loc[
        df_merged["location"] == location_name, "lmp_total_rt"
    ].dropna()

    if data.empty:
        log.warning("plot_lmp_histogram: no RT data for '%s'.", location_name)
        return

    mean_val = data.mean()
    p95_val = data.quantile(0.95)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(data, bins=80, edgecolor="black", linewidth=0.4, alpha=0.7, color="steelblue")
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.2, label=f"Mean = ${mean_val:.2f}")
    ax.axvline(p95_val, color="orange", linestyle="--", linewidth=1.2, label=f"P95 = ${p95_val:.2f}")

    ax.set_xlabel("RT LMP ($/MWh)")
    ax.set_ylabel("Hour Count")
    ax.set_title(f"RT LMP Distribution — {location_name}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = OUT_DIR / "plot_lmp_histogram.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)


def plot_lmp_vs_load(
    df_merged: pd.DataFrame,
    location_name: str,
) -> None:
    """
    Scatter plot of daily avg RT LMP vs daily avg load for one location.
    Annotates with Pearson correlation.  Saves plot_lmp_vs_load.png.
    """
    subset = df_merged[df_merged["location"] == location_name].copy()
    subset = subset[["time", "lmp_total_rt", "load_mw"]].dropna()

    if len(subset) < 5:
        log.warning("plot_lmp_vs_load: insufficient data for '%s'.", location_name)
        return

    subset["time"] = pd.to_datetime(subset["time"])
    daily = subset.set_index("time").resample(RESAMPLE_FREQ).mean().dropna()

    if len(daily) < 3:
        log.warning("plot_lmp_vs_load: insufficient daily data for '%s'.", location_name)
        return

    load = daily["load_mw"].values
    lmp = daily["lmp_total_rt"].values
    corr = np.corrcoef(load, lmp)[0, 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(load, lmp, alpha=0.5, s=18, color="teal", edgecolors="none")

    # OLS trend line.
    coeffs = np.polyfit(load, lmp, 1)
    trend_x = np.linspace(load.min(), load.max(), 100)
    trend_y = np.polyval(coeffs, trend_x)
    ax.plot(trend_x, trend_y, color="tomato", linewidth=1.5, linestyle="-")

    ax.annotate(
        f"r = {corr:.3f}",
        xy=(0.05, 0.92),
        xycoords="axes fraction",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray"),
    )

    ax.set_xlabel("Load (MW)")
    ax.set_ylabel("RT LMP ($/MWh)")
    ax.set_title(f"RT LMP vs Load (Daily Avg) — {location_name}")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = OUT_DIR / "plot_lmp_vs_load.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)


def plot_basis_timeseries(
    df_merged: pd.DataFrame,
    rankings: pd.DataFrame,
) -> None:
    """
    Daily average DA-RT basis for the TOP_N locations with lowest
    basis risk.  Saves plot_basis_timeseries.png.
    """
    top_locs = (
        rankings.nsmallest(TOP_N, "low_basis_rank")["location"].tolist()
    )
    subset = df_merged[df_merged["location"].isin(top_locs)].copy()
    if subset.empty or "basis" not in subset.columns:
        log.warning("plot_basis_timeseries: no basis data for top locations.")
        return

    subset["time"] = pd.to_datetime(subset["time"])
    daily = (
        subset
        .set_index("time")
        .groupby("location")["basis"]
        .resample(RESAMPLE_FREQ)
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    for loc in top_locs:
        loc_data = daily[daily["location"] == loc]
        ax.plot(loc_data["time"], loc_data["basis"], label=loc, linewidth=1.2)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xlabel("Date (ET)")
    ax.set_ylabel("DA − RT Basis ($/MWh)")
    ax.set_title(f"Daily DA-RT Basis — Top {TOP_N} Low-Basis-Risk Locations")
    ax.legend(loc="upper right", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=30)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = OUT_DIR / "plot_basis_timeseries.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)


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

    # Step 5: Plots – each in its own try/except so one failure
    # does not block the others.
    log.info("=== STEP 5: Generating Plots ===")
    best_loc = rankings.nsmallest(1, "cheap_stable_rank")["location"].iloc[0]
    log.info("Best location for single-location plots: %s", best_loc)

    plot_jobs = [
        (plot_daily_rt_lmp, (df_merged, rankings)),
        (plot_lmp_histogram, (df_merged, best_loc)),
        (plot_lmp_vs_load, (df_merged, best_loc)),
        (plot_basis_timeseries, (df_merged, rankings)),
    ]
    for fn, args in plot_jobs:
        try:
            fn(*args)
        except Exception as exc:
            log.warning("Plot %s failed: %s", fn.__name__, exc)

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
#        out/plot_daily_rt_lmp.png    Time series: top 5 cheapest
#        out/plot_lmp_histogram.png   Histogram: best location
#        out/plot_lmp_vs_load.png     Scatter: LMP vs load
#        out/plot_basis_timeseries.png  Basis over time: top 5
#
# 6. A textual executive summary is printed to the console.
