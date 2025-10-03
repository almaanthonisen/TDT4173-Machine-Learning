
import os
import json
from datetime import date
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd


DATA_ROOT_HELP = """
Expected structure (relative to --data-root):
  data/kernel/receivals.csv
  data/kernel/purchase_orders.csv         (optional for this baseline)
  data/extended/materials.csv             (optional)
  data/extended/transportation.csv        (optional)
"""


def quantile_loss_0p2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Quantile loss at tau=0.2 (pinball loss) for arrays of the same shape.
    Loss = max(0.2 * (y_true - y_pred), 0.8 * (y_pred - y_true))
    Returns the mean over all elements.
    """
    diff = y_true - y_pred
    loss = np.maximum(0.2 * diff, 0.8 * (-diff))
    return float(np.mean(loss))


def load_receivals(data_root: str) -> pd.DataFrame:
    """
    Load receivals.csv and parse dates. Only columns required for this baseline:
      - rm_id
      - date_arrival (UTC timestamp)
      - net_weight (kg)
    """
    path = os.path.join(data_root, "data", "kernel", "receivals.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find receivals.csv at: {path}\n{DATA_ROOT_HELP}"
        )
    df = pd.read_csv(path, low_memory=False)
    expected_cols = {"rm_id", "date_arrival", "net_weight"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in receivals.csv: {missing}")
    # Coerce types
    df = df.copy()
    df["date_arrival"] = pd.to_datetime(df["date_arrival"], utc=True, errors="coerce")
    df["net_weight"] = pd.to_numeric(df["net_weight"], errors="coerce")
    # Drop rows with missing critical fields
    df = df.dropna(subset=["rm_id", "date_arrival", "net_weight"])
    # Ensure rm_id is a string (IDs sometimes numeric in CSVs)
    df["rm_id"] = df["rm_id"].astype(str)
    # Also store date (no time) in UTC calendar for grouping
    df["date"] = df["date_arrival"].dt.tz_convert("UTC").dt.date
    return df[["rm_id", "date", "net_weight"]]


def make_daily_series_for_window(
    df_receivals: pd.DataFrame,
    start_month_day: Tuple[int, int] = (1, 1),
    end_month_day: Tuple[int, int] = (5, 31),
) -> pd.DataFrame:
    """
    Build a per-year, per-rm_id DAILY dataframe (Jan 1–May 31) with columns:
      rm_id, year, date, daily_net_weight
    Missing dates are filled with 0 per rm_id per year.
    """
    assert {"rm_id", "date", "net_weight"} <= set(df_receivals.columns)

    df = df_receivals.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year

    years = sorted(df["year"].unique())

    records = []
    for y in years:
        start_d = date(y, start_month_day[0], start_month_day[1])
        end_d = date(y, end_month_day[0], end_month_day[1])
        mask = (df["date"] >= start_d) & (df["date"] <= end_d) & (df["year"] == y)
        df_y = df.loc[mask]

        daily = (
            df_y.groupby(["rm_id", "date"], as_index=False)["net_weight"]
            .sum()
            .rename(columns={"net_weight": "daily_net_weight"})
        )

        rm_ids_y = daily["rm_id"].unique().tolist()
        full_index = pd.date_range(start=start_d, end=end_d, freq="D")
        for rm in rm_ids_y:
            sub = daily.loc[daily["rm_id"] == rm].set_index("date").reindex(full_index)
            sub = sub.rename_axis("date").reset_index()
            sub["rm_id"] = rm
            sub["daily_net_weight"] = sub["daily_net_weight"].fillna(0.0)
            sub["year"] = y
            records.append(sub)

    if not records:
        return pd.DataFrame(columns=["rm_id", "year", "date", "daily_net_weight"])

    out = pd.concat(records, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.date
    return out[["rm_id", "year", "date", "daily_net_weight"]]


def cumulative_by_year_window(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    From the DAILY dataframe, compute cumulative sums per (rm_id, year) ordered by date.
    Returns columns: rm_id, year, date, day_index (1..151), cum_net_weight
    """
    if daily_df.empty:
        return pd.DataFrame(
            columns=["rm_id", "year", "date", "day_index", "cum_net_weight"]
        )
    df = daily_df.copy()
    df = df.sort_values(["rm_id", "year", "date"])
    df["day_index"] = df.groupby(["rm_id", "year"]).cumcount() + 1
    df["cum_net_weight"] = df.groupby(["rm_id", "year"])["daily_net_weight"].cumsum()
    return df[["rm_id", "year", "date", "day_index", "cum_net_weight"]]


def q20_cumulative_profile(
    cum_df: pd.DataFrame,
    years_for_profile: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Compute the q=0.2 cumulative profile per rm_id for each day_index (1..151),
    using the given set of years_for_profile. If None, use all years in cum_df.
    Returns: rm_id, day_index, q20_cum
    """
    if cum_df.empty:
        return pd.DataFrame(columns=["rm_id", "day_index", "q20_cum"])

    df = cum_df.copy()
    if years_for_profile is not None:
        df = df[df["year"].isin(years_for_profile)]

    grouped = df.groupby(["rm_id", "day_index"])["cum_net_weight"]
    q20 = grouped.quantile(0.2).reset_index().rename(columns={"cum_net_weight": "q20_cum"})
    return q20[["rm_id", "day_index", "q20_cum"]]


def build_2025_forecast_from_q20(
    q20_df: pd.DataFrame,
    start: date = date(2025, 1, 1),
    end: date = date(2025, 5, 31),
) -> pd.DataFrame:
    """
    Turn q20 cumulative profile (per day_index) into dated 2025 forecasts.
    Output columns: rm_id, end_date, forecast_cum_kg
    """
    if q20_df.empty:
        return pd.DataFrame(columns=["rm_id", "end_date", "forecast_cum_kg"])

    date_index = pd.date_range(start=start, end=end, freq="D")
    mapping = pd.DataFrame({"end_date": date_index.date, "day_index": np.arange(1, len(date_index) + 1)})
    out = q20_df.merge(mapping, on="day_index", how="inner")
    out = out[["rm_id", "end_date", "q20_cum"]].rename(columns={"q20_cum": "forecast_cum_kg"})
    return out.sort_values(["rm_id", "end_date"]).reset_index(drop=True)


def backtest_2024_quantile_error(
    cum_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Backtest: use years <= 2023 to forecast 2024 cumulative (Jan–May),
    evaluate QuantileError@0.2 per day across rm_ids, then average.
    """
    if cum_df.empty:
        return {
            "mean_quantile_error_2024": float("nan"),
            "median_quantile_error_2024": float("nan"),
            "n_rm_ids_2024": 0,
            "n_days": 0,
        }

    years = sorted(cum_df["year"].unique())
    if 2024 not in years:
        return {
            "mean_quantile_error_2024": float("nan"),
            "median_quantile_error_2024": float("nan"),
            "n_rm_ids_2024": 0,
            "n_days": int(cum_df["day_index"].max() if not cum_df.empty else 0),
        }

    years_profile = [y for y in years if y <= 2023]
    if not years_profile:
        return {
            "mean_quantile_error_2024": float("nan"),
            "median_quantile_error_2024": float("nan"),
            "n_rm_ids_2024": int(cum_df.loc[cum_df["year"] == 2024, "rm_id"].nunique()),
            "n_days": int(cum_df["day_index"].max() if not cum_df.empty else 0),
        }

    q20 = q20_cumulative_profile(cum_df, np.array(years_profile))

    actual_2024 = (
        cum_df.loc[cum_df["year"] == 2024, ["rm_id", "day_index", "cum_net_weight"]]
        .rename(columns={"cum_net_weight": "actual_cum_kg"})
    )

    comp = actual_2024.merge(q20, on=["rm_id", "day_index"], how="inner")
    if comp.empty:
        return {
            "mean_quantile_error_2024": float("nan"),
            "median_quantile_error_2024": float("nan"),
            "n_rm_ids_2024": 0,
            "n_days": int(cum_df["day_index"].max() if not cum_df.empty else 0),
        }

    comp["qe_0p2"] = np.maximum(
        0.2 * (comp["actual_cum_kg"] - comp["q20_cum"]),
        0.8 * (comp["q20_cum"] - comp["actual_cum_kg"]),
    )

    summary = {
        "mean_quantile_error_2024": float(comp["qe_0p2"].mean()),
        "median_quantile_error_2024": float(comp["qe_0p2"].median()),
        "n_rm_ids_2024": int(actual_2024["rm_id"].nunique()),
        "n_days": int(comp["day_index"].max() if not comp.empty else 0),
    }
    return summary


def run_pipeline(data_root: str) -> Dict[str, str]:
    """
    Orchestrate:
      1) Load receivals
      2) Build daily and cumulative frames (Jan–May for each year)
      3) Backtest 2024 using <=2023 data
      4) Build 2025 forecast and export CSV
    """
    df_receivals = load_receivals(data_root)
    daily = make_daily_series_for_window(df_receivals, (1, 1), (5, 31))
    cumdf = cumulative_by_year_window(daily)

    # Backtest
    backtest_summary = backtest_2024_quantile_error(cumdf)
    backtest_path = "baseline_q20_backtest_2024_summary.json"
    with open(backtest_path, "w") as f:
        json.dump(backtest_summary, f, indent=2)

    # Forecast 2025 from q20 profile over all available years (incl. 2024)
    q20 = q20_cumulative_profile(cumdf, years_for_profile=None)
    forecast_2025 = build_2025_forecast_from_q20(q20)
    forecast_path = "baseline_q20_forecast_2025.csv"
    forecast_2025.to_csv(forecast_path, index=False)

    return {
        "backtest_summary_file": os.path.abspath(backtest_path),
        "forecast_file": os.path.abspath(forecast_path),
        "n_rm_ids": str(int(cumdf["rm_id"].nunique())) if not cumdf.empty else "0",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TDT4173 baseline q0.2 cumulative forecasting")
    parser.add_argument("--data-root", type=str, default=".", help="Path to dataset root (contains 'data/' folder)")
    args = parser.parse_args()

    try:
        outputs = run_pipeline(args.data_root)
        print("Backtest summary written to:", outputs["backtest_summary_file"])
        print("Forecast CSV written to:", outputs["forecast_file"])
        print("Distinct rm_ids seen:", outputs["n_rm_ids"])
    except Exception as e:
        print("ERROR:", str(e))
