import os
import json
from datetime import timedelta, date
from typing import Dict, List

import numpy as np
import pandas as pd


DATA_ROOT_HELP = '''
Expected structure (relative to --data-root):
  data/kernel/receivals.csv
  data/kernel/purchase_orders.csv
  data/extended/materials.csv             (optional)
  data/extended/transportation.csv        (optional)
'''

REQUIRED_PO_COLS = {"product_id", "delivery_date", "quantity"}
REQUIRED_REC_COLS = {"rm_id", "date_arrival", "net_weight"}


def load_receivals(data_root: str) -> pd.DataFrame:
    path = os.path.join(data_root, "data", "kernel", "receivals.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find receivals.csv at: {path}\n{DATA_ROOT_HELP}")
    df = pd.read_csv(path, low_memory=False)
    missing = REQUIRED_REC_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in receivals.csv: {missing}")
    df = df.copy()
    df["date_arrival"] = pd.to_datetime(df["date_arrival"], utc=True, errors="coerce")
    df["net_weight"] = pd.to_numeric(df["net_weight"], errors="coerce")
    df["rm_id"] = df["rm_id"].astype(str)
    df = df.dropna(subset=["rm_id", "date_arrival", "net_weight"])
    df["date"] = df["date_arrival"].dt.tz_convert("UTC").dt.date
    df["year"] = pd.to_datetime(df["date"]).dt.year
    return df[["rm_id", "date", "year", "net_weight"]]


def load_purchase_orders(data_root: str) -> pd.DataFrame:
    path = os.path.join(data_root, "data", "kernel", "purchase_orders.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find purchase_orders.csv at: {path}\n{DATA_ROOT_HELP}")
    df = pd.read_csv(path, low_memory=False)
    missing = REQUIRED_PO_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in purchase_orders.csv: {missing}")
    df = df.copy()
    df["delivery_date"] = pd.to_datetime(df["delivery_date"], utc=True, errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["product_id"] = df["product_id"].astype(str)
    df = df.dropna(subset=["product_id", "delivery_date", "quantity"])
    df["date"] = df["delivery_date"].dt.tz_convert("UTC").dt.date
    df["year"] = pd.to_datetime(df["date"]).dt.year
    return df[["product_id", "date", "year", "quantity"]]


def is_month_end(d: date) -> bool:
    next_day = d + timedelta(days=1)
    return next_day.day == 1  # rolled over to next month


def po_placeholder_stats(po: pd.DataFrame) -> Dict[str, float]:
    if po.empty:
        return {"n_pos": 0, "share_month_end": 0.0, "share_dec31": 0.0}
    month_end = po["date"].apply(is_month_end)
    dec31 = po["date"].apply(lambda d: (d.month == 12 and d.day == 31))
    return {
        "n_pos": int(len(po)),
        "share_month_end": float(np.mean(month_end)),
        "share_dec31": float(np.mean(dec31)),
    }


def build_daily_receivals(rec: pd.DataFrame, start_month_day=(1,1), end_month_day=(5,31)) -> pd.DataFrame:
    """Aggregate receivals to daily per rm_id, Jan-May per year, filling missing days with 0."""
    if rec.empty:
        return pd.DataFrame(columns=["rm_id", "year", "date", "daily_kg"])

    years = sorted(rec["year"].unique())
    out = []
    for y in years:
        start_d = date(y, start_month_day[0], start_month_day[1])
        end_d = date(y, end_month_day[0], end_month_day[1])
        r = rec[(rec["date"] >= start_d) & (rec["date"] <= end_d) & (rec["year"] == y)]
        g = (r.groupby(["rm_id", "date"], as_index=False)["net_weight"]
               .sum()
               .rename(columns={"net_weight": "daily_kg"}))
        rm_ids = g["rm_id"].unique().tolist()
        full_index = pd.date_range(start=start_d, end=end_d, freq="D")
        for rm in rm_ids:
            sub = g.loc[g["rm_id"] == rm].set_index("date").reindex(full_index)
            sub = sub.rename_axis("date").reset_index()
            sub["rm_id"] = rm
            sub["daily_kg"] = sub["daily_kg"].fillna(0.0)
            sub["year"] = y
            out.append(sub)

    if not out:
        return pd.DataFrame(columns=["rm_id", "year", "date", "daily_kg"])
    df = pd.concat(out, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df[["rm_id", "year", "date", "daily_kg"]]


def heuristic_po_to_rm_mapping(po: pd.DataFrame, daily_rec: pd.DataFrame, window_days: int = 7) -> pd.DataFrame:
    """
    Infer a product_id -> rm_id mapping by co-occurrence:
    For each product_id, score each rm_id by how much receivals (kg) arrive within +/- window_days of PO delivery_date.
    """
    if po.empty or daily_rec.empty:
        return pd.DataFrame(columns=[
            "product_id","rm_id","score_kg","po_count","po_total_qty","rec_matched_kg",
            "mean_match_kg_per_po","median_match_kg_per_po"
        ])

    daily_rec = daily_rec.copy()
    daily_rec["date"] = pd.to_datetime(daily_rec["date"]).dt.date
    rm_groups = {rm: grp.set_index("date") for rm, grp in daily_rec.groupby("rm_id")}

    rows = []
    for prod, group in po.groupby("product_id"):
        prod_po = group.sort_values("date")
        po_total_qty = float(prod_po["quantity"].sum())
        po_count = int(len(prod_po))

        # score each rm by sum receivals around each PO date
        best_rm, best_sum, best_list = None, -1.0, None
        for rm, recg in rm_groups.items():
            matched_kg = []
            for d in prod_po["date"]:
                start = d - timedelta(days=window_days)
                end = d + timedelta(days=window_days)
                window_dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]
                s = recg.reindex(window_dates)["daily_kg"].fillna(0.0).sum()
                matched_kg.append(float(s))
            total = float(np.sum(matched_kg))
            if total > best_sum:
                best_sum, best_rm, best_list = total, rm, matched_kg

        if best_rm is not None:
            rows.append({
                "product_id": prod,
                "rm_id": best_rm,
                "score_kg": float(best_sum),
                "po_count": po_count,
                "po_total_qty": po_total_qty,
                "rec_matched_kg": float(np.sum(best_list) if best_list else 0.0),
                "mean_match_kg_per_po": float(np.mean(best_list) if best_list else 0.0),
                "median_match_kg_per_po": float(np.median(best_list) if best_list else 0.0),
            })

    out = pd.DataFrame(rows)
    out = out.sort_values(["score_kg", "po_total_qty"], ascending=[False, False]).reset_index(drop=True)
    return out


def backtest_2024_coverage(mapping: pd.DataFrame, po: pd.DataFrame, daily_rec: pd.DataFrame, window_days: int = 7) -> Dict[str, float]:
    """
    Using inferred product_id->rm_id mapping, estimate how much of 2024 receivals (Jan-May) are 'explained' by POs.
    We compute the kg of receivals that fall within +/- window_days of PO dates for the mapped rm_id.
    """
    if mapping.empty or po.empty or daily_rec.empty:
        return {"receivals_2024_jan_may_kg": float("nan"),
                "po_explained_kg": float("nan"),
                "share_explained": float("nan"),
                "n_products_mapped": 0}

    po_2024 = po[po["year"] == 2024]
    rec_2024 = daily_rec[daily_rec["year"] == 2024]
    if po_2024.empty or rec_2024.empty:
        total_rec_kg = float(rec_2024["daily_kg"].sum() if not rec_2024.empty else 0.0)
        return {"receivals_2024_jan_may_kg": total_rec_kg,
                "po_explained_kg": 0.0,
                "share_explained": 0.0,
                "n_products_mapped": int(mapping["product_id"].nunique() if not mapping.empty else 0)}

    rec_by_rm = {rm: grp.set_index("date") for rm, grp in rec_2024.groupby("rm_id")}

    explained_kg = 0.0
    for _, row in mapping.iterrows():
        prod = row["product_id"]; rm = row["rm_id"]
        grp = po_2024[po_2024["product_id"] == prod]
        if grp.empty: 
            continue
        recg = rec_by_rm.get(rm)
        if recg is None:
            continue
        for d in grp["date"]:
            start = d - timedelta(days=window_days); end = d + timedelta(days=window_days)
            window_dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]
            s = recg.reindex(window_dates)["daily_kg"].fillna(0.0).sum()
            explained_kg += float(s)

    total_rec_kg = float(rec_2024["daily_kg"].sum())
    share = (explained_kg / total_rec_kg) if total_rec_kg > 0 else np.nan
    return {"receivals_2024_jan_may_kg": total_rec_kg,
            "po_explained_kg": explained_kg,
            "share_explained": share,
            "n_products_mapped": int(mapping["product_id"].nunique())}


def run(data_root: str, window_days: int = 7) -> Dict[str, str]:
    rec = load_receivals(data_root)
    po = load_purchase_orders(data_root)

    placeholder = po_placeholder_stats(po)
    with open("po_placeholder_stats.json", "w") as f:
        json.dump(placeholder, f, indent=2)

    # build mapping on <=2023 to avoid leakage
    po_hist = po[po["year"] <= 2023]
    rec_hist = rec[rec["year"] <= 2023]
    daily_hist = build_daily_receivals(rec_hist, (1,1), (5,31))

    mapping = heuristic_po_to_rm_mapping(po_hist, daily_hist, window_days=window_days)
    mapping.to_csv("po_to_rm_mapping.csv", index=False)

    # coverage on 2024
    daily_all = build_daily_receivals(rec, (1,1), (5,31))
    cov = backtest_2024_coverage(mapping, po, daily_all, window_days=window_days)
    with open("po_linkage_backtest_2024.json", "w") as f:
        json.dump(cov, f, indent=2)

    return {"placeholder_stats": os.path.abspath("po_placeholder_stats.json"),
            "mapping_csv": os.path.abspath("po_to_rm_mapping.csv"),
            "coverage_2024": os.path.abspath("po_linkage_backtest_2024.json")}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Heuristic linkage between purchase orders and receivals")
    parser.add_argument("--data-root", type=str, default=".", help="Path to dataset root (contains 'data/' folder)")
    parser.add_argument("--window-days", type=int, default=7, help="Days on each side of a PO delivery date to attribute receivals")
    args = parser.parse_args()
    try:
        outputs = run(args.data_root, window_days=args.window_days)
        print("Placeholder stats:", outputs["placeholder_stats"])
        print("PO->RM mapping CSV:", outputs["mapping_csv"])
        print("Backtest coverage 2024:", outputs["coverage_2024"])
    except Exception as e:
        print("ERROR:", str(e))
