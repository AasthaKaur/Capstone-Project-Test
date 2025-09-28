# --- Treasury Auctions Dashboard (CSV version) ---
# Author: Group 11
# Usage: streamlit run app.py

import os
import io
from datetime import date
from dateutil import parser as dateparser

import streamlit as st
import pandas as pd
import altair as alt

# -----------------------------
# CONFIG (edit paths if needed)
# -----------------------------
DATA_DIR = "data"
FILES = {
    "auctions": "auctions_202509052005.csv",
    "bidder": "bidder_details_202509052006.csv",
    "securities": "securities_202509052006.csv",
    "updates": "data_updates_202509052006.csv",
}

st.set_page_config(page_title="U.S. Treasury Auctions", layout="wide")


# -----------------------------
# Helpers
# -----------------------------
def _coerce_date(x):
    if pd.isna(x):
        return pd.NaT
    if isinstance(x, (pd.Timestamp,)):
        return x
    try:
        return pd.to_datetime(x, errors="coerce")
    except Exception:
        try:
            return pd.to_datetime(dateparser.parse(str(x)))
        except Exception:
            return pd.NaT


def _coerce_num(series):
    # handles $1,234.56, percents, blanks
    return (
        series.astype(str)
        .str.replace(r"[,\$%]", "", regex=True)
        .replace({"": None, "None": None, "nan": None})
        .astype(float)
    )


@st.cache_data(show_spinner=False)
def load_csvs(data_dir: str, files: dict):
    paths = {k: os.path.join(data_dir, v) for k, v in files.items()}
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        st.warning(
            "One or more default files were not found. Use the **Upload CSVs** section in the sidebar."
        )
        return None

    a = pd.read_csv(paths["auctions"])
    b = pd.read_csv(paths["bidder"])
    s = pd.read_csv(paths["securities"])
    u = pd.read_csv(paths["updates"])

    # Coerce types
    for col in ["announcement_date", "auction_date", "issue_date", "maturity_date", "dated_date"]:
        if col in a.columns:
            a[col] = a[col].apply(_coerce_date)

    # numeric fields commonly present
    for col in [
        "offering_amount",
        "total_tendered",
        "total_accepted",
        "bid_to_cover_ratio",
        "high_yield",
        "low_yield",
        "average_median_yield",
        "high_price",
        "low_price",
        "price_per_100",
    ]:
        if col in a.columns:
            a[col] = _coerce_num(a[col])

    # bidder %
    for col in [
        "primary_dealer_percentage",
        "direct_bidder_percentage",
        "indirect_bidder_percentage",
        "primary_dealer_accepted",
        "direct_bidder_accepted",
        "indirect_bidder_accepted",
    ]:
        if col in b.columns:
            b[col] = _coerce_num(b[col])

    # securities flags
    for col in ["tips", "floating_rate", "callable"]:
        if col in s.columns:
            s[col] = s[col].astype(bool, errors="ignore")

    # updates timestamps
    for col in ["update_timestamp", "created_at", "updated_at", "last_auction_date"]:
        if col in u.columns:
            u[col] = u[col].apply(_coerce_date)

    return {"auctions": a, "bidder": b, "securities": s, "updates": u}


def join_frames(frames):
    a, b, s = frames["auctions"].copy(), frames["bidder"].copy(), frames["securities"].copy()
    if "cusip" in a.columns and "cusip" in s.columns:
        a = a.merge(s, on="cusip", how="left", suffixes=("", "_sec"))
    if "auction_id" in a.columns and "auction_id" in b.columns:
        a = a.merge(b, on="auction_id", how="left", suffixes=("", "_bid"))
    return a


def kpi_block(df):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Auctions", f"{len(df):,}")
    if "bid_to_cover_ratio" in df.columns:
        c2.metric("Avg Bid-to-Cover", f"{df['bid_to_cover_ratio'].mean():.2f}")
    if "offering_amount" in df.columns:
        c3.metric("Total Offered (USD)", f"{df['offering_amount'].sum():,.0f}")
    if "high_yield" in df.columns:
        c4.metric("Avg High Yield", f"{df['high_yield'].mean():.3f}")


def alt_timeseries(df, x, y, title):
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X(x, title="Date"),
            y=alt.Y(y, title=title),
            tooltip=[x, y],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)


def alt_bar(df, x, y, title):
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(x, title=x),
            y=alt.Y(y, title=title),
            tooltip=[x, y],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)


# -----------------------------
# Sidebar: load data
# -----------------------------
st.sidebar.header("Upload CSVs (optional)")
uploaded = {}
for key, label in [
    ("auctions", "Auctions CSV"),
    ("securities", "Securities CSV"),
    ("bidder", "Bidder Details CSV"),
    ("updates", "Data Updates CSV"),
]:
    uploaded[key] = st.sidebar.file_uploader(label, type=["csv"])

if any(uploaded.values()):
    # Use uploaded instead of disk
    frames = {}
    for k, up in uploaded.items():
        if up is None:
            st.error(f"Please upload all four CSVs (missing: {k}).")
            st.stop()
        df = pd.read_csv(up)
        frames[k] = df
    # Temporarily save to process with the same coercion
    tmp_dir = st.session_state.get("_tmp_dir", "._tmp_data")
    os.makedirs(tmp_dir, exist_ok=True)
    for k, df in frames.items():
        df.to_csv(os.path.join(tmp_dir, f"{k}.csv"), index=False)
    frames = load_csvs(tmp_dir, {"auctions": "auctions.csv", "bidder": "bidder.csv", "securities": "securities.csv", "updates": "updates.csv"})
else:
    frames = load_csvs(DATA_DIR, FILES)

if frames is None:
    st.stop()

# -----------------------------
# Data prep & filters
# -----------------------------
df_all = join_frames(frames)

if "auction_date" not in df_all.columns:
    st.error("Expected column 'auction_date' not found. Please verify your auctions CSV.")
    st.stop()

min_d = pd.to_datetime(df_all["auction_date"]).min().date()
max_d = pd.to_datetime(df_all["auction_date"]).max().date()

st.sidebar.header("Filters")
date_range = st.sidebar.date_input(
    "Auction date range", value=(min_d, max_d), min_value=min_d, max_value=max_d
)

# Type/term filters (if present)
sec_types = sorted(df_all["security_type"].dropna().unique().tolist()) if "security_type" in df_all.columns else []
pick_types = st.sidebar.multiselect("Security Type", sec_types, default=sec_types[:3] if sec_types else [])

terms_col = "security_term" if "security_term" in df_all.columns else None
terms = sorted(df_all[terms_col].dropna().unique().tolist()) if terms_col else []
pick_terms = st.sidebar.multiselect("Security Term", terms, default=terms[:6] if terms else [])

# Apply filters
mask = (df_all["auction_date"].dt.date >= date_range[0]) & (df_all["auction_date"].dt.date <= date_range[1])
if pick_types:
    mask &= df_all["security_type"].isin(pick_types)
if pick_terms:
    mask &= df_all[terms_col].isin(pick_terms)

df = df_all.loc[mask].copy()
df.sort_values("auction_date", inplace=True)

# -----------------------------
# Header & KPIs
# -----------------------------
st.title("U.S. Treasury Auctions – Analytics (CSV)")
st.caption("Local CSV-driven dashboard • No live database connection required")
kpi_block(df)

st.markdown("---")

# -----------------------------
# Time Series
# -----------------------------
st.subheader("Time Series")
gran = st.radio("Granularity", ["Daily", "Weekly", "Monthly"], horizontal=True)

def resample(df, date_col, agg_map):
    x = df.set_index(date_col).sort_index()
    rule = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}[gran]
    return x.resample(rule).agg(agg_map).reset_index()

agg = {}
if "bid_to_cover_ratio" in df.columns:
    agg["bid_to_cover_ratio"] = "mean"
if "high_yield" in df.columns:
    agg["high_yield"] = "mean"
if "offering_amount" in df.columns:
    agg["offering_amount"] = "sum"

ts = resample(df, "auction_date", agg) if agg else pd.DataFrame()
if not ts.empty:
    if "bid_to_cover_ratio" in ts.columns:
        alt_timeseries(ts, "auction_date:T", "bid_to_cover_ratio:Q", "Avg Bid-to-Cover")
    if "high_yield" in ts.columns:
        alt_timeseries(ts, "auction_date:T", "high_yield:Q", "Avg High Yield")
    if "offering_amount" in ts.columns:
        alt_bar(ts, "auction_date:T", "offering_amount:Q", "Total Offering Amount")

# -----------------------------
# Dealer Participation
# -----------------------------
st.subheader("Dealer Participation (Averages over selected period)")

dealer_cols = [
    "primary_dealer_percentage",
    "direct_bidder_percentage",
    "indirect_bidder_percentage",
]
available = [c for c in dealer_cols if c in df.columns]

if available:
    dealer = df[available].mean().rename_axis("type").reset_index(name="pct")
    dealer["pct"] = dealer["pct"].fillna(0.0)
    chart = (
        alt.Chart(dealer)
        .mark_bar()
        .encode(
            x=alt.X("type:N", title="Bidder Type"),
            y=alt.Y("pct:Q", title="Average %"),
            tooltip=["type", alt.Tooltip("pct:Q", format=".2f")],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("No bidder percentage columns found in the current CSVs.")

# -----------------------------
# Distribution by Security Type
# -----------------------------
st.subheader("Distribution by Security Type")
if "security_type" in df.columns and "offering_amount" in df.columns:
    dist = df.groupby("security_type", as_index=False)["offering_amount"].sum().sort_values("offering_amount", ascending=False)
    chart = (
        alt.Chart(dist)
        .mark_bar()
        .encode(
            x=alt.X("security_type:N", title="Security Type"),
            y=alt.Y("offering_amount:Q", title="Total Offering (USD)"),
            tooltip=["security_type", alt.Tooltip("offering_amount:Q", format=",.0f")],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("Security type or offering amount not available.")

# -----------------------------
# Data Preview & Export
# -----------------------------
st.markdown("---")
st.subheader("Data Preview")
st.dataframe(df.head(200), use_container_width=True)

st.download_button(
    "Download filtered data (CSV)",
    df.to_csv(index=False).encode("utf-8"),
    "treasury_auctions_filtered.csv",
    "text/csv",
)

# -----------------------------
# Audit / Update Log
# -----------------------------
st.markdown("---")
st.caption("Recent Update Runs")
updates = frames.get("updates")
if isinstance(updates, pd.DataFrame) and not updates.empty:
    show_cols = [c for c in updates.columns if c in ["update_timestamp","records_fetched","records_inserted","records_updated","status","run_type","last_auction_date"]]
    st.dataframe(updates.sort_values("update_timestamp", ascending=False)[show_cols].head(20), use_container_width=True)
else:
    st.caption("No update log available in CSVs.")
