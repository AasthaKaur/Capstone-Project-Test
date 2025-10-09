
# app_fiscal_dashboard.py — CSV-Only, Plotly Interactive
# Run: pip install streamlit pandas plotly numpy
#      streamlit run app_fiscal_dashboard.py

import os
import glob
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta

st.set_page_config(page_title="Fiscal & Treasury Dashboard", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))

def find_file(patterns):
    if isinstance(patterns, str):
        patterns = [patterns]
    for pat in patterns:
        matches = sorted(glob.glob(os.path.join(HERE, pat)))
        if matches:
            return matches[-1]
    return None

def load_csv(file_or_buffer, parse_dates=None):
    if file_or_buffer is None:
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_or_buffer, low_memory=False)
        if parse_dates:
            for c in parse_dates:
                if c in df.columns:
                    df[c] = pd.to_datetime(df[c], errors="coerce")
        return df
    except Exception as e:
        st.warning(f"Could not load CSV: {e}")
        return pd.DataFrame()

def kpi_card(col, label, value):
    col.metric(label, value if value is not None else "—")

def num_fmt(n, decimals=2):
    if n is None or (isinstance(n, float) and (pd.isna(n))):
        return "—"
    if abs(n) >= 1_000_000_000:
        return f"{n/1_000_000_000:.{decimals}f}B"
    if abs(n) >= 1_000_000:
        return f"{n/1_000_000:.{decimals}f}M"
    if abs(n) >= 1_000:
        return f"{n/1_000:.{decimals}f}K"
    return f"{n:.{decimals}f}"

# ---------------------- Sidebar Uploads ----------------------
st.sidebar.header("Upload CSVs (or use local files)")

uploaded_auctions = st.sidebar.file_uploader("auctions_*.csv", type=["csv"], key="auctions")
uploaded_bidders  = st.sidebar.file_uploader("bidder_details_*.csv", type=["csv"], key="bidders")
uploaded_updates  = st.sidebar.file_uploader("data_updates_*.csv", type=["csv"], key="updates")
uploaded_secs     = st.sidebar.file_uploader("securities_*.csv", type=["csv"], key="securities")

# Fiscal CSV without date
uploaded_scores   = st.sidebar.file_uploader("wsj_articles_scores.csv (no date column)", type=["csv"], key="scores")

# Fallback to local files if not uploaded
auctions_path = uploaded_auctions if uploaded_auctions else find_file(["auctions_*.csv"])
bidders_path  = uploaded_bidders  if uploaded_bidders  else find_file(["bidder_details_*.csv"])
updates_path  = uploaded_updates  if uploaded_updates  else find_file(["data_updates_*.csv"])
secs_path     = uploaded_secs     if uploaded_secs     else find_file(["securities_*.csv"])
scores_path   = uploaded_scores   if uploaded_scores   else find_file(["wsj_articles_scores.csv", "fiscal_policy_index/wsj_articles_scores.csv"])

# Load dataframes
auctions = load_csv(auctions_path, parse_dates=["auction_date","announcement_date","issue_date","maturity_date"])
securities = load_csv(secs_path)
bidders = load_csv(bidders_path)
updates = load_csv(updates_path, parse_dates=["update_timestamp"])
scores = load_csv(scores_path)

# Normalize column names to lowercase for auctions, bidders, securities
if not auctions.empty:
    auctions.columns = [c.lower() for c in auctions.columns]
if not bidders.empty:
    bidders.columns = [c.lower() for c in bidders.columns]
if not securities.empty:
    securities.columns = [c.lower() for c in securities.columns]

# Merge tenor fields into auctions if possible
if not auctions.empty and not securities.empty and "cusip" in auctions.columns and "cusip" in securities.columns:
    tenor_cols = [c for c in ["security_type","security_term","series","tips","callable"] if c in securities.columns]
    auctions = auctions.merge(securities[["cusip"] + tenor_cols], on="cusip", how="left")

# ---------------------- Top Navigation ----------------------
st.title("Fiscal & Treasury Analytics")
top_choice = st.radio("Select view", ["Auction DB analysis", "Fiscal Dashboard"], horizontal=True, label_visibility="collapsed")

# Derive a generic date window for Auction page
if "auction_date" in auctions.columns and not auctions.empty:
    min_date = pd.to_datetime(auctions["auction_date"]).min().date()
    max_date = pd.to_datetime(auctions["auction_date"]).max().date()
else:
    today = date.today()
    min_date = today - timedelta(days=365)
    max_date = today

date_range = st.sidebar.date_input("Date range (for auctions visuals)", (min_date, max_date))
start_date, end_date = (date_range if isinstance(date_range, tuple) else (min_date, max_date))

# ---------------------- Page 1: Auction DB analysis ----------------------
if top_choice == "Auction DB analysis":
    st.subheader("Auction Database Analysis")

    if auctions.empty:
        st.info("No auctions data loaded. Upload 'auctions_*.csv'.")
    else:
        a = auctions.copy()
        if "auction_date" in a.columns:
            a["auction_date"] = pd.to_datetime(a["auction_date"], errors="coerce")
            a = a[(a["auction_date"].dt.date >= start_date) & (a["auction_date"].dt.date <= end_date)]

        sec_types = sorted(a["security_type"].dropna().unique().tolist()) if "security_type" in a.columns else []
        sec_terms = sorted(a["security_term"].dropna().unique().tolist()) if "security_term" in a.columns else []
        c1, c2 = st.columns(2)
        sel_type = c1.multiselect("Security type", options=sec_types, default=sec_types[:3] if len(sec_types)>3 else sec_types)
        sel_term = c2.multiselect("Security term", options=sec_terms, default=sec_terms[:6] if len(sec_terms)>6 else sec_terms)

        if sel_type and "security_type" in a.columns:
            a = a[a["security_type"].isin(sel_type)]
        if sel_term and "security_term" in a.columns:
            a = a[a["security_term"].isin(sel_term)]

        k1, k2, k3, k4 = st.columns(4)
        total_auctions = len(a)
        avg_btc = a["bid_to_cover_ratio"].dropna().mean() if "bid_to_cover_ratio" in a.columns else None
        avg_high_yield = a["high_yield"].dropna().mean() if "high_yield" in a.columns else None
        total_offered = a["offering_amount"].dropna().sum() if "offering_amount" in a.columns else None

        kpi_card(k1, "Total auctions", f"{total_auctions}")
        kpi_card(k2, "Avg bid-to-cover", f"{avg_btc:.2f}" if avg_btc is not None else None)
        kpi_card(k3, "Avg high yield (%)", f"{avg_high_yield:.2f}" if avg_high_yield is not None else None)
        kpi_card(k4, "Total offered ($)", num_fmt(total_offered, 0) if total_offered is not None else None)

        st.markdown("---")

        # Time series Bid-to-Cover
        if "bid_to_cover_ratio" in a.columns:
            ts_btc = a[["auction_date","bid_to_cover_ratio"]].dropna().sort_values("auction_date")
            ts_btc["MA_10"] = ts_btc["bid_to_cover_ratio"].rolling(10, min_periods=5).mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts_btc["auction_date"], y=ts_btc["bid_to_cover_ratio"], mode="lines", name="Bid-to-Cover"))
            fig.add_trace(go.Scatter(x=ts_btc["auction_date"], y=ts_btc["MA_10"], mode="lines", name="MA(10)"))
            fig.update_layout(title="Bid-to-Cover trend", xaxis_title="Auction date", yaxis_title="Bid-to-Cover", height=360, legend_title="")
            st.plotly_chart(fig, use_container_width=True)

        # Yields
        ycols = [c for c in ["high_yield","low_yield","average_median_yield"] if c in a.columns]
        if ycols:
            ts_y = a[["auction_date"] + ycols].dropna().sort_values("auction_date")
            figy = go.Figure()
            for col in ycols:
                figy.add_trace(go.Scatter(x=ts_y["auction_date"], y=ts_y[col], mode="lines", name=col))
            figy.update_layout(title="Yields over time", xaxis_title="Auction date", yaxis_title="Yield (%)", height=360, legend_title="")
            st.plotly_chart(figy, use_container_width=True)

        # Bidder participation
        if not bidders.empty and "auction_id" in a.columns and "auction_id" in bidders.columns:
            p = a[["auction_id","auction_date"]].merge(bidders, on="auction_id", how="left")
            part_cols = [c for c in ["primary_dealer_percentage","direct_bidder_percentage","indirect_bidder_percentage"] if c in p.columns]
            if part_cols:
                p_s = p[["auction_date"] + part_cols].dropna().sort_values("auction_date")
                p_m = p_s.melt("auction_date", var_name="Bidder type", value_name="Share")
                figp = px.area(p_m, x="auction_date", y="Share", color="Bidder type", groupnorm="fraction",
                               title="Bidder participation shares", labels={"Share":"Share"})
                figp.update_layout(height=360, legend_title="")
                st.plotly_chart(figp, use_container_width=True)

        # Avg Bid-to-Cover by term
        if "security_term" in a.columns and "bid_to_cover_ratio" in a.columns:
            comp = a.groupby("security_term", as_index=False)["bid_to_cover_ratio"].mean().sort_values("bid_to_cover_ratio", ascending=False)
            figc = px.bar(comp, x="security_term", y="bid_to_cover_ratio", title="Average bid-to-cover by term", labels={"security_term":"Term", "bid_to_cover_ratio":"Avg B/C"})
            figc.update_layout(height=360, xaxis_tickangle=-30)
            st.plotly_chart(figc, use_container_width=True)

        st.markdown("#### Latest auctions")
        show_cols = [c for c in ["auction_date","security_type","security_term","offering_amount","total_accepted","bid_to_cover_ratio","high_yield"] if c in a.columns]
        st.dataframe(a.sort_values("auction_date", ascending=False)[show_cols].head(30), use_container_width=True)

# ---------------------- Page 2: Fiscal Dashboard ----------------------
if top_choice == "Fiscal Dashboard":
    st.subheader("Fiscal Dashboard (Aggregated Index Snapshot)")

    if scores.empty:
        st.info("No fiscal summary data loaded. Upload 'wsj_articles_scores.csv'.")
    else:
        sc = scores.copy()
        sc.columns = [c.strip() for c in sc.columns]

        expected = [
            "total_articles","fiscal_articles","tariff_fiscal_articles","non_tariff_fiscal_articles",
            "rate","tariff_rate","non_tariff_rate",
            "fiscal_policy_index","tariff_fiscal_index","non_tariff_fiscal_index"
        ]
        missing = [c for c in expected if c not in sc.columns]
        if missing:
            st.warning(f"Missing expected columns: {missing}")

        # If multiple rows exist, treat row index as pseudo-time and allow simple row selection
        if len(sc) > 1:
            st.caption("Multiple rows detected. Using row index as pseudo-time for comparison.")
            idx = st.slider("Select row to summarise", min_value=0, max_value=len(sc)-1, value=len(sc)-1, step=1)
            row = sc.iloc[idx]
            sc["row_idx"] = range(len(sc))
        else:
            row = sc.iloc[0]

        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        kpi_card(k1, "Fiscal Policy Index", f"{row.get('fiscal_policy_index', np.nan):.1f}" if pd.notna(row.get('fiscal_policy_index', np.nan)) else None)
        kpi_card(k2, "Tariff FPI", f"{row.get('tariff_fiscal_index', np.nan):.1f}" if pd.notna(row.get('tariff_fiscal_index', np.nan)) else None)
        kpi_card(k3, "Non-Tariff FPI", f"{row.get('non_tariff_fiscal_index', np.nan):.1f}" if pd.notna(row.get('non_tariff_fiscal_index', np.nan)) else None)
        ratio = (row.get("fiscal_articles", 0) / row.get("total_articles", np.nan) * 100.0) if row.get("total_articles", 0) not in [0, np.nan] else np.nan
        kpi_card(k4, "Fiscal article share (%)", f"{ratio:.1f}%" if pd.notna(ratio) else None)

        st.markdown("---")

        # Bar comparisons: rates and indices
        left, right = st.columns(2)

        # Rates comparison
        rate_df = pd.DataFrame({
            "Category": ["Overall", "Tariff", "Non-tariff"],
            "Rate (%)": [
                row.get("rate", np.nan)*100.0 if pd.notna(row.get("rate", np.nan)) else np.nan,
                row.get("tariff_rate", np.nan)*100.0 if pd.notna(row.get("tariff_rate", np.nan)) else np.nan,
                row.get("non_tariff_rate", np.nan)*100.0 if pd.notna(row.get("non_tariff_rate", np.nan)) else np.nan,
            ]
        })
        fig_rate = px.bar(rate_df, x="Category", y="Rate (%)", title="Fiscal Article Rates")
        left.plotly_chart(fig_rate, use_container_width=True)

        # Indices comparison
        idx_df = pd.DataFrame({
            "Category": ["Overall", "Tariff", "Non-tariff"],
            "Index": [
                row.get("fiscal_policy_index", np.nan),
                row.get("tariff_fiscal_index", np.nan),
                row.get("non_tariff_fiscal_index", np.nan),
            ]
        })
        fig_idx = px.bar(idx_df, x="Category", y="Index", title="Fiscal Policy Indices (normalised)")
        right.plotly_chart(fig_idx, use_container_width=True)

        # Counts comparison
        st.markdown("#### Article counts")
        cnt_df = pd.DataFrame({
            "Category": ["Total", "Fiscal", "Tariff fiscal", "Non-tariff fiscal"],
            "Count": [
                row.get("total_articles", np.nan),
                row.get("fiscal_articles", np.nan),
                row.get("tariff_fiscal_articles", np.nan),
                row.get("non_tariff_fiscal_articles", np.nan),
            ]
        })
        fig_cnt = px.bar(cnt_df, x="Category", y="Count", title="Article Counts")
        st.plotly_chart(fig_cnt, use_container_width=True)

        # If multiple rows: simple trend using row index
        if len(sc) > 1:
            st.markdown("#### Trends across rows (pseudo-time)")
            trend_cols = [c for c in expected if c in sc.columns]
            tdf = sc[["row_idx"] + trend_cols].copy()
            # Plot FPI and rates over row index
            t1, t2 = st.columns(2)
            if "fiscal_policy_index" in tdf.columns:
                fig_t1 = go.Figure()
                fig_t1.add_trace(go.Scatter(x=tdf["row_idx"], y=tdf["fiscal_policy_index"], mode="lines+markers", name="Fiscal Policy Index"))
                if "tariff_fiscal_index" in tdf.columns:
                    fig_t1.add_trace(go.Scatter(x=tdf["row_idx"], y=tdf["tariff_fiscal_index"], mode="lines+markers", name="Tariff FPI"))
                if "non_tariff_fiscal_index" in tdf.columns:
                    fig_t1.add_trace(go.Scatter(x=tdf["row_idx"], y=tdf["non_tariff_fiscal_index"], mode="lines+markers", name="Non-tariff FPI"))
                fig_t1.update_layout(title="Index trends (by row)", xaxis_title="Row index", yaxis_title="Index")
                t1.plotly_chart(fig_t1, use_container_width=True)

            rate_cols = [c for c in ["rate","tariff_rate","non_tariff_rate"] if c in tdf.columns]
            if rate_cols:
                fig_t2 = go.Figure()
                for col in rate_cols:
                    fig_t2.add_trace(go.Scatter(x=tdf["row_idx"], y=tdf[col]*100.0, mode="lines+markers", name=col))
                fig_t2.update_layout(title="Rate trends (by row)", xaxis_title="Row index", yaxis_title="Rate (%)")
                t2.plotly_chart(fig_t2, use_container_width=True)

        # Linked analysis vs Auctions (if auctions provided and have dates)
        if not auctions.empty and "auction_date" in auctions.columns:
            st.markdown("---")
            st.subheader("Linked Analysis (Aggregate snapshot)")
            st.caption("Compares current fiscal indices with average auction outcomes in the selected date window.")
            a = auctions.copy()
            a["auction_date"] = pd.to_datetime(a["auction_date"], errors="coerce")
            a = a[(a["auction_date"].dt.date >= start_date) & (a["auction_date"].dt.date <= end_date)]
            if not a.empty:
                btc = a["bid_to_cover_ratio"].mean() if "bid_to_cover_ratio" in a.columns else np.nan
                hy  = a["high_yield"].mean() if "high_yield" in a.columns else np.nan
                # Scatter of index vs outcomes (single point each)
                lm, rm = st.columns(2)
                fig_s1 = go.Figure()
                fig_s1.add_trace(go.Scatter(
                    x=[row.get("fiscal_policy_index", np.nan)], y=[btc],
                    mode="markers+text", text=["Current snapshot"], textposition="top center", name="FPI vs B/C"
                ))
                fig_s1.update_layout(xaxis_title="Fiscal Policy Index", yaxis_title="Avg Bid-to-Cover", title="FPI vs Avg Bid-to-Cover")
                lm.plotly_chart(fig_s1, use_container_width=True)

                fig_s2 = go.Figure()
                fig_s2.add_trace(go.Scatter(
                    x=[row.get("fiscal_policy_index", np.nan)], y=[hy],
                    mode="markers+text", text=["Current snapshot"], textposition="top center", name="FPI vs Yield"
                ))
                fig_s2.update_layout(xaxis_title="Fiscal Policy Index", yaxis_title="Avg High Yield (%)", title="FPI vs Avg High Yield")
                rm.plotly_chart(fig_s2, use_container_width=True)

# ---------------------- Footer ----------------------
st.markdown("---")
if not updates.empty and "update_timestamp" in updates.columns:
    last_upd = pd.to_datetime(updates["update_timestamp"]).max()
    st.caption(f"Last auctions update: {last_upd}")
else:
    st.caption("Upload data_updates_*.csv to display the last update timestamp.")
