"""
CFPB Consumer Complaints — Semantic Clustering Dashboard
Run with: streamlit run app.py

Expects a parquet file named `dashboard_data.parquet` in the same directory,
with columns: 'Date received', 'Product_canonical', 'Issue', 'Company',
'State', 'Consumer complaint narrative', 'final_cluster'
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="CFPB Complaint Clusters",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Cluster labels — derived from TF-IDF top-terms analysis.
# Edit these manually to give clusters human-readable names.
# ---------------------------------------------------------------------------
CLUSTER_LABELS = {
    "40-2": "Bank/Card Account Disputes (Chase, Wells Fargo, etc.)",
    "42-19": "Debt Collection & Validation Requests",
    "42-15": "Credit Report Inaccuracies (FCRA)",
    "42-7": "Mortgage & Escrow Issues",
    "42-9": "Student Loans (Mohela, Navient)",
    "42-17": "Identity Theft on Credit Report",
    "42-8": "Vehicle Loans & Repossession",
    "42-13": "Credit Bureau Disputes (Equifax/TransUnion/Experian)",
    "42-11": "Unauthorized Hard Inquiries",
    "38": "Hard Inquiries / Collections on Credit Report",
    "42-4": "FCRA Section 1681 Citations",
    "40-0": "Cash App Disputes",
    "23": "Cash App — Regulatory/Unfair Practice Complaints",
    "2": "Consumer Reporting — FCRA Section 1681",
    "42-1": "Consumer Reporting — FCRA Section 1681 (variant)",
    "24": "Cash App — Account Blocking Complaints",
    "40-1": "PayPal Disputes",
    "34": "Zelle Transfer Failures",
    "3": "Legal/Template Boilerplate — Creditor Litigation Language",
    "9": "Credit Report Correction Requests",
    "22": "Legal/Template Boilerplate — FCRA 1681i/1692e Citations",
    "26": "Late Payment / Blocking Deletion Requests",
    "5": "Legal/Template Boilerplate — FCRA 1681i Citations",
    "18": "Banking Violation / TILA Citations",
    "1": "Unusual / Templated Feedback Language",
    "42-6": "Bankruptcy on Credit Report",
    "16": "Late Payment Reporting Disputes",
    "42-5": "Debt Collection — 1099/IRS Income Reporting",
    "42-12": "Fraudulent Accounts on Credit Report",
    "27": "Credit Report — Items Belonging to Others (FCRA 605b)",
    "0": "Unauthorized Transaction Reporting",
    "8": "Legal/Template Boilerplate — Unusual Citations",
    "4": "Credit Report — Item Deletion Requests",
    "42-0": "FCRA Section 1681 Citations (variant)",
    "37": "Falsely Reporting Credit Information",
    "7": "Zelle Transfer Failures (variant)",
    "41": "Credit Report Accuracy Verification Requests",
    "42-3": "FCRA Section 1681 Citations (variant 2)",
    "35": "Credit Dispute — Unprofessional Handling Complaints",
    "17": "Banking Violation / TILA Citations (variant)",
    "11": "Credit Report Discrepancies (FCRA 605b)",
    "42-16": "Fraudulent Accounts Hurting Credit",
    "20": "Legal/Template Boilerplate — Debt Estoppel Language",
    "39": "Debt Validation Failure (FCRA 609/611)",
    "25": "Navy Federal — Transaction Disputes",
    "10": "Inaccurate Date Reporting",
    "29": "Identity Theft — Item Deletion Requests",
    "21": "Outdated Late Payment Reporting",
    "33": "Credit Report — General Reporting Disputes",
    "32": "Navy Federal — Fee/Charge Disputes",
    "28": "Consumer Reporting Agency — Reseller Disputes",
    "31": "Creditor Application Materials Disputes",
    "19": "Unauthorized Transaction — Identity Theft",
    "14": "Credit Report — Unjust/Incorrect Reporting",
    "42-2": "Student Loan Privacy (FERPA)",
    "36": "Cash App Fraud Complaints",
    "15": "Credit Report Accuracy Review Requests",
    "12": "Credit Score — Goodwill/Gesture Requests",
    "42-14": "Credit Report — Balance/Theft Disputes (FCRA 605b)",
    "13": "Derogatory Rating Disputes",
    "30": "Late Payment — TILA Section 1637 Citations",
    "6": "Unusual / Emotionally-Written Complaints",
    "42-18": "Credit Report — Account Status/Balance Disputes",
    "42-10": "Identity Theft — Victim Information Disputes",
}


@st.cache_data
def load_data():
    df = pd.read_parquet("dashboard_data.parquet")
    df["final_cluster"] = df["final_cluster"].astype(str)
    df["cluster_label"] = df["final_cluster"].map(
        lambda c: CLUSTER_LABELS.get(c, f"Cluster {c}") if c != "-1" else "Unclustered / Noise"
    )
    df["Date received"] = pd.to_datetime(df["Date received"], errors="coerce")
    return df


df = load_data()

st.title("CFPB Consumer Complaints — Semantic Clustering")
st.caption(
    f"Stratified sample of {len(df):,} consumer complaints, embedded with "
    "sentence-transformers (all-mpnet-base-v2), reduced with UMAP, and "
    "clustered with HDBSCAN (two-level hierarchical clustering)."
)

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")

products = sorted(df["Product_canonical"].dropna().unique().tolist())
selected_products = st.sidebar.multiselect("Product category", products, default=[])

states = sorted(df["State"].dropna().unique().tolist())
selected_states = st.sidebar.multiselect("State", states, default=[])

show_noise = st.sidebar.checkbox("Include unclustered / noise points", value=False)

filtered = df.copy()
if selected_products:
    filtered = filtered[filtered["Product_canonical"].isin(selected_products)]
if selected_states:
    filtered = filtered[filtered["State"].isin(selected_states)]
if not show_noise:
    filtered = filtered[filtered["final_cluster"] != "-1"]

st.sidebar.metric("Complaints in view", f"{len(filtered):,}")

# ---------------------------------------------------------------------------
# Overview metrics
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total complaints (sample)", f"{len(df):,}")
col2.metric("Clusters found", f"{df[df['final_cluster'] != '-1']['final_cluster'].nunique()}")
noise_pct = (df["final_cluster"] == "-1").mean() * 100
col3.metric("Noise rate", f"{noise_pct:.1f}%")
col4.metric("Product categories", f"{df['Product_canonical'].nunique()}")

st.divider()

# ---------------------------------------------------------------------------
# Cluster size chart
# ---------------------------------------------------------------------------
st.subheader("Cluster Sizes")

cluster_counts = (
    filtered.groupby(["final_cluster", "cluster_label"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
    .head(25)
)

if len(cluster_counts) > 0:
    chart_df = cluster_counts.set_index("cluster_label")["count"]
    st.bar_chart(chart_df, horizontal=True)
else:
    st.info("No data matches the current filters.")

st.divider()

# ---------------------------------------------------------------------------
# Cluster explorer
# ---------------------------------------------------------------------------
st.subheader("Explore a Cluster")

available_clusters = (
    filtered[["final_cluster", "cluster_label"]]
    .drop_duplicates()
    .sort_values("cluster_label")
)
label_to_id = dict(zip(available_clusters["cluster_label"], available_clusters["final_cluster"]))

if len(label_to_id) > 0:
    chosen_label = st.selectbox("Select a cluster to inspect", list(label_to_id.keys()))
    chosen_id = label_to_id[chosen_label]

    cluster_rows = filtered[filtered["final_cluster"] == chosen_id]

    c1, c2, c3 = st.columns(3)
    c1.metric("Complaints in cluster", f"{len(cluster_rows):,}")
    c2.metric("Top product category", cluster_rows["Product_canonical"].mode().iloc[0] if len(cluster_rows) else "-")
    c3.metric("Top state", cluster_rows["State"].mode().iloc[0] if cluster_rows["State"].notna().any() else "-")

    st.markdown("**Product category breakdown within this cluster:**")
    prod_breakdown = cluster_rows["Product_canonical"].value_counts().head(10)
    st.bar_chart(prod_breakdown)

    st.markdown("**Example complaints from this cluster:**")
    n_examples = st.slider("Number of examples to show", 1, 10, 3, key="n_examples")
    examples = cluster_rows.sample(min(n_examples, len(cluster_rows)), random_state=42)
    for _, row in examples.iterrows():
        with st.container(border=True):
            st.markdown(f"**Company:** {row['Company']} &nbsp;|&nbsp; **State:** {row['State']} &nbsp;|&nbsp; **Issue:** {row['Issue']}")
            narrative = str(row["Consumer complaint narrative"])
            st.write(narrative[:800] + ("..." if len(narrative) > 800 else ""))
else:
    st.info("No clusters match the current filters.")

st.divider()

# ---------------------------------------------------------------------------
# Full table (searchable)
# ---------------------------------------------------------------------------
st.subheader("Browse All Complaints")
search_term = st.text_input("Search narrative text (case-insensitive)")

table_df = filtered[["Date received", "Product_canonical", "Issue", "Company", "State", "cluster_label"]].copy()
if search_term:
    mask = filtered["Consumer complaint narrative"].astype(str).str.contains(search_term, case=False, na=False)
    table_df = table_df[mask]

st.dataframe(table_df.head(500), width='stretch', height=400)
st.caption(f"Showing up to 500 of {len(table_df):,} matching rows.")
