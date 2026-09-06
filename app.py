import os
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="CFPB Complaint Theme Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application Header & Description
st.title("🔍 CFPB Complaint Theme Explorer")
st.caption("Unsupervised theme discovery from CFPB credit card complaints using SBERT embeddings, UMAP, HDBSCAN, and Llama 3.2 auto-labeling.")

@st.cache_data
def load_summary_data() -> pd.DataFrame:
    """Load cluster summary data with caching."""
    possible_paths = [
        "data/processed/cluster_summary.csv",
        "cluster_summary.csv"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Ensure proper types
            df["size"] = pd.to_numeric(df["size"], errors="coerce").fillna(0).astype(int)
            return df
    st.error("Cluster summary dataset not found. Please ensure `data/processed/cluster_summary.csv` exists.")
    return pd.DataFrame()

summary = load_summary_data()

if summary.empty:
    st.warning("No summary data available.")
    st.stop()

# --- Sidebar Controls ---
st.sidebar.header("🎯 Theme Filters & Selection")

search_term = st.sidebar.text_input("Search themes/issues", value="").strip()

if search_term:
    filtered_summary = summary[
        summary["label"].astype(str).str.contains(search_term, case=False, na=False) |
        summary["dominant_official_issue"].astype(str).str.contains(search_term, case=False, na=False)
    ]
else:
    filtered_summary = summary

if filtered_summary.empty:
    st.sidebar.warning("No clusters match your search query.")
    cluster_options = summary["label"].tolist()
else:
    cluster_options = filtered_summary["label"].tolist()

selected_cluster_label = st.sidebar.selectbox("Choose a Cluster Theme", options=cluster_options)

# Get row for selected cluster
selected_row = summary[summary["label"] == selected_cluster_label].iloc[0]

# --- Key Dashboard Metrics ---
total_clustered_complaints = summary["size"].sum()
total_clusters_count = len(summary)
avg_cluster_size = int(summary["size"].mean())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Themes Discovered", total_clusters_count)
m2.metric("Clustered Complaints", f"{total_clustered_complaints:,}")
m3.metric("Avg Theme Size", f"{avg_cluster_size:,}")
m4.metric("Silhouette Validation Score", "0.569")

st.divider()

# --- Selected Cluster Focus ---
st.subheader(f"🏷️ Selected Theme: {selected_row['label']}")

col1, col2 = st.columns([1, 2])

with col1:
    st.metric("Cluster Size (Complaints)", f"{int(selected_row['size']):,}")
    st.metric("Dominant Official CFPB Issue", selected_row["dominant_official_issue"])
    st.caption("Cluster ID: " + str(selected_row["cluster"]))

with col2:
    st.markdown("### Representative Complaint Example")
    st.info(f"\"{selected_row['example']}\"")

st.divider()

# --- Interactive Visualizations & Deep Dive Data ---
tab1, tab2, tab3 = st.tabs(["📊 Theme Visualizations", "📋 All Discovered Themes Table", "ℹ️ Methodology & Architecture"])

with tab1:
    col_v1, col_v2 = st.columns(2)

    with col_v1:
        st.markdown("#### Top 15 Discovered Cluster Themes by Size")
        top_15 = summary.sort_values("size", ascending=False).head(15)
        fig_bar = px.bar(
            top_15,
            x="size",
            y="label",
            orientation="h",
            color="size",
            color_continuous_scale="Blues",
            labels={"size": "Complaint Count", "label": "Discovered Theme"},
            text="size"
        )
        fig_bar.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False, height=450)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_v2:
        st.markdown("#### CFPB Official Issue Breakdown Across Clusters")
        issue_counts = summary["dominant_official_issue"].value_counts().reset_index()
        issue_counts.columns = ["CFPB Official Issue", "Cluster Count"]
        fig_pie = px.pie(
            issue_counts,
            values="Cluster Count",
            names="CFPB Official Issue",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_pie.update_layout(height=450)
        st.plotly_chart(fig_pie, use_container_width=True)

with tab2:
    st.markdown("#### Complete Cluster Summary")
    st.dataframe(
        summary[["cluster", "label", "size", "dominant_official_issue", "example"]]
        .sort_values("size", ascending=False)
        .reset_index(drop=True),
        use_container_width=True,
        column_config={
            "cluster": "Cluster ID",
            "label": "Auto-Generated Theme Label",
            "size": "Complaint Count",
            "dominant_official_issue": "Dominant CFPB Issue",
            "example": "Centroid Example Complaint"
        }
    )

with tab3:
    st.markdown("""
    ### Pipeline Architecture & Technical Methodology

    1. **Data Preprocessing & PII Redaction**:
       - Filtered CFPB Consumer Complaint Database for Credit Card narratives.
       - Stripped CFPB's `XXXX` PII redaction tokens into standardized markers.
       - Normalized whitespace and filtered narratives < 30 characters.

    2. **Sentence Embeddings & Dimension Reduction**:
       - Embedded complaint text using `all-MiniLM-L6-v2` SBERT model (384 dimensions).
       - Reduced dimensions to 10 UMAP components using cosine distance.

    3. **HDBSCAN Density-Based Clustering**:
       - Density-based clustering automatically surfaced 32 latent sub-themes.
       - Achieved **0.569 Silhouette Score** on assigned non-noise points.

    4. **Local LLM Auto-Labeling**:
       - Selected 5 narratives closest to each cluster centroid.
       - Prompted local **Llama 3.2 3B** via Ollama to synthesize short human-readable cluster titles.
    """)
