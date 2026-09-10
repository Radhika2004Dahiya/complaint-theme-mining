"""
Streamlit Dashboard for Complaint Theme Mining.
Displays interactive cluster visualizations, automated LLM theme summaries,
and narrative filtering with high-performance caching.
"""

import sys
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA

from src import config
from src.nlp_pipeline import run_pipeline, generate_embeddings, extract_cluster_keywords, evaluate_clusters
from src.llm_labeler import auto_label_all_clusters

st.set_page_config(
    page_title="CFPB Complaint Theme Mining Dashboard",
    page_icon="📊",
    layout="wide"
)


@st.cache_data
def cached_run_pipeline(data_path: str):
    """
    Cached function to run or load the NLP pipeline on dataset.
    """
    return run_pipeline(data_path=data_path)


@st.cache_data
def cached_compute_pca(df: pd.DataFrame, text_column: str = "consumer_complaint_narrative"):
    """
    Computes 2D PCA projection of complaint embeddings for fast visualization.
    """
    narratives = df[text_column].tolist()
    embeddings = generate_embeddings(narratives)

    if len(embeddings) > 1:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(embeddings)
    else:
        coords = np.zeros((len(embeddings), 2))

    coords_df = pd.DataFrame(coords, columns=["PCA1", "PCA2"])
    return coords_df


def main():
    st.title("📊 CFPB Complaint Theme Mining & LLM Auto-Labeling")
    st.markdown("""
    This dashboard analyzes consumer complaints using **SBERT embeddings**, **HDBSCAN clustering**,
    and **Llama 3.2 (Ollama)** for automatic theme labeling and summary generation.
    """)

    # Sidebar Data Controls
    st.sidebar.header("Data & Configuration")
    data_option = st.sidebar.radio(
        "Select Data Source:",
        ("Default Sample Dataset", "Upload Custom CSV")
    )

    data_path = str(config.DEFAULT_SAMPLE_DATA_PATH)
    if data_option == "Upload Custom CSV":
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            temp_path = config.PROCESSED_DATA_DIR / "uploaded_temp.csv"
            config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            data_path = str(temp_path)

    # Load and process data
    with st.spinner("Processing complaint pipeline..."):
        try:
            df, metrics = cached_run_pipeline(data_path)
        except Exception as e:
            st.error(f"Failed to process dataset: {e}")
            return

    # Auto-label themes with local LLM / fallback
    keywords_map = metrics.get("cluster_keywords", {})
    theme_labels = auto_label_all_clusters(keywords_map, df)

    # Map theme titles onto dataframe
    df["theme_title"] = df["cluster"].map(lambda c: theme_labels.get(c, {}).get("theme_title", f"Cluster {c}"))

    # Key KPI Metrics Display
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Complaints", len(df))
    col2.metric("Discovered Themes", metrics["n_clusters"])
    col3.metric("Noise Complaints", metrics["noise_count"])
    sil_str = f"{metrics['silhouette_score']:.3f}" if metrics['silhouette_score'] is not None else "N/A"
    col4.metric("Silhouette Score", sil_str)

    st.markdown("---")

    # Layout Tabs
    tab1, tab2, tab3 = st.tabs(["🗺️ Theme Clusters Visualization", "🏷️ Cluster Summaries & LLM Labels", "🔍 Narrative Explorer"])

    with tab1:
        st.subheader("2D Semantic Space Visualization")
        pca_coords = cached_compute_pca(df)
        viz_df = pd.concat([df.reset_index(drop=True), pca_coords], axis=1)

        viz_df["cluster_label"] = viz_df["cluster"].apply(lambda c: f"Cluster {c}" if c != -1 else "Noise (-1)")

        fig = px.scatter(
            viz_df,
            x="PCA1",
            y="PCA2",
            color="theme_title",
            hover_data=["complaint_id", "product", "issue"],
            title="SBERT Embedding Clusters (PCA Projection)",
            template="plotly_white",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Discovered Theme Summaries")

        for c_id, info in sorted(theme_labels.items()):
            if c_id == -1:
                continue
            with st.expander(f"📌 Cluster {c_id}: {info['theme_title']}", expanded=True):
                kws = keywords_map.get(c_id, [])
                st.markdown(f"**Keywords:** `{', '.join(kws)}`")
                st.markdown(f"**Executive Summary:** {info['summary']}")

                c_df = df[df["cluster"] == c_id]
                st.markdown(f"**Sample Count:** {len(c_df)} complaints")

    with tab3:
        st.subheader("Search & Inspect Complaint Narratives")

        selected_theme = st.selectbox(
            "Filter by Theme:",
            options=["All Themes"] + list(df["theme_title"].unique())
        )

        filtered_df = df if selected_theme == "All Themes" else df[df["theme_title"] == selected_theme]

        st.dataframe(
            filtered_df[[
                "complaint_id", "product", "issue", "theme_title",
                "consumer_complaint_narrative", "company"
            ]],
            use_container_width=True,
            height=400
        )


if __name__ == "__main__":
    main()
