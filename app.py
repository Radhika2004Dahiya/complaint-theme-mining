import streamlit as st
import pandas as pd

st.set_page_config(page_title="Complaint Theme Explorer", layout="wide")

st.title("Complaint Theme Explorer")
st.caption("Discovered themes from 30,000 CFPB credit card complaints using SBERT embeddings + HDBSCAN, labeled with a local LLM (Llama 3.2 3B).")

@st.cache_data
def load_data():
    return pd.read_csv("cluster_summary.csv")

summary = load_data()

st.sidebar.header("Choose a theme")
cluster_choice = st.sidebar.selectbox("Cluster", summary['label'])
row = summary[summary['label'] == cluster_choice].iloc[0]

col1, col2 = st.columns(2)
col1.metric("Cluster size", int(row['size']))
col2.metric("Dominant CFPB issue category", row['dominant_official_issue'])

st.subheader("Example complaint from this cluster")
st.info(row['example'])

st.divider()
st.subheader("All discovered themes")
st.dataframe(summary[['label', 'size', 'dominant_official_issue']].sort_values('size', ascending=False), width='stretch')
