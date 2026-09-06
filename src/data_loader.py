"""
Data loading and preprocessing module for CFPB complaint datasets.
"""

import os
import pandas as pd
from typing import Optional, Iterator
from src.utils import clean_narrative

DEFAULT_USECOLS = [
    "Date received",
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    "Consumer complaint narrative",
    "Company",
    "State",
    "ZIP code",
    "Submitted via",
    "Company response to consumer",
    "Timely response?",
    "Complaint ID"
]

def load_and_preprocess_complaints(
    file_path: str,
    product_filter: Optional[str] = "Credit card",
    min_narrative_len: int = 30,
    chunksize: Optional[int] = 100_000,
    max_rows: Optional[int] = None
) -> pd.DataFrame:
    """
    Load CFPB complaint dataset in a memory-efficient chunked manner, apply filtering,
    and clean consumer narratives.

    Args:
        file_path: Path to raw CSV or Parquet complaint file.
        product_filter: Product category to filter by (e.g., 'Credit card'). If None, no filter applied.
        min_narrative_len: Minimum character length for valid narratives.
        chunksize: Number of rows to read per chunk for CSV files.
        max_rows: Maximum total rows to return.

    Returns:
        Pandas DataFrame of filtered and cleaned complaints.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")

    if file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
        return _process_dataframe(df, product_filter, min_narrative_len, max_rows)

    # Process CSV files in chunks to avoid memory overflow on 100K+ / 13M+ rows
    chunks = []
    total_rows = 0

    reader = pd.read_csv(
        file_path,
        chunksize=chunksize,
        low_memory=False,
        on_bad_lines="skip",
        dtype=str
    )

    for chunk in reader:
        processed_chunk = _process_dataframe(chunk, product_filter, min_narrative_len, None)
        if not processed_chunk.empty:
            chunks.append(processed_chunk)
            total_rows += len(processed_chunk)
            if max_rows and total_rows >= max_rows:
                break

    if not chunks:
        return pd.DataFrame()

    full_df = pd.concat(chunks, ignore_index=True)
    if max_rows and len(full_df) > max_rows:
        full_df = full_df.iloc[:max_rows]

    return full_df

def _process_dataframe(
    df: pd.DataFrame,
    product_filter: Optional[str],
    min_narrative_len: int,
    max_rows: Optional[int]
) -> pd.DataFrame:
    """Internal helper to clean and filter a DataFrame or chunk."""
    # Ensure Consumer complaint narrative column exists
    narrative_col = "Consumer complaint narrative"
    if narrative_col not in df.columns:
        return pd.DataFrame()

    # Drop missing narratives
    df = df.dropna(subset=[narrative_col]).copy()

    # Apply product filter if specified
    if product_filter and "Product" in df.columns:
        df = df[df["Product"].str.contains(product_filter, case=False, na=False)]

    if df.empty:
        return df

    # Clean narrative texts
    df["cleaned_narrative"] = df[narrative_col].astype(str).apply(clean_narrative)

    # Filter out narratives below minimum length threshold
    df = df[df["cleaned_narrative"].str.len() >= min_narrative_len]

    if max_rows and len(df) > max_rows:
        df = df.iloc[:max_rows]

    return df
