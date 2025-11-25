"""Utility helpers for feature augmentation."""

from __future__ import annotations

import pandas as pd


def augment_feature_frame_with_rag_features(
    base_feature_df: pd.DataFrame,
    rag_feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge base model features with retrieved (RAG) statistics for a single event.

    Parameters
    ----------
    base_feature_df : pd.DataFrame
        DataFrame containing at least ``event_id`` and ``word`` columns for a
        single upcoming event.
    rag_feature_df : pd.DataFrame
        DataFrame containing retrieved contextual statistics with columns
        ``word``, ``retrieved_freq``, ``retrieved_last_date``,
        ``retrieved_avg_polarity``, and ``retrieved_avg_toxicity``.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the same rows as ``base_feature_df`` and appended RAG
        feature columns. Missing RAG values are filled with ``0`` to ensure all
        outputs remain numeric for downstream models.
    """

    merged = base_feature_df.merge(rag_feature_df, on="word", how="left")

    defaults = {
        "retrieved_freq": 0,
        "retrieved_last_date": 0,
        "retrieved_avg_polarity": 0.0,
        "retrieved_avg_toxicity": 0.0,
    }

    for column, default in defaults.items():
        if column not in merged.columns:
            merged[column] = default
        else:
            merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(default)

    return merged
