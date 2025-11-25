"""Stage 2 retrieval-augmented generation (RAG) utilities.

This module builds an embedding-based index for historical events and computes
numeric RAG-derived features for upcoming events based on retrieved similar
items.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


@dataclass
class EventRetrievalIndex:
    """Embedding-backed retrieval index for historical events."""

    text_column: str = "text"
    max_text_chars: int = 500
    model_name: str = "intfloat/e5-large-v2"
    model: Optional[SentenceTransformer] = field(default=None, init=False, repr=False)
    event_ids: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    embeddings: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    metadata: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)

    def fit(self, events_df: pd.DataFrame, model_name: str = "intfloat/e5-large-v2") -> "EventRetrievalIndex":
        """Build the retrieval index.

        Parameters
        ----------
        events_df:
            DataFrame containing historical events.
        model_name:
            Sentence-transformer model name to load.
        """

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        prepared_text = self._prepare_text(events_df)
        self.embeddings = self.model.encode(
            prepared_text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        self.event_ids = events_df["event_id"].astype(str).to_numpy()
        self.metadata = events_df.copy().reset_index(drop=True)
        return self

    def query(
        self,
        upcoming_event: dict,
        top_k: int = 50,
        filter_event_type: bool = False,
        filter_emotion: bool = False,
    ) -> pd.DataFrame:
        """Retrieve the most similar historical events.

        Parameters
        ----------
        upcoming_event:
            Mapping with keys such as "title", "location", "event_type", "channel", etc.
        top_k:
            Number of similar events to return.
        filter_event_type:
            If True, restrict retrievals to matching ``event_type``.
        filter_emotion:
            If True and an emotion label column is available, restrict to matching labels.
        """

        if self.model is None or self.embeddings is None or self.metadata is None:
            raise RuntimeError("The retrieval index has not been fitted yet.")

        candidate_indices = np.arange(len(self.metadata))
        if filter_event_type and "event_type" in upcoming_event and "event_type" in self.metadata:
            match = self.metadata["event_type"].fillna("") == str(upcoming_event.get("event_type", ""))
            candidate_indices = candidate_indices[match.to_numpy()]

        if filter_emotion and "emotion_label" in upcoming_event and "emotion_label" in self.metadata:
            match = self.metadata["emotion_label"].fillna("") == str(upcoming_event.get("emotion_label", ""))
            candidate_indices = candidate_indices[match.to_numpy()]

        if candidate_indices.size == 0:
            return self.metadata.head(0).copy()

        query_text = self._build_query_text(upcoming_event)
        query_embedding = self.model.encode(
            [query_text], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )[0]

        candidate_embeddings = self.embeddings[candidate_indices]
        similarities = np.dot(candidate_embeddings, query_embedding)

        top_indices = np.argsort(similarities)[::-1][:top_k]
        selected_rows = self.metadata.iloc[candidate_indices[top_indices]].copy()
        selected_rows.insert(1, "similarity", similarities[top_indices])

        return selected_rows.reset_index(drop=True)

    def _prepare_text(self, events_df: pd.DataFrame) -> List[str]:
        titles = events_df.get("title", "").fillna("")
        locations = events_df.get("location", "").fillna("")
        texts = events_df.get(self.text_column, "").fillna("").astype(str).str.slice(0, self.max_text_chars)
        return (titles + " " + locations + " " + texts).astype(str).tolist()

    def _build_query_text(self, upcoming_event: dict) -> str:
        title = str(upcoming_event.get("title", ""))
        location = str(upcoming_event.get("location", ""))
        text = str(upcoming_event.get(self.text_column, ""))[: self.max_text_chars]
        return f"{title} {location} {text}".strip()


def compute_rag_features_for_event(
    upcoming_event: dict,
    target_words: Iterable[str],
    retrieval_index: EventRetrievalIndex,
    top_k: int = 50,
) -> pd.DataFrame:
    """Compute RAG-derived numeric features for an upcoming event.

    Returns one row per target word with aggregated statistics drawn from the
    retrieved similar events.
    """

    retrieved_events = retrieval_index.query(upcoming_event, top_k=top_k)
    target_date = _safe_parse_date(upcoming_event.get("date"))

    rows = []
    for word in target_words:
        word_pattern = re.compile(rf"\b{re.escape(word)}\b", flags=re.IGNORECASE)
        contains_word = retrieved_events.get(retrieval_index.text_column, pd.Series(dtype=str)).fillna("").str.contains(
            word_pattern
        )

        freq = int(contains_word.sum())
        if freq:
            subset = retrieved_events[contains_word]
            most_recent_date = _most_recent_date(subset.get("date"), target_date)
            polarity_mean = subset.get("polarity", pd.Series(dtype=float)).mean()
            toxicity_mean = subset.get("toxicity", pd.Series(dtype=float)).mean()
        else:
            most_recent_date = np.nan
            polarity_mean = np.nan
            toxicity_mean = np.nan

        rows.append(
            {
                "word": word,
                "retrieved_freq": freq,
                "retrieved_last_date": most_recent_date,
                "retrieved_avg_polarity": polarity_mean,
                "retrieved_avg_toxicity": toxicity_mean,
            }
        )

    return pd.DataFrame(rows)


def _safe_parse_date(value: object) -> Optional[pd.Timestamp]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return pd.to_datetime(value)
    except Exception:
        return None


def _most_recent_date(dates: Optional[pd.Series], target_date: Optional[pd.Timestamp]) -> float:
    if dates is None or target_date is None:
        return np.nan

    parsed_dates = pd.to_datetime(dates, errors="coerce").dropna()
    if parsed_dates.empty:
        return np.nan

    delta = target_date - parsed_dates.max()
    return delta.total_seconds() / 86400.0
