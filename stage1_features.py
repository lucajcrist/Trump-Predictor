"""Stage 1 feature engineering and baseline model.

This module builds labels for target words, engineers numeric features, and trains a
simple baseline model. It is designed to be self-contained but allows injecting
external embedding functions for topic overlap computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score


def build_label_frame(events_df: pd.DataFrame, target_words: Sequence[str]) -> pd.DataFrame:
    """Construct labels for each (event, word) pair.

    Args:
        events_df: Event-level dataframe.
        target_words: Iterable of target phrases.

    Returns:
        DataFrame with columns [event_id, word, label, date, event_type, channel,
        location, country].
    """

    if "event_id" not in events_df or "text" not in events_df:
        raise ValueError("events_df must contain 'event_id' and 'text' columns")

    base_cols = [
        "event_id",
        "date",
        "event_type",
        "channel",
        "location",
        "country",
        "polarity",
        "toxicity",
        "text",
    ]
    missing = [c for c in base_cols if c not in events_df.columns]
    if missing:
        raise ValueError(f"events_df missing required columns: {missing}")

    events = events_df.copy()
    events["polarity"] = pd.to_numeric(events["polarity"], errors="coerce")
    events["toxicity"] = pd.to_numeric(events["toxicity"], errors="coerce")

    words_df = pd.DataFrame({"word": list(target_words)})
    words_df["key"] = 1

    events_small = events[base_cols].copy()
    events_small["key"] = 1

    cross = events_small.merge(words_df, on="key").drop(columns="key")
    cross["label"] = cross.apply(
        lambda row: int(row["word"].lower() in row["text"].lower()), axis=1
    )

    result = cross[
        [
            "event_id",
            "word",
            "label",
            "date",
            "event_type",
            "channel",
            "location",
            "country",
            "polarity",
            "toxicity",
        ]
    ].copy()
    result["date"] = pd.to_datetime(result["date"])
    return result


@dataclass
class _TfidfEmbedder:
    """Simple TF-IDF embedder used when no external embed_fn is provided."""

    vectorizer: Optional[TfidfVectorizer] = None

    def fit(self, texts: Sequence[str]) -> "_TfidfEmbedder":
        self.vectorizer = TfidfVectorizer(max_features=512)
        self.vectorizer.fit(texts)
        return self

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        if self.vectorizer is None:
            raise ValueError("Embedder not fitted")
        return self.vectorizer.transform(texts).toarray().astype(np.float32)


class FeatureEngineer:
    """Engineer numeric features for (event, word) pairs."""

    def __init__(self, embed_fn: Optional[Callable[[Sequence[str]], np.ndarray]] = None):
        self.embed_fn = embed_fn
        self.embedder: Optional[_TfidfEmbedder] = None
        self.target_words: List[str] = []
        self.macro_df: Optional[pd.DataFrame] = None
        self._feature_columns: List[str] = []

    def fit(
        self,
        events_df: pd.DataFrame,
        target_words: Sequence[str],
        macro_df: Optional[pd.DataFrame] = None,
    ) -> "FeatureEngineer":
        self.target_words = list(target_words)
        self.macro_df = None if macro_df is None else macro_df.copy()

        if self.embed_fn is None:
            title_source = events_df["title"] if "title" in events_df.columns else events_df["text"]
            self.embedder = _TfidfEmbedder().fit(title_source.fillna(""))
            self.embed_fn = self.embedder

        return self

    def _prepare_embeddings(self, events_df: pd.DataFrame) -> dict:
        if "title" in events_df.columns:
            title_source = events_df["title"].fillna("")
        else:
            title_source = events_df["text"].fillna("")
        titles = title_source.astype(str).tolist()
        embeddings = self.embed_fn(titles)
        return dict(zip(events_df["event_id"], embeddings))

    def _merge_macro(self, features_df: pd.DataFrame) -> pd.DataFrame:
        if self.macro_df is None or self.macro_df.empty:
            return features_df
        macro = self.macro_df.copy()
        if "date" not in macro.columns:
            macro = macro.reset_index().rename(columns={macro.index.name or "index": "date"})
        macro["date"] = pd.to_datetime(macro["date"])
        macro = macro.sort_values("date")
        merged = pd.merge_asof(
            features_df.sort_values("date"),
            macro,
            on="date",
            direction="backward",
        )
        return merged

    def transform(self, events_df: pd.DataFrame, target_words: Optional[Sequence[str]] = None) -> pd.DataFrame:
        if target_words is None:
            target_words = self.target_words
        if not target_words:
            raise ValueError("No target words provided")

        labels_df = build_label_frame(events_df, target_words)
        labels_df = labels_df.sort_values("date").reset_index(drop=True)

        embed_map = self._prepare_embeddings(events_df)
        features = []

        for word in target_words:
            word_df = labels_df[labels_df["word"] == word].copy().reset_index(drop=True)
            features.extend(self._compute_features_for_word(word_df, embed_map))

        features_df = pd.concat(features, ignore_index=True)
        features_df = self._merge_macro(features_df)

        # Track feature columns for downstream model usage
        self._feature_columns = [
            col
            for col in features_df.columns
            if col
            not in {
                "event_id",
                "word",
                "label",
                "date",
                "event_type",
                "channel",
                "location",
                "country",
            }
        ]
        return features_df

    def _compute_features_for_word(
        self, word_df: pd.DataFrame, embed_map: dict
    ) -> List[pd.DataFrame]:
        word_df = word_df.copy()
        word_df["date"] = pd.to_datetime(word_df["date"])

        pos_mask = word_df["label"] == 1
        pos_dates = word_df.loc[pos_mask, "date"].to_numpy()
        pos_event_types = word_df.loc[pos_mask, "event_type"].to_numpy()
        pos_locations = word_df.loc[pos_mask, "location"].fillna("").to_numpy()
        pos_countries = word_df.loc[pos_mask, "country"].fillna("").to_numpy()
        pos_polarity = (
            word_df.loc[pos_mask, "polarity"].to_numpy()
            if "polarity" in word_df
            else np.array([])
        )
        pos_toxicity = (
            word_df.loc[pos_mask, "toxicity"].to_numpy()
            if "toxicity" in word_df
            else np.array([])
        )
        pos_embeddings = np.array(
            [embed_map.get(eid) for eid in word_df.loc[pos_mask, "event_id"]], dtype=object
        )

        # Convert embeddings to numeric array if not empty
        if pos_embeddings.size and pos_embeddings[0] is not None:
            pos_embeddings = np.vstack(pos_embeddings)
            cumsum_embeddings = np.cumsum(pos_embeddings, axis=0)
        else:
            pos_embeddings = None
            cumsum_embeddings = None

        outputs = []
        all_dates = word_df["date"].to_numpy()
        pos_indices = np.flatnonzero(pos_mask.to_numpy())

        for idx, row in word_df.iterrows():
            current_date = row["date"]
            freq_7d = self._count_window(pos_dates, current_date, days=7)
            freq_30d = self._count_window(pos_dates, current_date, days=30)
            freq_90d = self._count_window(pos_dates, current_date, days=90)

            freq_eventtype_30d = self._count_window(
                pos_dates,
                current_date,
                days=30,
                extra_mask=pos_event_types == row.get("event_type"),
            )
            freq_location_365d = self._count_window(
                pos_dates,
                current_date,
                days=365,
                extra_mask=(pos_locations == str(row.get("location", "")))
                | (pos_countries == str(row.get("country", ""))),
            )

            last_use_idx = self._last_positive_index(pos_indices, idx)
            if last_use_idx is None:
                days_since_last_use = np.nan
                events_since_last_use = np.nan
            else:
                last_date = all_dates[last_use_idx]
                days_since_last_use = (current_date - last_date) / np.timedelta64(1, "D")
                events_since_last_use = idx - last_use_idx - 1

            topic_overlap = np.nan
            if cumsum_embeddings is not None and cumsum_embeddings.size:
                last_pos_before = np.searchsorted(pos_dates, current_date) - 1
                if last_pos_before >= 0:
                    historical_vec = cumsum_embeddings[last_pos_before] / float(
                        last_pos_before + 1
                    )
                    current_vec = embed_map.get(row["event_id"])
                    if current_vec is not None:
                        topic_overlap = float(
                            cosine_similarity(
                                current_vec.reshape(1, -1), historical_vec.reshape(1, -1)
                            )[0, 0]
                        )

            polarity_mean_30d = self._window_mean(
                pos_dates, pos_polarity, current_date, days=30
            )
            toxicity_mean_30d = self._window_mean(
                pos_dates, pos_toxicity, current_date, days=30
            )

            outputs.append(
                pd.DataFrame(
                    {
                        "event_id": [row["event_id"]],
                        "word": [row["word"]],
                        "label": [row["label"]],
                        "date": [row["date"]],
                        "event_type": [row.get("event_type")],
                        "channel": [row.get("channel")],
                        "location": [row.get("location")],
                        "country": [row.get("country")],
                        "freq_7d": [freq_7d],
                        "freq_30d": [freq_30d],
                        "freq_90d": [freq_90d],
                        "freq_eventtype_30d": [freq_eventtype_30d],
                        "freq_location_365d": [freq_location_365d],
                        "days_since_last_use": [days_since_last_use],
                        "events_since_last_use": [events_since_last_use],
                        "topic_overlap": [topic_overlap],
                        "polarity_mean_30d": [polarity_mean_30d],
                        "toxicity_mean_30d": [toxicity_mean_30d],
                    }
                )
            )

        return outputs

    @staticmethod
    def _count_window(
        dates: np.ndarray,
        current_date: pd.Timestamp,
        days: int,
        extra_mask: Optional[np.ndarray] = None,
    ) -> int:
        if dates.size == 0:
            return 0
        start = current_date - np.timedelta64(days, "D")
        end_idx = np.searchsorted(dates, current_date)
        start_idx = np.searchsorted(dates, start)
        if extra_mask is None:
            return int(end_idx - start_idx)
        return int(np.count_nonzero(extra_mask[start_idx:end_idx]))

    @staticmethod
    def _window_mean(
        dates: np.ndarray,
        values: np.ndarray,
        current_date: pd.Timestamp,
        days: int,
    ) -> float:
        if dates.size == 0 or values.size == 0:
            return np.nan
        start = current_date - np.timedelta64(days, "D")
        end_idx = np.searchsorted(dates, current_date)
        start_idx = np.searchsorted(dates, start)
        if end_idx == start_idx:
            return np.nan
        return float(np.nanmean(values[start_idx:end_idx]))

    @staticmethod
    def _last_positive_index(pos_indices: np.ndarray, current_idx: int) -> Optional[int]:
        if pos_indices.size == 0:
            return None
        pos = np.searchsorted(pos_indices, current_idx) - 1
        if pos < 0:
            return None
        return int(pos_indices[pos])

    @property
    def feature_columns(self) -> List[str]:
        return self._feature_columns


class BaselineModel:
    """Baseline classifier using gradient boosting."""

    def __init__(self):
        self.model = GradientBoostingClassifier(random_state=42)
        self.feature_columns: List[str] = []

    def fit(self, features_df: pd.DataFrame, label_col: str = "label") -> "BaselineModel":
        if label_col not in features_df.columns:
            raise ValueError(f"label column '{label_col}' not found")

        meta_cols = {
            "event_id",
            "word",
            "label",
            "date",
            "event_type",
            "channel",
            "location",
            "country",
        }
        feature_cols = [c for c in features_df.columns if c not in meta_cols]
        self.feature_columns = feature_cols

        X = features_df[feature_cols].fillna(0.0)
        y = features_df[label_col].values

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        self.model.fit(X_train, y_train)
        if len(np.unique(y_val)) > 1:
            preds = self.model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, preds)
            print(f"Validation ROC-AUC: {auc:.4f}")
        return self

    def predict_proba(self, features_df: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_columns:
            raise ValueError("Model not fitted")
        X = features_df[self.feature_columns].fillna(0.0)
        probs = self.model.predict_proba(X)[:, 1]
        return pd.DataFrame(
            {
                "event_id": features_df["event_id"],
                "word": features_df["word"],
                "p_baseline": probs,
            }
        )


def train_baseline(
    events_df: pd.DataFrame,
    target_words: Sequence[str],
    macro_df: Optional[pd.DataFrame] = None,
    embed_fn: Optional[Callable[[Sequence[str]], np.ndarray]] = None,
) -> Tuple[FeatureEngineer, BaselineModel]:
    """Build labels, engineer features, and train the baseline model."""

    feature_engineer = FeatureEngineer(embed_fn=embed_fn)
    feature_engineer.fit(events_df, target_words, macro_df=macro_df)
    features_df = feature_engineer.transform(events_df, target_words)

    baseline_model = BaselineModel()
    baseline_model.fit(features_df)

    return feature_engineer, baseline_model


__all__ = [
    "build_label_frame",
    "FeatureEngineer",
    "BaselineModel",
    "train_baseline",
]
