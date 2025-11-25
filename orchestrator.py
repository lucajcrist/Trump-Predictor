"""
End-to-end orchestration for the Trump predictor pipeline.

This module stitches together four pipeline stages plus optional backtesting:
- Stage 1 (stage1_features): feature generation and baseline numeric model
- Stage 2 (stage2_rag): retrieval-augmented numeric feature computation
- Stage 3 (stage3_llm_moe): LLM mixture-of-experts scoring for tie-breaking
- Stage 4 (stage4_trading): trading advisor that consumes final probabilities
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

import backtest
import stage1_features
import stage2_rag
import stage3_llm_moe
import stage4_trading


@dataclass
class PipelineConfig:
    """Configuration hyperparameters for the pipeline."""

    alpha: float = 0.7
    shortlist_top_k: int = 5
    shortlist_threshold: float = 0.1
    run_backtest: bool = False

    def as_dict(self) -> Dict[str, Any]:
        """Expose config as a plain dictionary for logging or serialization."""

        return {
            "alpha": self.alpha,
            "shortlist_top_k": self.shortlist_top_k,
            "shortlist_threshold": self.shortlist_threshold,
            "run_backtest": self.run_backtest,
        }


@dataclass
class EventMetadata:
    """Minimal event metadata passed between stages."""

    event_id: str
    date: pd.Timestamp
    event_type: str
    channel: str
    location: str
    country: str
    title: str
    text: str
    polarity: float
    toxicity: float

    def to_frame(self) -> pd.DataFrame:
        """Convert the event metadata into a single-row DataFrame."""
        return pd.DataFrame([self.__dict__])


@dataclass
class WordPrediction:
    """Container for per-word predictions across stages."""

    word: str
    p_baseline: float
    llm_confidence: Optional[float]
    p_final: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary suitable for trading or logging layers."""

        return {
            "word": self.word,
            "p_baseline": self.p_baseline,
            "llm_confidence": self.llm_confidence,
            "p_final": self.p_final,
        }


def load_historical_events(path: Path) -> pd.DataFrame:
    """Load historical events used to fit the baseline model."""

    events = pd.read_csv(path, parse_dates=["date"])
    required_cols = {
        "event_id",
        "date",
        "event_type",
        "channel",
        "location",
        "country",
        "title",
        "text",
        "polarity",
        "toxicity",
    }
    missing = required_cols.difference(events.columns)
    if missing:
        raise ValueError(f"Historical events file missing required columns: {sorted(missing)}")
    return events


def load_word_list(path: Path) -> List[str]:
    """Load the list of target words/phrases, one per line."""

    with path.open("r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]
    if not words:
        raise ValueError("Word list is empty; provide at least one target word/phrase.")
    return words


def train_baseline_model(
    historical_events: pd.DataFrame, word_list: Sequence[str]
) -> Any:
    """Train the baseline numeric model using historical data."""

    return stage1_features.train_baseline_model(
        events_df=historical_events, word_list=list(word_list)
    )


def shortlist_candidates(
    baseline_preds: pd.DataFrame, config: PipelineConfig
) -> List[str]:
    """Shortlist candidates based on baseline probabilities."""

    filtered = baseline_preds[baseline_preds["p_baseline"] >= config.shortlist_threshold]
    sorted_preds = filtered.sort_values("p_baseline", ascending=False)
    return sorted_preds.head(config.shortlist_top_k)["word"].tolist()


def _validate_baseline_df(df: pd.DataFrame, *, allow_empty: bool = False) -> None:
    """Ensure baseline predictions contain the expected schema."""

    required_cols = {"event_id", "word", "p_baseline"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Baseline predictions missing required columns: {sorted(missing)}")
    if not allow_empty and df.empty:
        raise ValueError("Baseline predictions are empty; cannot continue pipeline.")


def blend_probabilities(
    baseline_df: pd.DataFrame, llm_scores: Dict[str, float], alpha: float
) -> List[WordPrediction]:
    """Blend baseline probabilities with LLM confidence for shortlisted words."""

    predictions: List[WordPrediction] = []
    for _, row in baseline_df.iterrows():
        word = row["word"]
        p_baseline = float(row["p_baseline"])
        llm_confidence = llm_scores.get(word)
        if llm_confidence is not None:
            p_final = alpha * p_baseline + (1 - alpha) * llm_confidence
        else:
            p_final = p_baseline
        predictions.append(
            WordPrediction(
                word=word,
                p_baseline=p_baseline,
                llm_confidence=llm_confidence,
                p_final=p_final,
            )
        )
    return predictions


def run_pipeline_for_event(
    event_metadata: EventMetadata,
    model_artifacts: Any,
    market_data: Dict[str, float],
    word_list: Sequence[str],
    config: PipelineConfig,
    historical_events: Optional[pd.DataFrame] = None,
) -> Tuple[List[WordPrediction], Any]:
    """Run the full pipeline for a single event.

    Returns blended probabilities for every word alongside trading guidance.
    """

    event_df = event_metadata.to_frame()

    baseline_preds = stage1_features.predict_baseline(
        event_df=event_df, word_list=list(word_list), model_artifacts=model_artifacts
    )
    _validate_baseline_df(baseline_preds)

    rag_features = stage2_rag.build_rag_features(
        event_df=event_df,
        word_list=list(word_list),
        historical_events=historical_events,
    )

    if rag_features is not None and not rag_features.empty:
        baseline_preds = stage1_features.predict_baseline(
            event_df=event_df,
            word_list=list(word_list),
            model_artifacts=model_artifacts,
            additional_features=rag_features,
        )
        _validate_baseline_df(baseline_preds)

    candidate_words = shortlist_candidates(baseline_preds, config=config)

    llm_scores = stage3_llm_moe.llm_score_candidates(
        event_metadata=event_metadata,
        candidate_words=candidate_words,
        metadata={
            "event_type": event_metadata.event_type,
            "channel": event_metadata.channel,
            "location": event_metadata.location,
        },
    )

    predictions = blend_probabilities(
        baseline_df=baseline_preds, llm_scores=llm_scores, alpha=config.alpha
    )

    trading_inputs = [pred.to_dict() for pred in predictions]
    trading_actions = stage4_trading.generate_trade_recommendations(
        predictions=trading_inputs, market_prices=market_data
    )

    return predictions, trading_actions


def main() -> None:
    """Entry point for training and running the pipeline for a new event."""

    config = PipelineConfig()
    data_dir = Path("data")
    historical_path = data_dir / "events.csv"
    word_list_path = data_dir / "word_list.txt"

    historical_events = load_historical_events(historical_path)
    word_list = load_word_list(word_list_path)

    model_artifacts = train_baseline_model(
        historical_events=historical_events, word_list=word_list
    )

    latest_event_row = historical_events.sort_values("date").iloc[-1].to_dict()
    event_metadata = EventMetadata(**latest_event_row)

    market_data: Dict[str, float] = {word: 0.5 for word in word_list}

    predictions, trading_actions = run_pipeline_for_event(
        event_metadata=event_metadata,
        model_artifacts=model_artifacts,
        market_data=market_data,
        word_list=word_list,
        config=config,
        historical_events=historical_events,
    )

    print("Predictions:")
    for pred in predictions:
        print(pred)

    print("Trading actions:", trading_actions)

    if config.run_backtest:
        backtest.run_backtest(
            historical_events=historical_events,
            word_list=word_list,
            pipeline_runner=run_pipeline_for_event,
            model_artifacts=model_artifacts,
            config=config,
        )


if __name__ == "__main__":
    main()
