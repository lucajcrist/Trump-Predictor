"""Backtesting and ablation harness for the word-prediction system."""
from __future__ import annotations

import dataclasses
import datetime as dt
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from stage1_features import BaselineModel, FeatureEngineer
from stage2_rag import EventRetrievalIndex, compute_rag_features_for_event
from stage3_llm_moe import blend_scores, llm_score_candidates
from stage4_trading import generate_trade_actions

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def brier_score(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    """Compute the Brier score between true labels and predicted probabilities."""
    if y_true.shape != p_pred.shape:
        raise ValueError("Shapes of y_true and p_pred must match")
    return float(np.mean((p_pred - y_true) ** 2))


def calibration_curve(
    y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Calculate calibration bins for probability predictions.

    Returns a DataFrame with lower/upper bin edges, average predicted
    probability, observed frequency, and count per bin.
    """
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(p_pred, bins, right=True)

    rows = []
    for i in range(1, len(bins)):
        mask = bin_indices == i
        if not np.any(mask):
            rows.append(
                {
                    "bin_lower": bins[i - 1],
                    "bin_upper": bins[i],
                    "avg_pred": np.nan,
                    "avg_true": np.nan,
                    "count": 0,
                }
            )
            continue
        rows.append(
            {
                "bin_lower": bins[i - 1],
                "bin_upper": bins[i],
                "avg_pred": float(np.mean(p_pred[mask])),
                "avg_true": float(np.mean(y_true[mask])),
                "count": int(np.sum(mask)),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class BacktestConfig:
    """Configuration for the rolling window backtest."""

    horizon_days: int = 30
    step_days: int = 30
    warmup_days: int = 90
    alpha: float = 0.7
    shortlist_top_k: int = 8
    shortlist_threshold: float = 0.4


@dataclasses.dataclass
class PredictionResult:
    event_id: str
    word: str
    date: dt.datetime
    p_pred: float
    y_true: int
    mode: str


@dataclasses.dataclass
class PnLSeries:
    pnl_df: pd.DataFrame
    summary: Dict[str, float]


# ---------------------------------------------------------------------------
# Core backtest logic
# ---------------------------------------------------------------------------


def _split_by_time(
    events_df: pd.DataFrame,
    config: BacktestConfig,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield train/test splits using a simple rolling scheme."""
    if "date" not in events_df.columns:
        raise ValueError("events_df must include a 'date' column")

    events_df = events_df.sort_values("date").reset_index(drop=True)
    start_date = events_df["date"].min() + dt.timedelta(days=config.warmup_days)
    end_date = events_df["date"].max()

    splits: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    train_end = start_date
    while train_end < end_date:
        test_end = train_end + dt.timedelta(days=config.horizon_days)
        train_mask = events_df["date"] <= train_end
        test_mask = (events_df["date"] > train_end) & (events_df["date"] <= test_end)

        train_df = events_df.loc[train_mask]
        test_df = events_df.loc[test_mask]
        if not len(test_df):
            train_end = test_end
            continue
        splits.append((train_df, test_df))
        train_end = train_end + dt.timedelta(days=config.step_days)
    return splits


def _fit_stage1(
    train_df: pd.DataFrame,
    target_words: List[str],
) -> Tuple[FeatureEngineer, BaselineModel]:
    """Train the baseline model on historical events."""
    fe = FeatureEngineer()
    model = BaselineModel()

    if hasattr(fe, "fit_transform"):
        X_train, y_train = fe.fit_transform(train_df, target_words)
    elif hasattr(fe, "build_features"):
        X_train, y_train = fe.build_features(train_df, target_words)
    else:
        raise AttributeError("FeatureEngineer must implement fit_transform or build_features")

    if hasattr(model, "fit"):
        model.fit(X_train, y_train)
    else:
        raise AttributeError("BaselineModel must implement fit")
    return fe, model


def _predict_stage1(
    fe: FeatureEngineer,
    model: BaselineModel,
    test_df: pd.DataFrame,
    target_words: List[str],
) -> pd.DataFrame:
    """Generate baseline predictions for test events."""
    if hasattr(fe, "transform"):
        X_test, labels = fe.transform(test_df, target_words)
    elif hasattr(fe, "build_features"):
        X_test, labels = fe.build_features(test_df, target_words)
    else:
        raise AttributeError("FeatureEngineer must implement transform or build_features")

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)
    elif hasattr(model, "predict"):
        probs = model.predict(X_test)
    else:
        raise AttributeError("BaselineModel must implement predict_proba or predict")

    preds = pd.DataFrame(labels, columns=["event_id", "word", "y_true", "date"])
    preds["p_baseline"] = probs
    return preds


def _augment_with_rag(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    preds: pd.DataFrame,
    target_words: List[str],
) -> pd.DataFrame:
    """Compute RAG features per event and join with baseline outputs."""
    try:
        if hasattr(EventRetrievalIndex, "from_events"):
            index = EventRetrievalIndex.from_events(train_df, target_words=target_words)
        else:
            index = EventRetrievalIndex(train_df, target_words=target_words)
    except TypeError:
        index = EventRetrievalIndex(train_df)

    rag_rows = []
    for event_id in preds["event_id"].unique():
        event_row = test_df.loc[test_df["event_id"] == event_id].iloc[0]
        rag_features = compute_rag_features_for_event(index, event_row, target_words)
        for word, score in rag_features.items():
            rag_rows.append({"event_id": event_id, "word": word, "rag_score": score})
    rag_df = pd.DataFrame(rag_rows)
    merged = preds.merge(rag_df, on=["event_id", "word"], how="left")
    merged["p_rag"] = merged["rag_score"].fillna(0.0)
    return merged


def _apply_llm_blending(
    test_df: pd.DataFrame,
    preds: pd.DataFrame,
    target_words: List[str],
    alpha: float,
    shortlist_top_k: int,
    shortlist_threshold: float,
) -> pd.DataFrame:
    """Run LLM rescoring for shortlisted candidates and blend with baseline."""
    blended_rows = []
    for event_id, group in preds.groupby("event_id"):
        event_row = test_df.loc[test_df["event_id"] == event_id].iloc[0]
        candidates = (
            group.sort_values("p_baseline", ascending=False)
            .head(shortlist_top_k)["word"].tolist()
        )
        candidates += [w for w, p in zip(group["word"], group["p_baseline"]) if p >= shortlist_threshold]
        candidates = sorted(set(candidates))

        try:
            llm_scores = llm_score_candidates(event_row, candidates)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("LLM scoring failed, falling back to uniform scores: %s", exc)
            llm_scores = {word: 0.5 for word in candidates}

        for _, row in group.iterrows():
            word = row["word"]
            llm_score = llm_scores.get(word, 0.0)
            blended = blend_scores(row["p_baseline"], llm_score, alpha=alpha)
            blended_rows.append({
                "event_id": event_id,
                "word": word,
                "p_baseline": row["p_baseline"],
                "llm_score": llm_score,
                "p_final": blended,
                "y_true": row["y_true"],
                "date": row["date"],
            })
    return pd.DataFrame(blended_rows)


def _compute_hit_rate_top1(preds: pd.DataFrame) -> float:
    """Fraction of events where the top predicted word occurred."""
    hit_count = 0
    total = 0
    for event_id, group in preds.groupby("event_id"):
        top_word = group.sort_values("p_pred", ascending=False).iloc[0]
        if int(top_word["y_true"]) == 1:
            hit_count += 1
        total += 1
    return float(hit_count / total) if total else 0.0


def _simulate_pnl(
    preds: pd.DataFrame,
    market_data_df: pd.DataFrame,
) -> PnLSeries:
    """Simulate trading PnL given predictions and market data."""
    if not len(market_data_df):
        pnl_df = pd.DataFrame(columns=["date", "pnl", "cumulative_pnl"])
        return PnLSeries(pnl_df=pnl_df, summary={"total_pnl": 0.0})

    if "word" not in market_data_df.columns:
        raise ValueError("market_data_df must include a 'word' column to map predictions")

    merged = preds.merge(
        market_data_df,
        on=["event_id", "word"],
        how="left",
        suffixes=('', '_market'),
    )
    merged = merged.dropna(subset=["market_id"])

    trade_actions = []
    for _, row in merged.iterrows():
        trade_actions.extend(
            generate_trade_actions(
                event_id=row["event_id"],
                market_id=row["market_id"],
                prob=row["p_pred"],
                price_yes_open=row["price_yes_open"],
                price_yes_close=row["price_yes_close"],
                resolved_outcome=row["resolved_outcome"],
                timestamp=row.get("date"),
            )
        )

    pnl_df = pd.DataFrame(trade_actions)
    if not pnl_df.empty:
        pnl_df = pnl_df.sort_values("timestamp")
        pnl_df["cumulative_pnl"] = pnl_df["pnl"].cumsum()
        summary = {"total_pnl": float(pnl_df["pnl"].sum())}
    else:
        pnl_df = pd.DataFrame(columns=["timestamp", "pnl", "cumulative_pnl"])
        summary = {"total_pnl": 0.0}

    pnl_df = pnl_df.rename(columns={"timestamp": "date"})
    return PnLSeries(pnl_df=pnl_df, summary=summary)


def _evaluate_mode(
    mode: str,
    events_df: pd.DataFrame,
    target_words: List[str],
    market_data_df: pd.DataFrame,
    config: BacktestConfig,
) -> Dict[str, object]:
    """Run the backtest for a single ablation mode."""
    preds_all: List[PredictionResult] = []
    for train_df, test_df in _split_by_time(events_df, config):
        fe, model = _fit_stage1(train_df, target_words)
        baseline_preds = _predict_stage1(fe, model, test_df, target_words)

        if mode == "baseline":
            stage_preds = baseline_preds.rename(columns={"p_baseline": "p_pred"})
        elif mode == "baseline_rag":
            rag_preds = _augment_with_rag(train_df, test_df, baseline_preds, target_words)
            rag_preds["p_pred"] = rag_preds[["p_baseline", "p_rag"]].mean(axis=1)
            stage_preds = rag_preds
        elif mode == "full":
            rag_preds = _augment_with_rag(train_df, test_df, baseline_preds, target_words)
            blended = _apply_llm_blending(
                test_df,
                rag_preds,
                target_words,
                alpha=config.alpha,
                shortlist_top_k=config.shortlist_top_k,
                shortlist_threshold=config.shortlist_threshold,
            )
            blended = blended.rename(columns={"p_final": "p_pred"})
            stage_preds = blended
        else:
            raise ValueError("mode must be one of {'baseline', 'baseline_rag', 'full'}")

        for _, row in stage_preds.iterrows():
            preds_all.append(
                PredictionResult(
                    event_id=row["event_id"],
                    word=row["word"],
                    date=row["date"],
                    p_pred=float(row["p_pred"]),
                    y_true=int(row["y_true"]),
                    mode=mode,
                )
            )

    preds_df = pd.DataFrame(dataclasses.asdict(p) for p in preds_all)
    if preds_df.empty:
        logger.warning("No predictions generated for mode=%s", mode)
        return {
            "brier_score": np.nan,
            "calibration": calibration_curve(np.array([]), np.array([])),
            "hit_rate_top1": np.nan,
            "pnl_timeseries": pd.DataFrame(),
            "summary": {},
        }

    brier = brier_score(preds_df["y_true"].values, preds_df["p_pred"].values)
    calib = calibration_curve(preds_df["y_true"].values, preds_df["p_pred"].values)
    hit_rate = _compute_hit_rate_top1(preds_df)
    pnl = _simulate_pnl(preds_df, market_data_df)

    return {
        "brier_score": brier,
        "calibration": calib,
        "hit_rate_top1": hit_rate,
        "pnl_timeseries": pnl.pnl_df,
        "summary": {
            "total_pnl": pnl.summary.get("total_pnl", 0.0),
            "n_predictions": len(preds_df),
        },
        "predictions": preds_df,
    }


def run_backtest(
    events_df: pd.DataFrame,
    target_words: List[str],
    market_data_df: pd.DataFrame,
    mode: str = "full",
    alpha: float = 0.7,
    shortlist_top_k: int = 8,
    shortlist_threshold: float = 0.4,
) -> Dict[str, object]:
    """Run rolling backtests for multiple ablation modes.

    Args:
        events_df: Historical event-word level dataframe. Expected columns include
            ``event_id``, ``date``, ``word``, and ``y_true`` (label indicating
            whether the word appeared in the event speech/transcript).
        target_words: List of candidate phrases to score.
        market_data_df: Market prices and resolutions with a ``word`` column for
            alignment to predictions.
        mode: Which mode to emphasize in the return structure; all modes are
            still evaluated.
        alpha: Blending weight for baseline vs. LLM score.
        shortlist_top_k: Top-k shortlist size for LLM rescoring.
        shortlist_threshold: Minimum baseline probability for LLM consideration.

    Returns:
        Dictionary containing metrics per mode and a concise summary table.
    """
    config = BacktestConfig(
        alpha=alpha,
        shortlist_top_k=shortlist_top_k,
        shortlist_threshold=shortlist_threshold,
    )

    modes = ["baseline", "baseline_rag", "full"]
    metrics = {}
    summary_rows = []
    for m in modes:
        logger.info("Running backtest mode=%s", m)
        result = _evaluate_mode(m, events_df, target_words, market_data_df, config)
        metrics[m] = result
        summary_rows.append(
            {
                "mode": m,
                "brier_score": result.get("brier_score"),
                "total_pnl": result.get("summary", {}).get("total_pnl"),
                "hit_rate_top1": result.get("hit_rate_top1"),
                "n_predictions": result.get("summary", {}).get("n_predictions", 0),
            }
        )

    summary_table = pd.DataFrame(summary_rows).set_index("mode")
    logger.info("Ablation summary:\n%s", summary_table)

    return {
        "modes": metrics,
        "summary_table": summary_table,
        "selected_mode": mode,
        "selected_results": metrics.get(mode, {}),
    }


__all__ = [
    "BacktestConfig",
    "PredictionResult",
    "PnLSeries",
    "brier_score",
    "calibration_curve",
    "run_backtest",
]
