"""Advisory trading decision module for binary prediction markets.

This module computes edge-based recommendations for trading binary
prediction markets while respecting strict risk caps. It produces
advisory-only recommendations and does not submit orders or interact
with external systems.
"""
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class TradeAction:
    """Represents an advisory trading action for a single market."""

    market_id: str
    recommended_action: str  # "BUY_YES", "BUY_NO", or "HOLD"
    edge: float
    size_shares: int
    size_dollars: float
    mode: str
    notes: str


def _market_implied_probability(price_yes: float) -> float:
    """Compute the market-implied probability for a YES contract.

    Currently, this returns the yes price directly without adjustments.
    """

    return price_yes


def _select_action(
    p_final: float,
    price_yes: float,
    side_preference: str,
    edge_threshold: float,
) -> Dict[str, float | str]:
    """Select the recommended action based on edge and preferences.

    Returns a dictionary with keys:
        - action: str ("BUY_YES", "BUY_NO", or "HOLD")
        - edge: float
        - price: float
        - notes: str
    """

    market_prob = _market_implied_probability(price_yes)
    edge_yes = p_final - market_prob
    edge_no = market_prob - p_final

    best_edge_magnitude = max(abs(edge_yes), abs(edge_no))
    if best_edge_magnitude < edge_threshold:
        return {"action": "HOLD", "edge": 0.0, "price": 0.0, "notes": "Edge below threshold"}

    if side_preference == "YES_ONLY":
        if edge_yes <= 0:
            return {"action": "HOLD", "edge": 0.0, "price": 0.0, "notes": "Positive edge not available for YES"}
        return {"action": "BUY_YES", "edge": edge_yes, "price": price_yes, "notes": "Buying YES per preference"}

    if side_preference == "NO_ONLY":
        if edge_no <= 0:
            return {"action": "HOLD", "edge": 0.0, "price": 0.0, "notes": "Positive edge not available for NO"}
        price_no = 1 - price_yes
        return {"action": "BUY_NO", "edge": edge_no, "price": price_no, "notes": "Buying NO per preference"}

    # ANY preference: choose side with positive edge
    if edge_yes > edge_no and edge_yes > 0:
        return {"action": "BUY_YES", "edge": edge_yes, "price": price_yes, "notes": "Highest positive edge on YES"}
    if edge_no > 0:
        price_no = 1 - price_yes
        return {"action": "BUY_NO", "edge": edge_no, "price": price_no, "notes": "Highest positive edge on NO"}

    return {"action": "HOLD", "edge": 0.0, "price": 0.0, "notes": "No positive edge despite threshold"}


def generate_trade_actions(
    bankroll: float,
    predictions: List[Dict],
    market_quotes: Dict[str, Dict],
    edge_threshold: float,
    max_position_fraction: float,
    max_total_risk_fraction: float,
    fee_rate: float,
) -> List[TradeAction]:
    """Generate advisory trade actions for each prediction.

    Parameters
    ----------
    bankroll:
        Total capital available, denominated in dollars.
    predictions:
        List of prediction dictionaries containing model outputs.
    market_quotes:
        Mapping of market identifiers to current quote information.
    edge_threshold:
        Minimum absolute edge required before recommending a trade.
    max_position_fraction:
        Maximum fraction of bankroll to allocate to any single contract.
    max_total_risk_fraction:
        Maximum fraction of bankroll to allocate across the entire batch.
    fee_rate:
        Estimated round-trip fee fraction on notional (unused placeholder
        for future adjustments).

    Returns
    -------
    List[TradeAction]
        Advisory actions corresponding to each prediction.
    """

    del fee_rate  # placeholder for future fee adjustments

    actions: List[TradeAction] = []
    allocated_fraction = 0.0

    for prediction in predictions:
        market_id = prediction["market_id"]
        p_final = prediction.get("p_final")

        quote = market_quotes.get(market_id)
        if quote is None:
            actions.append(
                TradeAction(
                    market_id=market_id,
                    recommended_action="HOLD",
                    edge=0.0,
                    size_shares=0,
                    size_dollars=0.0,
                    mode="ADVISORY",
                    notes="Missing market quotes",
                )
            )
            continue

        price_yes = quote.get("price_yes")
        side_pref = quote.get("side_preference", "ANY")
        if price_yes is None:
            actions.append(
                TradeAction(
                    market_id=market_id,
                    recommended_action="HOLD",
                    edge=0.0,
                    size_shares=0,
                    size_dollars=0.0,
                    mode="ADVISORY",
                    notes="Missing price information",
                )
            )
            continue

        decision = _select_action(p_final, price_yes, side_pref, edge_threshold)
        action = decision["action"]
        edge = decision["edge"]
        price = decision["price"]
        notes = decision["notes"]

        if action == "HOLD":
            actions.append(
                TradeAction(
                    market_id=market_id,
                    recommended_action=action,
                    edge=0.0,
                    size_shares=0,
                    size_dollars=0.0,
                    mode="ADVISORY",
                    notes=notes,
                )
            )
            continue

        remaining_fraction = max_total_risk_fraction - allocated_fraction
        if remaining_fraction <= 0:
            actions.append(
                TradeAction(
                    market_id=market_id,
                    recommended_action="HOLD",
                    edge=0.0,
                    size_shares=0,
                    size_dollars=0.0,
                    mode="ADVISORY",
                    notes="Risk cap reached",
                )
            )
            continue

        raw_size_fraction = min(max_position_fraction, remaining_fraction)
        size_dollars = raw_size_fraction * bankroll

        if price <= 0:
            actions.append(
                TradeAction(
                    market_id=market_id,
                    recommended_action="HOLD",
                    edge=0.0,
                    size_shares=0,
                    size_dollars=0.0,
                    mode="ADVISORY",
                    notes="Invalid price for position sizing",
                )
            )
            continue

        size_shares = int(round(size_dollars / price))
        if size_shares <= 0:
            actions.append(
                TradeAction(
                    market_id=market_id,
                    recommended_action="HOLD",
                    edge=0.0,
                    size_shares=0,
                    size_dollars=0.0,
                    mode="ADVISORY",
                    notes="Rounded position resulted in zero shares",
                )
            )
            continue

        allocated_fraction += raw_size_fraction

        actions.append(
            TradeAction(
                market_id=market_id,
                recommended_action=action,
                edge=edge,
                size_shares=size_shares,
                size_dollars=size_dollars,
                mode="ADVISORY",
                notes=notes,
            )
        )

    return actions


def actions_to_json(actions: List[TradeAction]) -> Dict[str, List[Dict]]:
    """Convert TradeAction objects to a JSON-friendly dictionary."""

    return {
        "actions": [
            {
                "market_id": action.market_id,
                "recommended_action": action.recommended_action,
                "edge": action.edge,
                "size_shares": action.size_shares,
                "size_dollars": action.size_dollars,
                "mode": action.mode,
                "notes": action.notes,
            }
            for action in actions
        ]
    }
