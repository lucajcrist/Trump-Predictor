"""Stage 3: Mixture-of-Experts router and LLM scoring helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Placeholder LLM client.

    External callers should replace this with a concrete implementation that
    invokes an LLM API. The function should return a JSON string that matches
    the schema described in ``build_user_prompt``.
    """

    del system_prompt, user_prompt
    return json.dumps({"simulated_snippet": "", "word_scores": []})


def choose_expert(event_type: str, channel: str) -> str:
    """
    Route an event to a conceptual expert persona based on metadata.

    Parameters
    ----------
    event_type : str
        Type of the event (e.g., "rally", "press_conference", "policy").
    channel : str
        Communication channel (e.g., "speech", "tv_interview", "tweet").

    Returns
    -------
    str
        One of "policy", "rally", "interview", or "social" according to the
        routing rules.
    """

    if event_type == "rally":
        return "rally"
    if event_type in ["press_conference", "policy"]:
        return "policy"
    if channel == "tv_interview":
        return "interview"
    return "policy"


SYSTEM_PROMPT_POLICY = (
    "You are a policy-focused Trump speech analyst. "
    "Stay grounded in the provided event description, candidate words, and retrieved snippets. "
    "Do not invent new phrases. Respond only with JSON containing 'simulated_snippet' and 'word_scores'."
)

SYSTEM_PROMPT_RALLY = (
    "You are modeling Trump's energetic rally rhetoric. "
    "Use the event description, candidate words, and retrieved snippets to guide tone. "
    "Do not invent new phrases. Respond only with JSON containing 'simulated_snippet' and 'word_scores'."
)

SYSTEM_PROMPT_INTERVIEW = (
    "You are modeling Trump's interview style for television conversations. "
    "Use the provided event description, candidate words, and retrieved snippets. "
    "Do not invent new phrases. Respond only with JSON containing 'simulated_snippet' and 'word_scores'."
)


def build_user_prompt(
    event_metadata: dict,
    candidate_words: List[dict],
    retrieved_snippets: Optional[List[dict]],
) -> str:
    """
    Construct the user prompt for the selected expert persona.

    Parameters
    ----------
    event_metadata : dict
        Dictionary containing event details such as title, event_type, channel,
        location, date, and known_topic.
    candidate_words : list of dict
        Candidate word entries sorted by ``p_baseline`` in descending order.
    retrieved_snippets : list of dict or None
        Optional snippets from past events to provide extra context.

    Returns
    -------
    str
        A formatted prompt instructing the LLM to produce a simulated snippet
        and per-word confidence scores in JSON format.
    """

    summary_lines = [
        "Upcoming event summary:",
        f"Title: {event_metadata.get('title', 'N/A')}",
        f"Type: {event_metadata.get('event_type', 'unknown')}",
        f"Channel: {event_metadata.get('channel', 'unknown')}",
        f"Location: {event_metadata.get('location', 'unknown')}",
        f"Date: {event_metadata.get('date', 'unknown')}",
    ]

    if event_metadata.get("known_topic"):
        summary_lines.append(f"Known topic: {event_metadata['known_topic']}")

    candidate_lines = ["Candidate words with baseline probabilities:"]
    for candidate in candidate_words:
        candidate_lines.append(
            f"- {candidate['word']}: p_baseline={candidate.get('p_baseline', 0):.4f}"
        )

    snippet_lines: List[str] = []
    if retrieved_snippets:
        snippet_lines.append("Retrieved snippets for context:")
        for snippet in retrieved_snippets:
            tags = snippet.get("tags", {})
            tag_str = ", ".join(f"{k}: {v}" for k, v in tags.items())
            snippet_lines.append(
                f"- From event {snippet.get('event_id', 'unknown')} ({tag_str}): {snippet.get('context', '')}"
            )

    instructions = (
        "Write a simulated 100-150 word snippet in the requested style. "
        "Then return JSON with fields: "
        "simulated_snippet (string) and word_scores (list of {word, llm_confidence in [0,1]}). "
        "Use the candidate words exactly as provided; do not add new words."
    )

    prompt_parts = summary_lines + [""] + candidate_lines
    if snippet_lines:
        prompt_parts += [""] + snippet_lines
    prompt_parts += ["", instructions]

    return "\n".join(prompt_parts)


@dataclass
class LLMWordScore:
    """Container for an LLM confidence score assigned to a candidate word."""

    word: str
    llm_confidence: float


def _parse_llm_response(
    response_text: str, candidate_words: List[Dict[str, str]]
) -> tuple[str, List[LLMWordScore]]:
    """Parse the LLM JSON response and align scores to candidate words."""

    simulated_snippet = ""
    llm_score_map: Dict[str, float] = {}

    try:
        payload = json.loads(response_text)
        if isinstance(payload, dict):
            simulated_snippet = str(payload.get("simulated_snippet", ""))
            for entry in payload.get("word_scores", []) or []:
                word = str(entry.get("word", ""))
                confidence = float(entry.get("llm_confidence", 0))
                llm_score_map[word] = min(1.0, max(0.0, confidence))
    except (json.JSONDecodeError, TypeError, ValueError):
        simulated_snippet = ""

    scores: List[LLMWordScore] = []
    for candidate in candidate_words:
        word = candidate["word"]
        confidence = llm_score_map.get(word, 0.05)
        scores.append(LLMWordScore(word=word, llm_confidence=confidence))

    return simulated_snippet, scores


def llm_score_candidates(
    event_metadata: dict,
    candidates: List[Dict],
    retrieved_snippets: Optional[List[Dict]] = None,
    llm_client=call_llm,
) -> tuple[str, List[LLMWordScore]]:
    """
    Route an event to an expert persona and request LLM confidence scores.

    Parameters
    ----------
    event_metadata : dict
        Event information consumed by the router and the prompt builder.
    candidates : list of dict
        Candidate words with baseline probabilities.
    retrieved_snippets : list of dict, optional
        Optional contextual snippets from prior events.
    llm_client : callable
        Function responsible for calling the LLM; defaults to ``call_llm``.

    Returns
    -------
    tuple[str, list[LLMWordScore]]
        The simulated snippet and aligned LLM confidence scores.
    """

    expert = choose_expert(event_metadata.get("event_type", ""), event_metadata.get("channel", ""))
    system_prompt = {
        "policy": SYSTEM_PROMPT_POLICY,
        "rally": SYSTEM_PROMPT_RALLY,
        "interview": SYSTEM_PROMPT_INTERVIEW,
        "social": SYSTEM_PROMPT_INTERVIEW,
    }.get(expert, SYSTEM_PROMPT_POLICY)

    user_prompt = build_user_prompt(event_metadata, candidates, retrieved_snippets or [])
    response_text = llm_client(system_prompt=system_prompt, user_prompt=user_prompt)

    return _parse_llm_response(response_text, candidates)


def blend_scores(
    candidates: List[Dict], llm_scores: List[LLMWordScore], alpha: float
) -> List[Dict]:
    """
    Blend baseline probabilities with LLM confidence scores.

    Parameters
    ----------
    candidates : list of dict
        Candidate words containing ``word`` and ``p_baseline`` keys.
    llm_scores : list of LLMWordScore
        Confidence scores returned by the LLM.
    alpha : float
        Weighting factor in ``[0, 1]`` for combining baseline and LLM outputs.

    Returns
    -------
    list of dict
        Combined scores for each candidate word.
    """

    llm_score_lookup = {score.word: score.llm_confidence for score in llm_scores}
    combined: List[Dict] = []
    for candidate in candidates:
        word = candidate["word"]
        p_baseline = float(candidate.get("p_baseline", 0))
        llm_confidence = float(llm_score_lookup.get(word, 0.05))
        p_final = alpha * p_baseline + (1 - alpha) * llm_confidence
        combined.append(
            {
                "word": word,
                "p_baseline": p_baseline,
                "llm_confidence": llm_confidence,
                "p_final": p_final,
            }
        )

    return combined
