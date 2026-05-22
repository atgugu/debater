"""Filesystem-backed debate state: directory layout, state.json I/O, debate listing."""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone

from debater.config import DEFAULT_EFFORTS, DEFAULT_MODELS

log = logging.getLogger("debater")


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sanitize_claim(claim: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", claim.lower()).strip("_")[:50]
    return slug or "debate"


def get_debate_dir(debates_dir: str, debate_id: str) -> str | None:
    safe_id = os.path.basename(debate_id)
    if not safe_id or safe_id.startswith(".") or "/" in debate_id or "\\" in debate_id:
        return None
    return os.path.join(debates_dir, safe_id)


def load_state(debate_dir: str | None) -> dict | None:
    if debate_dir is None:
        return None
    path = os.path.join(debate_dir, "state.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save_state(debate_dir: str, state: dict) -> None:
    """Atomically write state.json."""
    state["updated_at"] = now_iso()
    path = os.path.join(debate_dir, "state.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def create_debate(debates_dir: str, claim: str, settings: dict) -> str:
    """Create a new debate directory and initial state.json. Returns debate_id."""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    slug = sanitize_claim(claim)
    debate_id = f"{date_str}_{slug}"

    debate_dir = get_debate_dir(debates_dir, debate_id)
    counter = 1
    while os.path.exists(debate_dir):
        debate_id = f"{date_str}_{slug}_{counter}"
        debate_dir = get_debate_dir(debates_dir, debate_id)
        counter += 1

    os.makedirs(debate_dir, exist_ok=True)

    merged_models = dict(DEFAULT_MODELS)
    merged_efforts = dict(DEFAULT_EFFORTS)
    if "models" in settings:
        merged_models.update(settings["models"])
    if "efforts" in settings:
        merged_efforts.update(settings["efforts"])

    state = {
        "debate_id": debate_id,
        "claim": claim,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "status": "pending",
        "settings": {
            "max_iterations": settings.get("max_iterations", 5),
            "early_stop_threshold": settings.get("early_stop_threshold", 3),
            "models": merged_models,
            "efforts": merged_efforts,
            "max_atomic_claims_per_side": settings.get("max_atomic_claims_per_side", 4),
            "max_questions_per_claim": settings.get("max_questions_per_claim", 2),
            "evaluator_samples": settings.get("evaluator_samples", 3),
        },
        "current_round": 0,
        "early_stopped": False,
        "rounds": [],
        "user_messages": [],
        "last_seen_user_msg_idx": 0,
        "verdict": None,
        "confidence": None,
        "summary": None,
        "active_agent": None,
        "agent_status": None,
        "prior_debates": settings.get("prior_debates", []),
        "claim_class": None,
        "analysis_entry_reason": None,
        "verdict_confidence": None,
        "key_claims_cited": None,
        "verdict_guard_applied": False,
        "verdict_guard_reason": None,
    }
    save_state(debate_dir, state)
    return debate_id


def list_debates(debates_dir: str) -> list[dict]:
    """Scan debates_dir and return summary list of all debates."""
    debates: list[dict] = []
    if not os.path.isdir(debates_dir):
        return debates
    for name in sorted(os.listdir(debates_dir), reverse=True):
        debate_dir = os.path.join(debates_dir, name)
        if not os.path.isdir(debate_dir):
            continue
        state = load_state(debate_dir)
        if not state:
            continue
        latest_score = None
        rounds_with_eval = [r for r in state.get("rounds", []) if r.get("evaluator")]
        if rounds_with_eval:
            e = rounds_with_eval[-1]["evaluator"]
            latest_score = {"defender": e["defender_score"], "skeptic": e["skeptic_score"]}
        debates.append({
            "id": state["debate_id"],
            "claim": state["claim"],
            "title": state.get("title", ""),
            "created_at": state["created_at"],
            "status": state["status"],
            "current_round": state["current_round"],
            "max_iterations": state["settings"]["max_iterations"],
            "verdict": state.get("verdict"),
            "latest_score": latest_score,
            "active_agent": state.get("active_agent"),
            "agent_status": state.get("agent_status"),
        })
    debates.sort(key=lambda d: d["created_at"], reverse=True)
    return debates


def read_file(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except (FileNotFoundError, PermissionError):
        return ""


def append_md(filepath: str, header: str, content: str) -> None:
    """Append a markdown section to a file."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"## {header}\n\n")
        f.write(content)
        f.write("\n\n---\n\n")
