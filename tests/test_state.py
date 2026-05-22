"""Tests for filesystem-backed debate state."""
from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

from debater.state import (
    create_debate,
    get_debate_dir,
    list_debates,
    load_state,
    sanitize_claim,
    save_state,
)


class TestSanitizeClaim:
    def test_lowercase_words(self):
        assert sanitize_claim("Trump will win") == "trump_will_win"

    def test_strips_punctuation(self):
        assert sanitize_claim("S&P 500 will drop!") == "s_p_500_will_drop"

    def test_collapses_repeated_separators(self):
        assert sanitize_claim("hello---world!!!foo") == "hello_world_foo"

    def test_strips_leading_trailing_underscores(self):
        assert sanitize_claim("!!!hello!!!") == "hello"

    def test_empty_returns_fallback(self):
        assert sanitize_claim("") == "debate"

    def test_only_punctuation_returns_fallback(self):
        assert sanitize_claim("!!!???") == "debate"

    def test_unicode_stripped(self):
        # Non-ascii letters are stripped by the [^a-z0-9] regex
        assert sanitize_claim("café résumé") == "caf_r_sum"

    def test_caps_at_50_chars(self):
        long = "x" * 200
        out = sanitize_claim(long)
        assert len(out) <= 50


class TestGetDebateDir:
    def test_returns_joined_path(self, tmp_debates_dir):
        assert get_debate_dir(tmp_debates_dir, "2026-01-01_foo") == os.path.join(
            tmp_debates_dir, "2026-01-01_foo"
        )

    def test_rejects_parent_traversal(self, tmp_debates_dir):
        assert get_debate_dir(tmp_debates_dir, "../etc/passwd") is None

    def test_rejects_absolute_path(self, tmp_debates_dir):
        assert get_debate_dir(tmp_debates_dir, "/etc/passwd") is None

    def test_rejects_backslash(self, tmp_debates_dir):
        assert get_debate_dir(tmp_debates_dir, "foo\\bar") is None

    def test_rejects_leading_dot(self, tmp_debates_dir):
        assert get_debate_dir(tmp_debates_dir, ".hidden") is None

    def test_rejects_empty(self, tmp_debates_dir):
        assert get_debate_dir(tmp_debates_dir, "") is None


class TestStateRoundTrip:
    def test_save_then_load_preserves_data(self, tmp_debates_dir):
        debate_dir = os.path.join(tmp_debates_dir, "x")
        os.makedirs(debate_dir)
        state = {"debate_id": "x", "claim": "hello", "rounds": [{"round": 1}]}
        save_state(debate_dir, state)

        loaded = load_state(debate_dir)
        assert loaded["debate_id"] == "x"
        assert loaded["claim"] == "hello"
        assert loaded["rounds"] == [{"round": 1}]
        # save_state stamps updated_at
        assert "updated_at" in loaded

    def test_load_missing_returns_none(self, tmp_debates_dir):
        assert load_state(os.path.join(tmp_debates_dir, "nope")) is None

    def test_load_none_dir_returns_none(self):
        assert load_state(None) is None

    def test_load_corrupted_json_returns_none(self, tmp_debates_dir):
        debate_dir = os.path.join(tmp_debates_dir, "corrupt")
        os.makedirs(debate_dir)
        with open(os.path.join(debate_dir, "state.json"), "w") as f:
            f.write("{not valid json")
        assert load_state(debate_dir) is None


class TestSaveStateAtomicity:
    def test_existing_state_survives_replace_failure(self, tmp_debates_dir):
        debate_dir = os.path.join(tmp_debates_dir, "atomic")
        os.makedirs(debate_dir)
        save_state(debate_dir, {"version": 1, "data": "original"})

        # Sanity: original is there
        assert load_state(debate_dir)["version"] == 1

        # Simulate os.replace failure mid-write
        with patch("debater.state.os.replace", side_effect=OSError("disk full")):
            with pytest.raises(OSError):
                save_state(debate_dir, {"version": 2, "data": "new"})

        # Original must still be intact
        loaded = load_state(debate_dir)
        assert loaded["version"] == 1
        assert loaded["data"] == "original"


class TestCreateDebate:
    def test_creates_directory_and_state(self, tmp_debates_dir):
        debate_id = create_debate(tmp_debates_dir, "trump will lose", {})
        assert debate_id.endswith("_trump_will_lose")
        debate_dir = os.path.join(tmp_debates_dir, debate_id)
        assert os.path.isdir(debate_dir)
        assert os.path.isfile(os.path.join(debate_dir, "state.json"))

    def test_duplicate_claims_get_unique_ids(self, tmp_debates_dir):
        id1 = create_debate(tmp_debates_dir, "hello", {})
        id2 = create_debate(tmp_debates_dir, "hello", {})
        id3 = create_debate(tmp_debates_dir, "hello", {})
        assert len({id1, id2, id3}) == 3

    def test_initial_state_fields(self, tmp_debates_dir):
        debate_id = create_debate(tmp_debates_dir, "hello", {"max_iterations": 7})
        state = load_state(os.path.join(tmp_debates_dir, debate_id))
        assert state["status"] == "pending"
        assert state["current_round"] == 0
        assert state["rounds"] == []
        assert state["settings"]["max_iterations"] == 7

    def test_merges_partial_model_overrides(self, tmp_debates_dir):
        settings = {"models": {"defender": "claude-sonnet-X"}}
        debate_id = create_debate(tmp_debates_dir, "x", settings)
        state = load_state(os.path.join(tmp_debates_dir, debate_id))
        assert state["settings"]["models"]["defender"] == "claude-sonnet-X"
        # Other roles fell back to defaults
        assert "claude-" in state["settings"]["models"]["skeptic"]


class TestListDebates:
    def test_empty_dir(self, tmp_debates_dir):
        assert list_debates(tmp_debates_dir) == []

    def test_missing_dir(self, tmp_path):
        assert list_debates(str(tmp_path / "nope")) == []

    def test_skips_non_debate_files(self, tmp_debates_dir):
        # Stray file at debates root — should not crash or appear
        with open(os.path.join(tmp_debates_dir, "README.txt"), "w") as f:
            f.write("not a debate")
        assert list_debates(tmp_debates_dir) == []

    def test_skips_dirs_without_state(self, tmp_debates_dir):
        os.makedirs(os.path.join(tmp_debates_dir, "empty"))
        assert list_debates(tmp_debates_dir) == []

    def test_orders_newest_first(self, tmp_debates_dir):
        # Two debates with explicit different created_at
        for did, ts in [("a", "2026-01-01T00:00:00Z"), ("b", "2026-02-01T00:00:00Z")]:
            dd = os.path.join(tmp_debates_dir, did)
            os.makedirs(dd)
            with open(os.path.join(dd, "state.json"), "w") as f:
                json.dump({
                    "debate_id": did, "claim": did, "created_at": ts,
                    "status": "complete", "current_round": 1,
                    "settings": {"max_iterations": 1}, "rounds": [],
                }, f)
        ordered = list_debates(tmp_debates_dir)
        assert [d["id"] for d in ordered] == ["b", "a"]

    def test_includes_latest_score_from_last_eval(self, tmp_debates_dir):
        dd = os.path.join(tmp_debates_dir, "z")
        os.makedirs(dd)
        with open(os.path.join(dd, "state.json"), "w") as f:
            json.dump({
                "debate_id": "z", "claim": "z", "created_at": "2026-01-01T00:00:00Z",
                "status": "complete", "current_round": 2,
                "settings": {"max_iterations": 5},
                "rounds": [
                    {"round": 1, "evaluator": {"defender_score": 4, "skeptic_score": 6}},
                    {"round": 2, "evaluator": {"defender_score": 3, "skeptic_score": 7}},
                ],
            }, f)
        out = list_debates(tmp_debates_dir)
        assert out[0]["latest_score"] == {"defender": 3, "skeptic": 7}
