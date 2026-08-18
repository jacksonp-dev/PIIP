"""Tests for iip/premarket_ai.py -- the 3 explicit guardrails: frozen-snapshot reproducibility
(the AI call itself never fetches), hard schema validation (no silent repair, no AI-invented
numbers/trade-state), and the persistence layer that stores the AI's answer immutably, linked to
its exact original thesis. PIIP audit 2026-08, Premarket Thesis AI layer.
"""
from __future__ import annotations

import gc
import os
import sys
import tempfile
import time
import uuid
from datetime import date, datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from iip import premarket_ai as pai
from iip import premarket_thesis as pt


@pytest.fixture
def tmp_db():
    path = os.path.join(tempfile.gettempdir(), f"test_piip_premarket_ai_{uuid.uuid4().hex}.db")
    yield path
    gc.collect()
    for _ in range(5):
        try:
            if os.path.exists(path):
                os.remove(path)
            break
        except PermissionError:
            time.sleep(0.1)


def _valid_response() -> dict:
    return {
        "trend_context": {"analysis": "Premarket move aligns with the daily trend."},
        "signal_conviction": {"analysis": "6 of 8 families agree bearish."},
        "risk_environment": {"analysis": "Elevated risk increases reversal odds."},
        "catalyst_risk": {"analysis": "No catalyst before the open."},
        "overnight_news": {"analysis": "Headlines are risk-off, consistent with the setup."},
        "historical_evidence": {"analysis": "Both tiers lean bearish, moderate sample."},
        "synthesis": {"analysis": "Evidence favors the bearish thesis.", "verdict": "SUPPORTS",
                     "failure_mode": "A sharp opening reversal on short covering."},
    }


# ---------- validate_ai_response: happy path ----------

def test_valid_response_passes():
    assert pai.validate_ai_response(_valid_response()) is not None


# ---------- validate_ai_response: structural violations ----------

def test_rejects_non_dict():
    assert pai.validate_ai_response(["not", "a", "dict"]) is None
    assert pai.validate_ai_response("a string") is None


def test_rejects_missing_section():
    bad = _valid_response()
    del bad["catalyst_risk"]
    assert pai.validate_ai_response(bad) is None


def test_rejects_extra_section():
    bad = _valid_response()
    bad["extra_section"] = {"analysis": "not allowed"}
    assert pai.validate_ai_response(bad) is None


def test_rejects_empty_analysis():
    bad = _valid_response()
    bad["trend_context"]["analysis"] = "   "
    assert pai.validate_ai_response(bad) is None


def test_rejects_non_string_analysis():
    bad = _valid_response()
    bad["trend_context"]["analysis"] = 42
    assert pai.validate_ai_response(bad) is None


# ---------- validate_ai_response: the specific forbidden fields (guardrail #2) ----------

@pytest.mark.parametrize("forbidden_key", ["confidence", "ai_confidence", "trade_state",
                                           "trade_permission", "position_size", "stop_loss",
                                           "recommendation"])
def test_rejects_forbidden_key_at_top_of_section(forbidden_key):
    bad = _valid_response()
    bad["signal_conviction"][forbidden_key] = "should never be here"
    assert pai.validate_ai_response(bad) is None


def test_rejects_forbidden_key_nested_deep():
    """Forbidden keys must be caught no matter how deeply nested -- an AI response that buries a
    confidence score inside a sub-object is still rejected."""
    bad = _valid_response()
    bad["synthesis"]["extra"] = {"nested": {"confidence": 0.9}}
    assert pai.validate_ai_response(bad) is None


def test_rejects_bare_numeric_leaf_anywhere():
    """No numeric statistic the AI didn't get from PIIP -- a bare int/float ANYWHERE in the
    response is rejected, even a plausible-looking one like a sample size or a percentage."""
    bad = _valid_response()
    bad["historical_evidence"]["sample_size"] = 47
    assert pai.validate_ai_response(bad) is None


def test_allows_numbers_referenced_in_prose():
    """A number mentioned IN WORDS inside the analysis text is fine -- only a bare JSON numeric
    leaf (the model inventing its OWN new number) is rejected."""
    ok = _valid_response()
    ok["historical_evidence"]["analysis"] = "The sample of 47 days leans bearish at a 61% rate."
    assert pai.validate_ai_response(ok) is not None


def test_rejects_bool_is_not_treated_as_forbidden_numeric():
    """bool is technically an int subclass in Python -- confirm the numeric-leaf check doesn't
    accidentally reject a legitimate boolean value if one were ever present."""
    ok = _valid_response()
    ok["synthesis"]["some_flag"] = True
    # still valid shape-wise (extra unexpected key at top level of synthesis is fine, only the
    # REQUIRED_SECTIONS set at the outer level is strict) -- this just confirms bool doesn't trip
    # the numeric-leaf rejection the way a bare int/float would.
    assert pai._no_numeric_leaves(ok) is True


# ---------- validate_ai_response: synthesis-specific requirements ----------

@pytest.mark.parametrize("bad_verdict", ["LIKELY", "supports", "Yes", None, ""])
def test_rejects_invalid_verdict(bad_verdict):
    bad = _valid_response()
    bad["synthesis"]["verdict"] = bad_verdict
    assert pai.validate_ai_response(bad) is None


def test_rejects_missing_failure_mode():
    bad = _valid_response()
    del bad["synthesis"]["failure_mode"]
    assert pai.validate_ai_response(bad) is None


def test_rejects_empty_failure_mode():
    bad = _valid_response()
    bad["synthesis"]["failure_mode"] = ""
    assert pai.validate_ai_response(bad) is None


# ---------- ask_ai_thesis: never fetches, hard-validates, never silently repairs ----------

class _FakeClient:
    def __init__(self, response, cost=0.0021):
        self._response = response
        self._cost = cost

    def call(self, prompt: str):
        assert isinstance(prompt, str) and len(prompt) > 0
        return self._response, self._cost


def test_ask_ai_thesis_valid_response():
    snapshot = {"snapshot_timestamp": "2026-08-18T08:45:00"}
    result = pai.ask_ai_thesis(snapshot, _FakeClient(_valid_response()))
    assert result["_valid"] is True
    assert result["synthesis"]["verdict"] == "SUPPORTS"
    assert result["ai_layer_version"] == pai.AI_LAYER_VERSION


def test_ask_ai_thesis_invalid_response_no_silent_repair():
    snapshot = {"snapshot_timestamp": "2026-08-18T08:45:00"}
    malformed = {"trend_context": {"analysis": "ok"}}   # missing 6 sections
    result = pai.ask_ai_thesis(snapshot, _FakeClient(malformed))
    assert result["_valid"] is False
    assert result["_reason"] == "AI response failed schema validation"
    assert "_raw" in result   # preserved for debugging, but never rendered as if it were valid


def test_ask_ai_thesis_dry_run():
    snapshot = {"snapshot_timestamp": "2026-08-18T08:45:00"}
    result = pai.ask_ai_thesis(snapshot, _FakeClient(None))   # LLMClient.call() returns (None, 0.0) on dry-run
    assert result["_valid"] is False
    assert result["_dry_run"] is True


def test_ask_ai_thesis_budget_skipped():
    snapshot = {"snapshot_timestamp": "2026-08-18T08:45:00"}
    result = pai.ask_ai_thesis(snapshot, _FakeClient({"_skipped": "daily cap reached"}, cost=0.0))
    assert result["_valid"] is False
    assert result["_reason"] == "daily cap reached"


def test_ask_ai_thesis_never_fetches_additional_data(monkeypatch):
    """Guardrail #1's central claim, made concrete: patch every data-fetching function this
    project has access to so it raises if called, then confirm ask_ai_thesis() still completes
    successfully using ONLY the snapshot already passed in -- it never reaches out for more."""
    def _boom(*a, **k):
        raise AssertionError("ask_ai_thesis() must never fetch data -- it only reads the frozen snapshot")

    monkeypatch.setattr(pt, "overnight_headlines", _boom)
    import iip.zero_dte as zd
    import iip.macro as macro
    import iip.premarket_backtest as bt
    monkeypatch.setattr(zd, "fetch_intraday_batch", _boom)
    monkeypatch.setattr(zd, "fetch_daily_batch", _boom)
    monkeypatch.setattr(macro, "fetch_macro_batch", _boom)
    monkeypatch.setattr(bt, "fetch_tier1_history", _boom)

    snapshot = {"snapshot_timestamp": "2026-08-18T08:45:00", "spot_price": 640.0}
    result = pai.ask_ai_thesis(snapshot, _FakeClient(_valid_response()))
    assert result["_valid"] is True   # completed without tripping any of the patched fetchers


# ---------- build_snapshot: trend relationship (deterministic, PIIP-computed) ----------

def test_trend_relationship_continuation_bullish():
    assert "Continuation" in pai._trend_relationship("Bullish", "Bullish")


def test_trend_relationship_opposing():
    label = pai._trend_relationship("Bullish", "Bearish")
    assert "Opposing" in label


def test_trend_relationship_neutral_premarket():
    label = pai._trend_relationship("Bullish", "Neutral")
    assert "too small to characterize" in label


def test_trend_relationship_unknown_daily():
    label = pai._trend_relationship(None, "Bearish")
    assert "unknown" in label.lower()


# ---------- persistence: AI output write-once, linked to the exact thesis row ----------

def _thesis(direction="Bearish", confidence=72.0, spot=640.0, risk="ELEVATED"):
    return {"spot_price": spot, "market_state": {"direction": direction, "confidence": confidence},
           "risk_environment": {"level": risk}, "families": {}, "confirmation": {},
           "invalidation": {}, "trade_permission": {}}


def test_get_todays_thesis_ai_fields_none_before_ai_call(tmp_db):
    pt.log_thesis_once("SPY", _thesis(), path=tmp_db)
    readback = pt.get_todays_thesis("SPY", path=tmp_db)
    assert readback["_thesis_id"] is not None
    assert readback["_ai_snapshot"] is None
    assert readback["_ai_output"] is None
    assert readback["_historical_methodology_version"] is None


def test_log_thesis_ai_output_write_once(tmp_db):
    pt.log_thesis_once("SPY", _thesis(), path=tmp_db)
    snap1, out1 = {"v": 1}, {"_valid": True, "synthesis": {"verdict": "MIXED"}}
    wrote1 = pt.log_thesis_ai_output("SPY", snap1, out1, "v1", path=tmp_db)
    assert wrote1 is True

    # A second AI call the SAME morning must NOT overwrite the first real answer.
    snap2, out2 = {"v": 2}, {"_valid": True, "synthesis": {"verdict": "SUPPORTS"}}
    wrote2 = pt.log_thesis_ai_output("SPY", snap2, out2, "v1", path=tmp_db)
    assert wrote2 is False

    readback = pt.get_todays_thesis("SPY", path=tmp_db)
    assert readback["_ai_output"]["synthesis"]["verdict"] == "MIXED"
    assert readback["_ai_snapshot"]["v"] == 1
    assert readback["_historical_methodology_version"] == "v1"


def test_log_thesis_ai_output_noop_when_no_thesis_row_exists(tmp_db):
    """If the deterministic thesis was never logged (e.g. a transient failure), attaching an AI
    output has nothing to attach to -- must be a clean no-op, never create an orphan row."""
    wrote = pt.log_thesis_ai_output("SPY", {}, {"_valid": True}, "v1", path=tmp_db)
    assert wrote is False


def test_checkpoint_linked_to_exact_thesis_id(tmp_db):
    pt.log_thesis_once("SPY", _thesis(), path=tmp_db)
    thesis_id = pt.get_todays_thesis("SPY", path=tmp_db)["_thesis_id"]
    checkpoint = {"directional_accuracy": "CORRECT", "tradeability": "GOOD"}
    pt.log_checkpoint("SPY", "10:00 AM", checkpoint, thesis_id=thesis_id, path=tmp_db)
    stored = pt.get_checkpoints("SPY", path=tmp_db)
    assert len(stored) == 1
    assert stored[0]["thesis_id"] == thesis_id


# ---------- migration: additive, safe against an already-live pre-AI-layer database ----------

def test_migration_preserves_existing_rows_and_adds_columns(tmp_db):
    """Simulates a database created BEFORE this session's AI-layer columns existed (exactly the
    live iip.db's own pre-migration shape) -- must migrate cleanly via ALTER TABLE, never lose
    the existing row, and old rows must read back with the new fields as None."""
    import sqlite3
    con = sqlite3.connect(tmp_db)
    con.executescript("""
        CREATE TABLE premarket_thesis_log (
          id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, date TEXT NOT NULL,
          ticker TEXT NOT NULL, spot_price REAL, direction TEXT, confidence REAL,
          risk_environment TEXT, thesis_json TEXT NOT NULL, UNIQUE(date, ticker)
        );
        CREATE TABLE premarket_checkpoint_log (
          id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, date TEXT NOT NULL,
          ticker TEXT NOT NULL, checkpoint_label TEXT NOT NULL, detail_json TEXT NOT NULL,
          UNIQUE(date, ticker, checkpoint_label)
        );
    """)
    import json as _json
    old_thesis = _thesis(direction="Bullish")
    con.execute("INSERT INTO premarket_thesis_log (ts, date, ticker, spot_price, direction, "
               "confidence, risk_environment, thesis_json) VALUES (?,?,?,?,?,?,?,?)",
               ("2020-01-02T08:00:00", "2020-01-02", "SPY", 100.0, "Bullish", 65.0, "NORMAL",
                _json.dumps(old_thesis)))
    con.commit()
    con.close()

    # Round-trip through the real module -- must not raise, must migrate columns in place.
    readback = pt.get_todays_thesis("SPY", path=tmp_db)   # today != 2020-01-02, so None expected
    assert readback is None   # confirms no crash reading a pre-migration DB at all

    con2 = pt._connect(tmp_db)
    cols = {r[1] for r in con2.execute("PRAGMA table_info(premarket_thesis_log)").fetchall()}
    assert {"snapshot_json", "ai_output_json", "historical_methodology_version"} <= cols
    old_row = con2.execute("SELECT * FROM premarket_thesis_log WHERE date='2020-01-02'").fetchone()
    assert old_row is not None   # the pre-existing row survived the migration
    ckpt_cols = {r[1] for r in con2.execute("PRAGMA table_info(premarket_checkpoint_log)").fetchall()}
    assert "thesis_id" in ckpt_cols
    con2.close()
