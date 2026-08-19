"""Tests for iip/catalyst_ai.py -- the AI document-read layer for Catalyst Radar's 8-K filings.
Same 2 guardrails as iip/premarket_ai.py, applied to a different surface: the AI never fetches
data itself (analyze_filing() does that once, deterministically, before ever calling the model),
and the response schema is hard-validated so the model can never invent its own confidence/
probability/reliability number -- historical grounding comes entirely from
catalyst_calibration.lookup()'s real, pre-researched data. PIIP audit 2026-08.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from iip import catalyst_ai as cai


# ---------- _clean_filing_html ----------

def test_clean_filing_html_strips_tags_and_entities():
    raw = "<html><body><p>NVIDIA&#160;announced&#160;a&#160;deal.</p></body></html>"
    cleaned = cai._clean_filing_html(raw)
    assert "<" not in cleaned and ">" not in cleaned
    assert "&#160;" not in cleaned
    assert "NVIDIA announced a deal." in cleaned


def test_clean_filing_html_collapses_whitespace():
    raw = "<p>Line one</p>\n\n\n   <p>Line   two</p>"
    cleaned = cai._clean_filing_html(raw)
    assert "   " not in cleaned
    assert "\n" not in cleaned


def test_clean_filing_html_truncates():
    raw = "<p>" + ("x" * 100) + "</p>"
    cleaned = cai._clean_filing_html(raw, max_chars=10)
    assert len(cleaned) == 10


# ---------- historical_calibration: real researched data, deterministic ----------

def test_historical_calibration_matches_real_china_restriction_category():
    text = "NVIDIA announced new export restriction rules affecting China sales."
    calib = cai.historical_calibration("NVDA", text)
    assert any("china_restriction" in k for k in calib)
    tiers = next(v for k, v in calib.items() if "china_restriction" in k)
    assert "tiers" in tiers
    assert "outright_ban_no_exceptions" in tiers["tiers"]


def test_historical_calibration_falls_back_to_earnings_reference_scale():
    """A filing that names the company but matches no SPECIFIC researched category still shows
    something (the earnings baseline, clearly labeled context-only), not nothing and not a
    fabricated number."""
    text = "NVIDIA entered into a data center financing partnership."
    calib = cai.historical_calibration("NVDA", text)
    assert "NVDA_reference_scale" in calib
    assert "context only" in calib["NVDA_reference_scale"]["note"] or \
           "No specific researched" in calib["NVDA_reference_scale"]["note"]


def test_historical_calibration_empty_when_nothing_matches():
    text = "A completely generic filing about routine corporate housekeeping."
    calib = cai.historical_calibration("ZZZZ", text)
    assert calib == {}


def test_historical_calibration_matches_macro_category():
    text = "The Federal Reserve announced a rate cut affecting markets broadly."
    calib = cai.historical_calibration("SPY", text)
    assert any(v.get("category") == "fomc_cut" for v in calib.values())


# ---------- validate_filing_analysis: happy path ----------

def _valid_analysis():
    return {"key_dates": [{"date": "2028-01-01", "description": "Leases expected ready-for-service"}],
           "trend": "Neutral", "summary": "The filing discloses a financing partnership."}


def test_valid_analysis_passes():
    assert cai.validate_filing_analysis(_valid_analysis()) is not None


def test_valid_analysis_empty_key_dates_ok():
    ok = _valid_analysis()
    ok["key_dates"] = []
    assert cai.validate_filing_analysis(ok) is not None


# ---------- validate_filing_analysis: structural violations ----------

def test_rejects_non_dict():
    assert cai.validate_filing_analysis(["not", "a", "dict"]) is None


def test_rejects_missing_field():
    bad = _valid_analysis()
    del bad["trend"]
    assert cai.validate_filing_analysis(bad) is None


def test_rejects_extra_field():
    bad = _valid_analysis()
    bad["extra"] = "not allowed"
    assert cai.validate_filing_analysis(bad) is None


def test_rejects_invalid_trend_value():
    bad = _valid_analysis()
    bad["trend"] = "Very Bullish"
    assert cai.validate_filing_analysis(bad) is None


def test_rejects_empty_summary():
    bad = _valid_analysis()
    bad["summary"] = "   "
    assert cai.validate_filing_analysis(bad) is None


def test_rejects_key_dates_not_a_list():
    bad = _valid_analysis()
    bad["key_dates"] = "2028-01-01"
    assert cai.validate_filing_analysis(bad) is None


def test_rejects_key_date_missing_description():
    bad = _valid_analysis()
    bad["key_dates"] = [{"date": "2028-01-01"}]
    assert cai.validate_filing_analysis(bad) is None


def test_rejects_key_date_extra_field():
    bad = _valid_analysis()
    bad["key_dates"] = [{"date": "2028-01-01", "description": "x", "importance": "high"}]
    assert cai.validate_filing_analysis(bad) is None


# ---------- validate_filing_analysis: forbidden fields + numeric leaves (guardrail #2) ----------

@pytest.mark.parametrize("forbidden_key", ["confidence", "probability", "likelihood", "score", "reliability"])
def test_rejects_forbidden_key_at_top_level(forbidden_key):
    bad = _valid_analysis()
    bad[forbidden_key] = "should never be here"
    assert cai.validate_filing_analysis(bad) is None


def test_rejects_forbidden_key_nested_in_key_dates():
    bad = _valid_analysis()
    bad["key_dates"] = [{"date": "2028-01-01", "description": "x"}]
    bad["key_dates"][0]["confidence"] = "high"   # makes the dict have 3 keys -> also caught by
    # the exact-keys check, but this test is specifically about the forbidden-key sweep finding it
    # even if that check somehow didn't fire first.
    assert cai.validate_filing_analysis(bad) is None


def test_rejects_bare_numeric_leaf_in_summary():
    """summary must be a string -- a bare number in its place is rejected (covered by the
    type check), and _no_numeric_leaves() independently confirms it as a numeric leaf too."""
    poisoned = _valid_analysis()
    poisoned["summary"] = 42
    assert cai.validate_filing_analysis(poisoned) is None
    assert cai._no_numeric_leaves(poisoned) is False


def test_no_numeric_leaves_sweep_finds_deeply_nested_number():
    obj = {"key_dates": [{"date": "x", "description": "y", "extra": {"nested": {"n": 7}}}]}
    assert cai._no_numeric_leaves(obj) is False


def test_allows_numbers_referenced_in_prose():
    ok = _valid_analysis()
    ok["summary"] = "NVIDIA's obligation is capped at $105 billion under the agreement."
    assert cai.validate_filing_analysis(ok) is not None


# ---------- analyze_filing: never fetches beyond the one document, hard-validates ----------

class _FakeClient:
    def __init__(self, response, cost=0.003):
        self._response = response
        self._cost = cost

    def call(self, prompt: str):
        assert isinstance(prompt, str) and len(prompt) > 0
        return self._response, self._cost


def test_analyze_filing_unfetchable_url_returns_invalid(monkeypatch):
    monkeypatch.setattr(cai, "fetch_filing_text", lambda url, max_chars=cai.MAX_DOC_CHARS: None)
    result = cai.analyze_filing("NVDA", "https://example.com/bad", _FakeClient(_valid_analysis()))
    assert result["_valid"] is False
    assert "fetch" in result["_reason"].lower()


def test_analyze_filing_valid_response(monkeypatch):
    monkeypatch.setattr(cai, "fetch_filing_text",
                        lambda url, max_chars=cai.MAX_DOC_CHARS: "NVIDIA export restriction filing text.")
    result = cai.analyze_filing("NVDA", "https://example.com/8k", _FakeClient(_valid_analysis()))
    assert result["_valid"] is True
    assert result["trend"] == "Neutral"
    # the calibration attached must be the SAME deterministic result the standalone function
    # would produce -- proves it wasn't left to the AI to decide.
    expected_calib = cai.historical_calibration("NVDA", "NVIDIA export restriction filing text.")
    assert result["_calibration"] == expected_calib


def test_analyze_filing_invalid_ai_response_no_silent_repair(monkeypatch):
    monkeypatch.setattr(cai, "fetch_filing_text",
                        lambda url, max_chars=cai.MAX_DOC_CHARS: "Some filing text.")
    malformed = {"trend": "Neutral"}   # missing key_dates/summary
    result = cai.analyze_filing("XYZ", "https://example.com/8k", _FakeClient(malformed))
    assert result["_valid"] is False
    assert result["_reason"] == "AI response failed schema validation"
    assert "_raw" in result


def test_analyze_filing_dry_run(monkeypatch):
    monkeypatch.setattr(cai, "fetch_filing_text",
                        lambda url, max_chars=cai.MAX_DOC_CHARS: "Some filing text.")
    result = cai.analyze_filing("XYZ", "https://example.com/8k", _FakeClient(None))
    assert result["_valid"] is False
    assert result["_dry_run"] is True


def test_analyze_filing_budget_skipped(monkeypatch):
    monkeypatch.setattr(cai, "fetch_filing_text",
                        lambda url, max_chars=cai.MAX_DOC_CHARS: "Some filing text.")
    result = cai.analyze_filing("XYZ", "https://example.com/8k",
                                _FakeClient({"_skipped": "daily cap reached"}, cost=0.0))
    assert result["_valid"] is False
    assert result["_reason"] == "daily cap reached"
