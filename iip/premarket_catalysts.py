"""Static macro-catalyst calendar for the Premarket Thesis AI's catalyst-risk question.

FOMC_DATES_2026 confirmed LIVE (2026-08-18) against federalreserve.gov/monetarypolicy/
fomccalendars.htm, cross-checked against a second independent source. NOT auto-updating: FOMC
dates are published about a year ahead and change rarely enough that a hardcoded, dated, sourced
table is honest and low-maintenance here -- re-verify against the Fed's own calendar each year
rather than try to scrape/parse it live for eight dates.

CPI/Nonfarm Payrolls do NOT have a similarly fixed, publicly-scheduled-far-ahead source buildable
for free in this project. The one real candidate -- Finnhub's economic calendar endpoint, via the
FINNHUB_API_KEY already configured for catalyst_terminal.py's news feed -- was tested LIVE
(2026-08-18) and returns `403 You don't have access to this resource`: it's a paid-tier-only
endpoint on this account. So CPI/Payrolls timing here is an explicit APPROXIMATION from the
typical monthly release cadence (gap since the last real print, via
macro.economic_releases_snapshot()'s own as_of date) -- never presented as an exact scheduled
date, always labeled as approximate.
"""
from __future__ import annotations

from datetime import date

# Confirmed live 2026-08-18 against federalreserve.gov/monetarypolicy/fomccalendars.htm.
# Statement released 2:00pm ET on the SECOND day of each meeting; every 2026 meeting has a press
# conference at 2:30pm ET (standard practice since 2019, not limited to Summary-of-Economic-
# Projections meetings).
FOMC_DATES_2026 = [
    (date(2026, 1, 27), date(2026, 1, 28)),
    (date(2026, 3, 17), date(2026, 3, 18)),
    (date(2026, 4, 28), date(2026, 4, 29)),
    (date(2026, 6, 16), date(2026, 6, 17)),
    (date(2026, 7, 28), date(2026, 7, 29)),
    (date(2026, 9, 15), date(2026, 9, 16)),
    (date(2026, 10, 27), date(2026, 10, 28)),
    (date(2026, 12, 8), date(2026, 12, 9)),
]

STANDARD_RELEASE_TIMES_ET = {"FOMC Statement": "14:00", "CPI": "08:30", "Nonfarm Payrolls": "08:30"}

# Typical monthly cadence in days -- used ONLY to say a release is "roughly due" from the gap
# since the last real print, never claimed as an exact date. First-pass/unvalidated, same honesty
# standard as every other threshold in this codebase. Limited to labels macro.py's
# ECON_RELEASE_SERIES actually fetches (CPI, Nonfarm Payrolls) -- no phantom entries for series
# this project doesn't have real data for (e.g. PCE).
TYPICAL_CADENCE_DAYS = {"CPI": 30, "Nonfarm Payrolls": 30}


def fomc_today(today: date | None = None) -> dict:
    """Whether today falls within a scheduled FOMC meeting window, and whether the 2:00pm ET
    statement lands specifically today (the second day of the meeting)."""
    today = today or date.today()
    for start, end in FOMC_DATES_2026:
        if start <= today <= end:
            statement_today = today == end
            return {"in_meeting": True, "start": start.isoformat(), "end": end.isoformat(),
                   "statement_today": statement_today,
                   "statement_time_et": STANDARD_RELEASE_TIMES_ET["FOMC Statement"]
                   if statement_today else None}
    return {"in_meeting": False}


def _days_since_last(label: str, releases: dict) -> int | None:
    entry = releases.get(label)
    if not entry or not entry.get("as_of"):
        return None
    last = date.fromisoformat(entry["as_of"])
    return (date.today() - last).days


def catalyst_risk_today(releases: dict, today: date | None = None) -> dict:
    """Assembles the catalyst-risk read for the AI snapshot -- FOMC is exact (real published Fed
    calendar); CPI/Payrolls are approximate ('roughly due' from typical cadence), and that
    distinction is carried in the output, never flattened into one undifferentiated 'catalyst
    today: yes/no'."""
    today = today or date.today()
    approx_due = {}
    for label, cadence in TYPICAL_CADENCE_DAYS.items():
        gap = _days_since_last(label, releases)
        if gap is not None:
            approx_due[label] = {"days_since_last_release": gap,
                                 "roughly_due": gap >= cadence - 3,
                                 "typical_release_time_et": STANDARD_RELEASE_TIMES_ET.get(label)}
    return {"fomc": fomc_today(today), "approx_due": approx_due,
           "note": ("FOMC dates are exact (published Federal Reserve calendar, confirmed "
                    "2026-08-18). CPI/Payrolls are approximate -- 'roughly due' from typical "
                    "monthly cadence, not a real scheduled calendar (no free source available: "
                    "Finnhub's economic calendar endpoint returns 403 on this account's tier, "
                    "confirmed live).")}
