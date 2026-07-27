"""Decision Journal — log DECISIONS (not just trades) with a pre-committed thesis, the falsifiers that
would prove you WRONG, and an exit plan; then review the OUTCOME honestly.

Falsification-first by construction: you write what would prove you wrong BEFORE the outcome, and the
review forces the process-vs-luck distinction ('was my reasoning wrong, or was I just unlucky?').
Each directional decision also logs a prediction to the Scorecard, so your calls get graded vs reality
over time. All local SQLite (shares iip.db), $0, no LLM. See [[investintel-agent]].
"""
from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone

from . import predictions as pred

DB_PATH = "iip.db"

# The 2×2 that separates decision QUALITY from luck — the heart of the review.
VERDICTS = [
    "✅ Reasoning right · it worked (good process, good outcome)",
    "🍀 Reasoning wrong · got lucky (bad process, good outcome)",
    "🌧️ Reasoning right · unlucky (good process, bad outcome)",
    "❌ Reasoning wrong · it failed (bad process, bad outcome)",
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS journal (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created TEXT, ticker TEXT, company TEXT,
  decision TEXT, direction TEXT, confidence INTEGER,
  thesis TEXT, falsifiers TEXT, assumptions TEXT, catalyst TEXT,
  target REAL, stop REAL, max_loss REAL, position_size TEXT, exit_plan TEXT,
  horizon_days INTEGER, review_date TEXT, spot_at_entry REAL,
  status TEXT DEFAULT 'OPEN',
  outcome TEXT, verdict TEXT, repeat_again TEXT, lessons TEXT, reviewed_ts TEXT,
  pred_id INTEGER
);
"""


def _con(db):
    c = sqlite3.connect(db)
    c.row_factory = sqlite3.Row
    return c


def init_db(db: str = DB_PATH) -> None:
    con = _con(db)
    con.executescript(SCHEMA)
    con.commit()
    con.close()


def add(db: str, rec: dict) -> int:
    """Insert a decision. If it has a direction + horizon, also log a Scorecard prediction so the call
    gets graded vs reality. Returns the new journal id."""
    init_db(db)
    pred_id = None
    if rec.get("direction") in ("bullish", "bearish") and rec.get("horizon_days"):
        conf = (rec.get("confidence") or 50) / 100.0
        prob_up = round(conf if rec["direction"] == "bullish" else 1 - conf, 3)
        try:
            pred.init_db(db)                      # ensure the Scorecard table exists before linking
            pred_id = pred.log_prediction({
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "ticker": rec["ticker"], "horizon_days": int(rec["horizon_days"]), "source": "journal",
                "spot": rec.get("spot_at_entry"), "stock_direction": rec["direction"],
                "stock_prob_up": prob_up, "confidence": conf,
                "reasoning": {"mode": "decision_journal", "thesis": rec.get("thesis", "")[:200]},
            }, db)
        except Exception:
            pred_id = None
    con = _con(db)
    cur = con.execute(
        """INSERT INTO journal
           (created,ticker,company,decision,direction,confidence,thesis,falsifiers,assumptions,catalyst,
            target,stop,max_loss,position_size,exit_plan,horizon_days,review_date,spot_at_entry,status,pred_id)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?, 'OPEN', ?)""",
        (date.today().isoformat(), rec["ticker"], rec.get("company"), rec["decision"], rec.get("direction"),
         rec.get("confidence"), rec.get("thesis"), rec.get("falsifiers"), rec.get("assumptions"),
         rec.get("catalyst"), rec.get("target"), rec.get("stop"), rec.get("max_loss"),
         rec.get("position_size"), rec.get("exit_plan"), rec.get("horizon_days"), rec.get("review_date"),
         rec.get("spot_at_entry"), pred_id))
    con.commit()
    rid = cur.lastrowid
    con.close()
    return rid


def all_entries(db: str = DB_PATH, status: str | None = None) -> list[dict]:
    init_db(db)
    con = _con(db)
    if status:
        rows = con.execute("SELECT * FROM journal WHERE status=? ORDER BY id DESC", (status,)).fetchall()
    else:
        rows = con.execute("SELECT * FROM journal ORDER BY id DESC").fetchall()
    con.close()
    return [dict(r) for r in rows]


def entries_for_ticker(db: str, ticker: str) -> list[dict]:
    """All journal entries (any status) for one ticker, newest first — powers the Ticker Page."""
    return [e for e in all_entries(db) if e["ticker"] == ticker.upper()]


def review(db: str, jid: int, outcome: str, verdict: str, repeat_again: str, lessons: str) -> None:
    init_db(db)
    con = _con(db)
    con.execute(
        """UPDATE journal SET status='REVIEWED', outcome=?, verdict=?, repeat_again=?, lessons=?,
           reviewed_ts=? WHERE id=?""",
        (outcome, verdict, repeat_again, lessons,
         datetime.now(timezone.utc).isoformat(timespec="seconds"), jid))
    con.commit()
    con.close()


def delete(db: str, jid: int) -> None:
    init_db(db)
    con = _con(db)
    con.execute("DELETE FROM journal WHERE id=?", (jid,))
    con.commit()
    con.close()


def due_for_review(db: str = DB_PATH) -> list[dict]:
    """Open decisions whose review_date has arrived — nudges you to close the loop honestly."""
    today = date.today().isoformat()
    return [e for e in all_entries(db, "OPEN") if e.get("review_date") and e["review_date"] <= today]


def process_score(db: str = DB_PATH) -> dict:
    """A tiny 'Investment IQ' seed: what fraction of your decisions followed good process — wrote a
    thesis, pre-committed falsifiers, set an exit plan, AND got reviewed after the outcome."""
    entries = all_entries(db)
    if not entries:
        return {"n": 0}
    def _has(e, *keys):
        return all((e.get(k) or "").strip() for k in keys)
    n = len(entries)
    thesis = sum(1 for e in entries if _has(e, "thesis"))
    falsi = sum(1 for e in entries if _has(e, "falsifiers"))
    planned = sum(1 for e in entries if (e.get("stop") is not None or (e.get("exit_plan") or "").strip()
                                         or e.get("max_loss") is not None))
    reviewed = sum(1 for e in entries if e.get("status") == "REVIEWED")
    return {"n": n, "thesis_pct": round(100 * thesis / n), "falsifiers_pct": round(100 * falsi / n),
            "exitplan_pct": round(100 * planned / n), "reviewed_pct": round(100 * reviewed / n)}
