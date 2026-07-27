"""Paper-trading account for options — start with $1,000, buy calls/puts, track live P&L.

Accounting (audited):
  buy cost      = contracts × premium × 100
  open value    = contracts × current_premium × 100   (marked live off the chain; intrinsic if expired)
  close proceeds= contracts × exit_premium × 100
  realized pnl  = (exit − entry) × contracts × 100
  equity        = cash + Σ open value ;  total P&L = equity − start ;  return = that / start
Fills at MID price with ZERO commission — real trading pays the bid-ask spread + fees, so live
results are worse. This is a paper sandbox to learn, not a P&L promise.
"""
from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone

from . import data
from . import deterministic as det

DB_PATH = "iip.db"
CONTRACT = 100  # one option contract controls 100 shares

SCHEMA = """
CREATE TABLE IF NOT EXISTS account (
  id INTEGER PRIMARY KEY CHECK (id=1), cash REAL, start_cash REAL, created TEXT);
CREATE TABLE IF NOT EXISTS positions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT, opt_type TEXT, strike REAL, expiry TEXT,
  contracts INTEGER, entry_premium REAL, entry_spot REAL, entry_ts TEXT,
  status TEXT DEFAULT 'OPEN', exit_premium REAL, exit_ts TEXT, realized_pnl REAL);
CREATE TABLE IF NOT EXISTS equity_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT UNIQUE, ts TEXT, equity REAL, cash REAL);
"""


def _con(db):
    c = sqlite3.connect(db)
    c.row_factory = sqlite3.Row
    return c


def get_or_create(db: str = DB_PATH, start: float = 1000.0) -> dict:
    con = _con(db)
    con.executescript(SCHEMA)
    row = con.execute("SELECT * FROM account WHERE id=1").fetchone()
    if row is None:
        con.execute("INSERT INTO account(id,cash,start_cash,created) VALUES(1,?,?,?)",
                    (start, start, datetime.now(timezone.utc).isoformat(timespec="seconds")))
        con.commit()
        row = con.execute("SELECT * FROM account WHERE id=1").fetchone()
    d = dict(row)
    con.close()
    return d


def reset(db: str = DB_PATH, start: float = 1000.0) -> None:
    """Wipes positions, account, AND equity_history. Leaving equity_history behind was a real bug
    (found in this session's accounting audit): the old daily snapshots would survive a reset and
    equity_curve() would keep blending them in, showing a fake historical climb/drop that belonged
    to the pre-reset account -- e.g. day_pnl() comparing a fresh $1,000 against a stale pre-reset
    balance from the same calendar date. A reset must zero out every trace of the old account."""
    con = _con(db)
    con.executescript(SCHEMA)
    con.execute("DELETE FROM positions")
    con.execute("DELETE FROM account")
    con.execute("DELETE FROM equity_history")
    con.execute("INSERT INTO account(id,cash,start_cash,created) VALUES(1,?,?,?)",
                (start, start, datetime.now(timezone.utc).isoformat(timespec="seconds")))
    con.commit()
    con.close()


def get_cash(db: str = DB_PATH) -> float:
    get_or_create(db)
    con = _con(db)
    v = con.execute("SELECT cash FROM account WHERE id=1").fetchone()[0]
    con.close()
    return v


def buy(db, ticker, opt_type, strike, expiry, contracts, premium, spot) -> float:
    get_or_create(db)
    cost = contracts * premium * CONTRACT
    con = _con(db)
    cash = con.execute("SELECT cash FROM account WHERE id=1").fetchone()[0]
    if cost > cash + 1e-9:
        con.close()
        raise ValueError(f"Cost ${cost:,.0f} exceeds available cash ${cash:,.0f}")
    con.execute("UPDATE account SET cash=cash-? WHERE id=1", (cost,))
    con.execute("""INSERT INTO positions(ticker,opt_type,strike,expiry,contracts,entry_premium,
                   entry_spot,entry_ts,status) VALUES(?,?,?,?,?,?,?,?, 'OPEN')""",
                (ticker, opt_type, float(strike), expiry, int(contracts), float(premium),
                 float(spot), datetime.now(timezone.utc).isoformat(timespec="seconds")))
    con.commit()
    con.close()
    return cost


def _historical_close_on(ticker: str, target_date) -> float | None:
    """Actual closing price on a specific past date (or the most recent trading day at/before it,
    for weekends/holidays) -- None on any failure, never fabricated."""
    try:
        df = data.get_prices(ticker, period="1y")
    except Exception:
        return None
    idx = df.index[df.index.date <= target_date]
    if len(idx) == 0:
        return None
    return float(df.loc[idx[-1], "Close"])


def current_premium(ticker, opt_type, strike, expiry) -> tuple[float, float]:
    """Live mid premium of a held contract; intrinsic value if expired/delisted. Returns (premium, spot)."""
    exp = datetime.strptime(expiry, "%Y-%m-%d").date()
    today = date.today()
    if today > exp:
        # Already expired (not just "expires today," which still uses live spot below since the
        # market's still open and nothing's settled yet) -- price it off the ACTUAL closing spot
        # ON the expiration date, not today's live spot. Found in this session's accounting audit:
        # this value is what close_expired() actually uses to credit/debit real cash, so if a user
        # doesn't open the app for days after expiration, using live spot would silently settle at
        # the WRONG price -- drifting further from the true expiration-day value the longer they
        # wait, not just a cosmetic "Now" display discrepancy.
        spot = _historical_close_on(ticker, exp)
        if spot is None:
            spot = data.get_spot(ticker)   # best-effort fallback if the history lookup fails
        intrinsic = max(0.0, spot - strike) if opt_type == "call" else max(0.0, strike - spot)
        return intrinsic, spot
    spot = data.get_spot(ticker)
    intrinsic = max(0.0, spot - strike) if opt_type == "call" else max(0.0, strike - spot)
    if today == exp or expiry not in data.list_expiries(ticker):
        return intrinsic, spot
    chain = data.get_option_chain(ticker, expiry)
    side = chain["calls"] if opt_type == "call" else chain["puts"]
    r = side[side["strike"] == strike]
    if len(r) == 0:
        return intrinsic, spot
    mid = det._mid(r.iloc[0])
    return (mid if mid == mid else intrinsic), spot


def open_positions(db: str = DB_PATH):
    # executescript(SCHEMA) -- a REAL bug found while testing a genuinely fresh clone (no iip.db
    # at all yet): this is called unconditionally on every single page load via
    # render_trade_drawer(), before anything else has necessarily created the tables, so a brand
    # new database crashed the whole app on its very first render with "no such table: positions."
    # Never surfaced on an existing developer install since real usage had already created the
    # schema long ago.
    con = _con(db)
    con.executescript(SCHEMA)
    rows = con.execute("SELECT * FROM positions WHERE status='OPEN' ORDER BY id").fetchall()
    con.close()
    return rows


def close(db, pos_id, exit_premium) -> float | None:
    con = _con(db)
    con.executescript(SCHEMA)
    p = con.execute("SELECT * FROM positions WHERE id=?", (pos_id,)).fetchone()
    if p is None or p["status"] != "OPEN":
        con.close()
        return None
    proceeds = p["contracts"] * exit_premium * CONTRACT
    realized = (exit_premium - p["entry_premium"]) * p["contracts"] * CONTRACT
    con.execute("UPDATE account SET cash=cash+? WHERE id=1", (proceeds,))
    con.execute("UPDATE positions SET status='CLOSED', exit_premium=?, exit_ts=?, realized_pnl=? WHERE id=?",
                (float(exit_premium), datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 realized, pos_id))
    con.commit()
    con.close()
    return realized


def summary(db: str = DB_PATH) -> dict:
    acct = get_or_create(db)
    con = _con(db)
    opens = con.execute("SELECT * FROM positions WHERE status='OPEN'").fetchall()
    closed = con.execute("SELECT * FROM positions WHERE status='CLOSED' ORDER BY id DESC").fetchall()
    con.close()

    pos_rows, open_value = [], 0.0
    for p in opens:
        cp, spot = current_premium(p["ticker"], p["opt_type"], p["strike"], p["expiry"])
        cur_val = p["contracts"] * cp * CONTRACT
        pnl = (cp - p["entry_premium"]) * p["contracts"] * CONTRACT
        open_value += cur_val
        pos_rows.append({**dict(p), "current_premium": round(cp, 2), "current_value": round(cur_val, 2),
                         "unrealized_pnl": round(pnl, 2),
                         "unrealized_pct": round((cp / p["entry_premium"] - 1) * 100, 1) if p["entry_premium"] else None,
                         "cur_spot": round(spot, 2)})

    cash, start = acct["cash"], acct["start_cash"]
    equity = cash + open_value
    realized = sum((c["realized_pnl"] or 0) for c in closed)
    return {"cash": cash, "start": start, "equity": equity, "open_value": open_value,
            "total_pnl": equity - start, "return_pct": (equity - start) / start * 100 if start else 0,
            "realized_pnl": realized, "unrealized_pnl": equity - start - realized,
            "n_open": len(opens), "positions": pos_rows, "closed": [dict(c) for c in closed]}


def log_snapshot(db: str = DB_PATH, current_equity: float | None = None,
                 current_cash: float | None = None) -> None:
    """Log today's equity (dedup by date — overwrites today's row on every call, so it always
    reflects the latest mark). No historical backfill: there's no free source for historical
    options pricing to mark old positions to market at past dates, so the equity curve only
    starts from whenever this is first called and fills in for real from there.

    `current_equity`/`current_cash`: pass values you already have (e.g. a cached `summary()` call)
    to skip re-running `summary()` here -- it does a live network round-trip PER open position, so
    calling it again just to log a snapshot is real, avoidable network cost."""
    if current_equity is None or current_cash is None:
        s = summary(db)
        current_equity = s["equity"] if current_equity is None else current_equity
        current_cash = s["cash"] if current_cash is None else current_cash
    today = date.today().isoformat()
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    con = _con(db)
    con.executescript(SCHEMA)
    con.execute(
        "INSERT INTO equity_history(date,ts,equity,cash) VALUES(?,?,?,?) "
        "ON CONFLICT(date) DO UPDATE SET ts=excluded.ts, equity=excluded.equity, cash=excluded.cash",
        (today, ts, current_equity, current_cash))
    con.commit()
    con.close()


def close_expired(db: str, positions: list[dict]) -> list[dict]:
    """Auto-settle any position in an already-computed positions list (e.g. from `summary()`)
    whose expiry has passed -- options don't just sit OPEN forever once expired, they settle to
    intrinsic value (or worthless) at expiration. Uses the `current_premium` ALREADY computed for
    each position (summary()'s current_premium() already returns intrinsic value for an expired
    contract, exactly what's already shown as "Now" in the table) -- this needs ZERO additional
    network calls, it just acts on data the caller already fetched. Returns the positions closed
    this call (each with its own `realized_pnl` added), for a one-line "auto-closed N" notice."""
    today = date.today()
    closed = []
    for p in positions:
        try:
            exp = datetime.strptime(p["expiry"], "%Y-%m-%d").date()
        except Exception:
            continue
        if exp < today:
            realized = close(db, p["id"], p["current_premium"])
            if realized is not None:
                closed.append({**p, "realized_pnl": realized})
    return closed


def equity_series(db: str = DB_PATH) -> list[dict]:
    con = _con(db)
    con.executescript(SCHEMA)
    rows = con.execute("SELECT * FROM equity_history ORDER BY date").fetchall()
    con.close()
    return [dict(r) for r in rows]


def _local_date(ts: str) -> str:
    """UTC ISO timestamp -> the local calendar date it falls on. entry_ts/exit_ts/created are all
    stored in UTC (datetime.now(timezone.utc)...), but log_snapshot()/equity_series() bucket by
    date.today() (local) -- bucketing historical_realized_equity's points by a raw UTC date
    substring instead of converting to local time is a real basis mismatch (found in this
    session's accounting audit): a late-evening local trade that's already past midnight UTC would
    land one day ahead in the merged curve, showing a stray future-dated point on the equity
    chart. Converting to local time before bucketing keeps both sources on the same calendar."""
    return datetime.fromisoformat(ts).astimezone().date().isoformat()


def historical_realized_equity(db: str = DB_PATH) -> list[dict]:
    """Reconstructs a REAL (not estimated) equity curve from the actual trade log: start_cash plus
    cumulative realized P&L at each closed trade's exit time. Every point is a recorded fact —
    exact entry/exit prices and timestamps already in the `positions` table.

    What this does NOT capture: unrealized P&L of positions that were still OPEN at each historical
    moment (there's no free source for historical options pricing to mark those to market
    retroactively). This is NOT a lower bound -- if an open position was a LOSER at some historical
    point, the true equity then was actually LOWER than what this curve shows, not higher. It's an
    honest omission (real recorded facts only, no fabrication), not a guaranteed floor in either
    direction."""
    acct = get_or_create(db)
    con = _con(db)
    closed = con.execute(
        "SELECT exit_ts, realized_pnl FROM positions WHERE status='CLOSED' AND exit_ts IS NOT NULL "
        "ORDER BY exit_ts").fetchall()
    con.close()
    running = acct["start_cash"]
    points = [{"date": _local_date(acct["created"]), "equity": running}]
    for c in closed:
        running += c["realized_pnl"] or 0.0
        points.append({"date": _local_date(c["exit_ts"]), "equity": running})
    return points


def equity_curve(db: str = DB_PATH, current_equity: float | None = None) -> list[dict]:
    """The fullest honest daily equity curve available: real daily snapshots (which include live
    unrealized P&L of open positions, logged going forward by `log_snapshot`) where they exist,
    falling back to the realized-P&L-only trade-log replay for earlier dates before snapshot
    logging began. Today always gets the current LIVE equity. One point per calendar date
    (last-of-day), sorted ascending, all on the same local-date basis. Every number traces back to
    a real recorded trade or a live snapshot — nothing here is estimated or fabricated.

    `current_equity`: pass an already-computed value (e.g. from a cached `summary()` call the
    caller already has) to skip re-running `summary()` here -- it does a live network round-trip
    PER open position (current_premium() -> get_spot()/get_option_chain()), so calling it again
    just to re-derive today's equity is real, avoidable network cost, not a free re-read."""
    by_date = {p["date"]: p["equity"] for p in historical_realized_equity(db)}
    by_date.update({h["date"]: h["equity"] for h in equity_series(db)})
    by_date[date.today().isoformat()] = current_equity if current_equity is not None else summary(db)["equity"]
    return [{"date": d, "equity": by_date[d]} for d in sorted(by_date)]


def day_pnl(db: str = DB_PATH, current_equity: float | None = None) -> dict | None:
    """Today's P&L vs the most recent PRIOR day's equity in the combined curve. None only if
    there's truly no prior data point at all (brand-new account) — never fabricated/estimated.
    Checks `is None` explicitly, not falsiness -- equity can legitimately be exactly $0.00 (fully
    spent down), and treating that as "no data" would silently hide a real -100% day.

    `current_equity`: same as equity_curve() -- pass a value you already have to avoid two more
    redundant summary() network round-trips (one here, one inside equity_curve())."""
    curve = equity_curve(db, current_equity=current_equity)
    today = date.today().isoformat()
    prior = [p for p in curve if p["date"] < today]
    if not prior or prior[-1]["equity"] is None:
        return None
    prev_equity = prior[-1]["equity"]
    cur_equity = current_equity if current_equity is not None else summary(db)["equity"]
    return {"pnl": cur_equity - prev_equity, "pct": (cur_equity / prev_equity - 1) * 100}