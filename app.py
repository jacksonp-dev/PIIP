"""PIIP visual layer (Streamlit).  Run:  streamlit run app.py

A clickable front-end over the same engine the CLI uses. Deterministic analysis is free; the LLM
layer is opt-in per click on the Ticker Page (real spend, cost-governed) or via the CLI's
--live-llm flag. Research/education tool — NOT investment advice.
"""
import base64
import io
import json
import math
import os
import struct
import wave
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
import streamlit as st

from iip import agents
from iip import appstate
from iip import baseline, backtest as bt, data
from iip import catalyst_calibration as catcal
from iip import catalyst_terminal as ct
from iip import deterministic as det
from iip import deep_research as dr
from iip import predictions as pred
from iip import assess
from iip import fundamentals
from iip import glossary
from iip import journal
from iip import macro
from iip import market_dna as mdna
from iip import portfolio
from iip import premarket_ai as pai
from iip import premarket_backtest as pbt
from iip import premarket_thesis as pt
from iip import reddit_momentum as rm
from iip import scanner
from iip import screener
from iip import scorer
from iip import sec_edgar
from iip import social
from iip import timeframe as tf
from iip import update_check
from iip import watchlist as wl
from iip import zero_dte as zd
from iip import zero_dte_log as zdlog

DB = "iip.db"   # single on-disk database — predictions AND the paper account persist here


def _pnl_color(v):
    if v > 0:
        return "background-color: rgba(76,175,80,0.25)"
    if v < 0:
        return "background-color: rgba(244,67,54,0.22)"
    return ""


# Shared muted tile-background lookup for the assess.py-style emoji flags (🟢🟡🔴⚪) used across
# Feed/Reddit Momentum/Lottery/Catalysts -- dark, desaturated tints matching this app's existing
# dark theme (not bright full-saturation fills), one definition so every card-style page reads
# consistently instead of each page inventing its own palette.
_TILE_BY_EMOJI = {
    "🟢": {"bg": "#152a1e", "border": "#2c5a3c"},
    "🟡": {"bg": "#2a2413", "border": "#5a4a1f"},
    "🔴": {"bg": "#2a1515", "border": "#5a2c2c"},
    "⚪": {"bg": "#15191a", "border": "#232b2d"},
    "🔵": {"bg": "#132530", "border": "#2c4a5a"},
}


def _tile_style(emoji: str) -> dict:
    return _TILE_BY_EMOJI.get(emoji, _TILE_BY_EMOJI["⚪"])


# Bright text companion to the muted tile backgrounds above -- same palette used everywhere else
# in the app (success/warning/danger/info) for the one accent value that should stand out on a tile.
_TEXT_BY_EMOJI = {"🟢": "#79ed8e", "🟡": "#fabf6b", "🔴": "#ff8080", "⚪": "#8b9a9d", "🔵": "#87d1ff"}


def _esc(s) -> str:
    """Escape free text before dropping it into raw HTML -- crucially turns $ into &#36; so a
    dollar amount can't get misread as a LaTeX delimiter (st.markdown treats a $...$ pair as math
    notation; AI-generated text is full of paired dollar amounts like "$1,000 start, now at $836",
    which rendered as garbled italic math -- confirmed live on the Paper page's AI section)."""
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace("$", "&#36;"))


def _md_safe(s) -> str:
    """Same $-as-LaTeX fix as _esc(), for plain (non-HTML) st.markdown/st.caption calls -- markdown
    treats a backslash-escaped \\$ as a literal dollar sign instead of a math delimiter."""
    return str(s).replace("$", "\\$")


def _info_header(label: str, tooltip: str, size: str = "0.95rem",
                 color: str = "#87d1ff") -> None:
    """Section header with a bright, noticeable hover-info icon (per user request, app-wide
    cohesiveness pass 2026-08) -- replaces long always-visible caption paragraphs that used to sit
    permanently under a header, eating vertical space, with a single-line header plus an info icon
    that reveals the same text on hover.

    Renders a plain "i" glyph, NOT the pre-circled Unicode "ⓘ" -- confirmed live that stacking the
    already-circled glyph inside this function's OWN circular badge produced a lumpy double-circle
    blob (reported by the user as looking like "a butt"), not a clean icon. A plain letter inside
    one single badge shape avoids that entirely.

    Deliberately custom HTML (colored circular badge + native `title` attribute) instead of
    Streamlit's own `st.caption(..., help=...)` icon -- confirmed live the native one renders as a
    small pale grey circle that's easy to miss entirely; this is the same title-attribute tooltip
    pattern already used for the DNA metrics tiles, just packaged as a reusable header. `color`
    defaults to the app's existing info/neutral accent (#87d1ff, already used for Breadth "Mixed",
    NVDA RS "In-line", etc.) -- pass a different bright platform color (e.g. #79ed8e green,
    #fabf6b amber) to match a specific section's own accent instead."""
    safe_tooltip = _esc(tooltip).replace('"', "&quot;")
    safe_label = _esc(label)
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:0.45rem;margin:0.1rem 0 0.2rem 0">'
        f'<span style="font-weight:700;font-size:{size}">{safe_label}</span>'
        f'<span title="{safe_tooltip}" style="display:inline-flex;align-items:center;'
        f'justify-content:center;width:1.15rem;height:1.15rem;border-radius:50%;flex-shrink:0;'
        f'background:{color};color:#0a0f10;font-family:Georgia,\'Times New Roman\',serif;'
        f'font-size:0.8rem;font-weight:800;font-style:italic;cursor:help;'
        f'line-height:1">i</span></div>', unsafe_allow_html=True)


def _render_kpi_tiles(s: dict, dpnl: dict | None):
    """Shared 6-tile KPI grid (Equity/Today/Realized P&L, Cash available/Open positions/
    Unrealized P&L) -- used by BOTH the Home page and the Paper tab's positions block
    (_render_positions_block below) so the two read identically instead of drifting apart, same
    'pixel-identical' reasoning as that function's shared positions table. Originally only built
    for Home; the user pointed out Paper still showed the old plain st.metric row after all of
    Home's redesign work, which read as inconsistent/"reverted" even though nothing had actually
    regressed -- Paper just never got the same treatment. Order matches the user's marked-up
    screenshot from the Home redesign: Equity/Today/Realized P&L on top, Cash available/Open
    positions/Unrealized P&L on the bottom. Reuses the SAME _tile_style()/_TEXT_BY_EMOJI palette
    already used app-wide for Feed/Lottery/Catalysts cards, not a one-off color scheme."""
    total_arrow = "▲" if s["total_pnl"] >= 0 else "▼"
    eq_emoji = "🟢" if s["total_pnl"] >= 0 else "🔴"
    eq_tile, eq_text = _tile_style(eq_emoji), _TEXT_BY_EMOJI[eq_emoji]
    day_emoji = ("🟢" if dpnl["pnl"] >= 0 else "🔴") if dpnl else "⚪"
    day_tile, day_text = _tile_style(day_emoji), _TEXT_BY_EMOJI[day_emoji]
    cash_tile = _tile_style("🔵")
    open_tile = _tile_style("⚪")
    real_emoji = "🟢" if s["realized_pnl"] >= 0 else "🔴"
    real_tile, real_text = _tile_style(real_emoji), _TEXT_BY_EMOJI[real_emoji]
    unreal_emoji = "🟢" if s["unrealized_pnl"] >= 0 else "🔴"
    unreal_tile, unreal_text = _tile_style(unreal_emoji), _TEXT_BY_EMOJI[unreal_emoji]

    def _kpi_tile(col, tile, label, big, big_color, sub, sub_color="#8b9a9d"):
        tip = glossary.help_for(label).replace('"', "'")
        title_attr = f' title="{tip}"' if tip else ""
        col.markdown(
            f'<div style="background:{tile["bg"]};border:1px solid {tile["border"]};'
            'border-radius:10px;padding:1rem 1.1rem;height:100%">'
            f'<div style="color:#8b9a9d;font-size:0.72rem;text-transform:uppercase;'
            f'letter-spacing:0.05em;margin-bottom:0.5rem"{title_attr}>{label}</div>'
            f'<div style="font-size:1.5rem;font-weight:700;color:{big_color}">{big}</div>'
            f'<div style="font-size:0.8rem;margin-top:0.3rem;font-weight:600;color:{sub_color}">'
            f'{sub}</div></div>', unsafe_allow_html=True)

    t1, t2, t3 = st.columns(3)
    _kpi_tile(t1, eq_tile, "Equity", f"${s['equity']:,.2f}", "#e8ecec",
             f"{total_arrow} ${abs(s['total_pnl']):,.2f} ({s['return_pct']:+.1f}%)", eq_text)
    if dpnl:
        day_arrow = "▲" if dpnl["pnl"] >= 0 else "▼"
        _kpi_tile(t2, day_tile, "Today", f"{day_arrow} ${abs(dpnl['pnl']):,.2f}", day_text,
                 f"{dpnl['pct']:+.1f}% today", day_text)
    else:
        _kpi_tile(t2, day_tile, "Today", "—", "#e8ecec", "Shows once a prior day is logged")
    _kpi_tile(t3, real_tile, "Realized P&L", f"${s['realized_pnl']:+,.2f}", real_text,
             "from closed trades")
    t4, t5, t6 = st.columns(3)
    _kpi_tile(t4, cash_tile, "Cash available", f"${s['cash']:,.2f}", "#e8ecec",
             f"of ${s['equity']:,.2f} equity")
    _kpi_tile(t5, open_tile, "Open positions", str(s["n_open"]), "#e8ecec",
             f"${s['open_value']:,.2f} value")
    _kpi_tile(t6, unreal_tile, "Unrealized P&L", f"${s['unrealized_pnl']:+,.2f}", unreal_text,
             "on open positions")


def _score_color(score: float, low: float = 40, high: float = 65) -> str:
    if score >= high:
        return "#79ed8e"
    if score <= low:
        return "#ff8080"
    return "#fabf6b"


def _pos_dte(exp: str) -> int | None:
    try:
        return (datetime.strptime(exp, "%Y-%m-%d").date() - date.today()).days
    except Exception:
        return None


def _pnl_text_color(v):
    if v is None or v != v:
        return ""
    return f"color: {'#79ed8e' if v >= 0 else '#ff8080'}; font-weight: 600"


def _signed_money(v):
    return "" if v is None or v != v else f"{'+' if v >= 0 else '-'}${abs(v):,.0f}"


def _signed_pct(v):
    return "—" if v is None or v != v else f"{v:+.0f}%"


def _pnl_bar_style(v, vmax):
    """Translucent colored bar behind the value, width proportional to |v| vs the column's own
    max -- same 'polish, not rebuild' treatment already chosen for the options chain, applied here."""
    if v is None or v != v or not vmax:
        return _pnl_text_color(v)
    w = min(abs(v) / vmax, 1.0) * 100
    rgb = "121,237,142" if v >= 0 else "255,128,128"
    return (f"background: linear-gradient(90deg, rgba({rgb},0.22) 0%, rgba({rgb},0.22) {w:.0f}%, "
            f"transparent {w:.0f}%); color: {'#79ed8e' if v >= 0 else '#ff8080'}; font-weight:600;")


def _safe_abs_max(series, default: float = 1.0) -> float:
    """|max| of a pandas column, safe against the all-missing case. `series.abs().max()` looks
    fine until every value is None -- then the column stays object-dtype and .abs() raises
    TypeError outright (confirmed live: a Watchlist/Screener row where every ticker's fetch failed,
    or a Positions table with no P&L yet, would crash the whole page, not just show a flat bar).
    pd.to_numeric(..., errors='coerce') forces real NaNs instead of raising, and pd.isna() catches
    both None and NaN so the empty-column case falls back to `default` instead of propagating NaN
    into every row's bar-width calculation downstream (which would render as literal "nan%" CSS)."""
    m = pd.to_numeric(series, errors="coerce").abs().max()
    return default if pd.isna(m) else float(m)


def _fetch_paper_summary():
    """Shared cache for both the Paper tab and the global Positions popover -- live-marks
    positions to market only when stale (no cache yet, explicitly flagged, or open-count changed),
    never on every rerun. open_positions() is a cheap local SQLite read, safe to call unconditionally."""
    live_n = len(portfolio.open_positions(DB))
    if ("paper_summary" not in st.session_state or st.session_state.pop("paper_stale", False)
            or st.session_state["paper_summary"].get("n_open") != live_n):
        with st.spinner("Marking open positions to live market…"):
            st.session_state["paper_summary"] = portfolio.summary(DB)
    return st.session_state["paper_summary"]


@st.fragment
def _render_positions_block(s: dict, key_prefix: str):
    """Top metrics + the open-positions table with inline P&L bars -- shared by the full Paper tab
    and the global Positions popover so both stay pixel-identical instead of drifting apart.
    key_prefix keeps widget keys unique when both are on screen in the same rerun (e.g. the popover
    open while already on the Paper tab).

    @st.fragment: ticking a row checkbox uses on_select="rerun", which without a fragment reruns
    the ENTIRE app (nav, both popovers, everything) -- visible as Streamlit's whole-page "running"
    fade on every single checkbox click, for something that should feel instant/local. Scoping to
    a fragment means only this table's own subtree reruns on selection. "Close selected" still
    calls plain st.rerun() (not scope="fragment") deliberately -- closing a position needs a real
    full-app rerun so the Positions button's live count and the Paper tab's other metrics actually
    refresh, which is exactly the one action that SHOULD show the normal loading state."""
    # Same 6-tile KPI grid as the Home page (_render_kpi_tiles, shared) -- was a plain
    # st.metric row here, which after Home's redesign work read as inconsistent/"reverted" even
    # though nothing had regressed; Paper just never got the same treatment. day_pnl() is a cheap
    # local SQLite read (no network call), safe to compute fresh on every fragment render.
    dpnl = portfolio.day_pnl(DB, current_equity=s["equity"])
    _render_kpi_tiles(s, dpnl)

    st.subheader("Open positions")
    if not s["positions"]:
        st.write("None yet — buy some from the 📰 Feed tab.")
        return

    group_by = st.radio("Group by", ["None", "Ticker", "Type"], horizontal=True,
                        key=f"{key_prefix}_groupby", label_visibility="collapsed")
    # Sort ONE list and derive both the display frame AND the close-action index mapping from it --
    # `event.selection.rows` below indexes into whatever order actually got displayed, so grouping
    # (sorting) `positions` and building `odf` from that same list keeps "Close selected" pointing
    # at the right position even when the table's no longer in raw open-date order.
    positions = s["positions"]
    if group_by == "Ticker":
        positions = sorted(positions, key=lambda p: (p["ticker"], p["expiry"]))
    elif group_by == "Type":
        positions = sorted(positions, key=lambda p: (p["opt_type"], p["ticker"]))

    odf = pd.DataFrame([{
        "Contract": f"{p['ticker']} {p['opt_type'].upper()} ${p['strike']:g}",
        "Opened": (p.get("entry_ts") or "")[:10],
        "Expiry": p["expiry"] + (f"  ({d}d)" if (d := _pos_dte(p["expiry"])) is not None else ""),
        "Qty": p["contracts"],
        "Entry": p["entry_premium"],
        "Now": p["current_premium"],
        "Cost": p["entry_premium"] * p["contracts"] * 100,
        "Value": p["current_value"],
        "P&L $": p["unrealized_pnl"],
        "P&L %": p["unrealized_pct"],
    } for p in positions])
    max_pnl_dollar = _safe_abs_max(odf["P&L $"])
    max_pnl_pct = _safe_abs_max(odf["P&L %"])
    styled = (odf.style
              .apply(lambda col: [_pnl_bar_style(v, max_pnl_dollar) for v in col], subset=["P&L $"])
              .apply(lambda col: [_pnl_bar_style(v, max_pnl_pct) for v in col], subset=["P&L %"])
              .format({"Entry": "${:.2f}", "Now": "${:.2f}", "Cost": "${:,.0f}",
                       "Value": "${:,.0f}", "P&L $": _signed_money, "P&L %": _signed_pct}))
    # width="stretch" + explicit per-column widths, not "content" -- same fix as Home's positions
    # table: "content" left a large dead gap next to the table below a full-width page instead of
    # filling it, and plain "stretch" (no column_config widths) re-introduces the OTHER bug this
    # session already fixed for Watchlist/Screener (every column, including 1-2 digit Qty/P&L
    # cells, padded out evenly). "medium" for the 3 text columns concentrates the fill there.
    pos_col_cfg = {c: st.column_config.Column(c, help=glossary.help_for(c),
                   width="medium" if c in ("Contract", "Opened", "Expiry") else "small")
                   for c in odf.columns}
    event = st.dataframe(styled, width="stretch", hide_index=True,
                         height=(len(odf) + 1) * 35 + 3,
                         column_config=pos_col_cfg,
                         on_select="rerun", selection_mode="multi-row", key=f"{key_prefix}_openpos_tbl")
    tot_cost = sum(p["entry_premium"] * p["contracts"] * 100 for p in s["positions"])
    tot_val = sum(p["current_value"] for p in s["positions"])
    tot_pnl = tot_val - tot_cost
    tot_pct = (tot_pnl / tot_cost * 100) if tot_cost else 0.0
    m = st.columns([1, 1, 1, 4])
    m[0].metric("Total cost basis", f"${tot_cost:,.0f}", help=glossary.help_for("Total cost basis"))
    m[1].metric("Current value", f"${tot_val:,.0f}", help=glossary.help_for("Current value"))
    m[2].metric("Unrealized P&L", f"${tot_pnl:+,.0f}", f"{tot_pct:+.1f}%",
                help=glossary.help_for("Unrealized P&L"))
    sel = list(event.selection.rows) if getattr(event, "selection", None) else []
    cA, cB = st.columns([1, 4])
    if cA.button(f"✖ Close selected ({len(sel)})", type="primary", disabled=not sel,
                key=f"{key_prefix}_closesel"):
        for i in sel:
            p = positions[i]   # the SAME (possibly grouped/sorted) list the table was built from
            portfolio.close(DB, p["id"], p["current_premium"])
        st.session_state["paper_stale"] = True
        st.rerun()
    cB.caption("Tick the checkbox on a row (or several), then **Close selected**.  "
               "**Cost** = what you paid in · **Value** = current mark · **P&L** = unrealized "
               "(mid price, no fees/spread — real exits are worse).")


# Shared tight click-to-expand row list, used by both 0DTE Intelligence and Reddit Momentum.
# Previously an embedded st.iframe replicating a mockup's inline click-to-expand rows (native
# st.expander's fixed chrome couldn't match the spacing). Dropped that: an auto-height iframe
# rebuilt every 30s inside a st.fragment(run_every=30) was observed stacking up in the browser
# instead of being replaced -- the page grew without bound. Rebuilt with plain native widgets/
# markdown instead, which Streamlit already knows how to diff and replace in place each rerun.
# `container_key` must be unique per call site (becomes the CSS scope + every row's session_state
# key) so two list views on different pages never collide.
def _render_list_view(groups: list[dict], container_key: str = "zd_list_rows"):
    """One combined markdown call per row (label/context/value in a single CSS-grid div) plus one
    button, not three separate markdown calls + a button + an <hr> -- stacking that many elements
    let Streamlit's default per-element block margin add up to ~1.5rem of dead space per row.
    Scoped CSS via st.container(key=...)'s auto-generated `st-key-<container_key>` class collapses
    that remaining margin to zero so only this block's own deliberate padding controls row rhythm.
    Rows with a `"ticker"` field get a 'Research this' button under their evidence when expanded,
    navigating to the Ticker Page -- optional per row, unused rows just omit the key."""
    st.markdown(
        f'<style>'
        f'.st-key-{container_key} {{ background:#0c1414 !important; }}'
        f'.st-key-{container_key} [data-testid="stElementContainer"] {{ margin-bottom: 0 !important; }}'
        f'.st-key-{container_key} [data-testid="stHorizontalBlock"] {{ gap: 0.4rem !important; '
        f'align-items: center !important; }}'
        # The [0.6, 9.4] column ratio still leaves a widening gap between arrow and text on wider
        # pages -- Streamlit sizes columns as a PROPORTION of the container, so 0.6/10 is a bigger
        # absolute pixel gap the wider the page/panel is (confirmed from a live screenshot: fine at
        # the width this was first tuned at, visibly "spaced out" in a wider placement). Forcing the
        # first column to a fixed pixel width -- not a ratio -- decouples it from container width
        # entirely, same trick already used for the Trade/Positions popover panel width above.
        f'.st-key-{container_key} [data-testid="stHorizontalBlock"] > div:first-child {{ '
        f'flex: 0 0 26px !important; width: 26px !important; min-width: 26px !important; '
        f'max-width: 26px !important; }}'
        f'.st-key-{container_key} [data-testid="stHorizontalBlock"] > div:nth-child(2) {{ '
        f'flex: 1 1 auto !important; width: auto !important; max-width: none !important; }}'
        # flex-end, not flex-start: the button column is wider than the button itself (Streamlit's
        # column ratio reserves real width regardless of content), so left-aligning left a visible
        # gap between the arrow and the row text -- hugging the right edge of its own column
        # instead puts the arrow immediately next to the text with no dead space in between.
        f'.st-key-{container_key} [data-testid="stButton"] {{ display: flex; justify-content: flex-end; }}'
        f'.st-key-{container_key} [data-testid="stButton"] button {{ padding: 0 0.4rem !important; '
        f'min-height: 1.7rem !important; height: 1.7rem !important; line-height: 1 !important; '
        # Dim to a faint hint by default; only the row you're actually hovering brightens its own
        # arrow -- addresses "such a big nuisance" without giving up the click target entirely
        # (a pure hover-to-appear-from-nothing needs an overlay hack that's fragile across browsers).
        f'opacity: 0.25 !important; transition: opacity 120ms ease; }}'
        f'.st-key-{container_key} [data-testid="stHorizontalBlock"]:hover [data-testid="stButton"] button '
        f'{{ opacity: 1 !important; }}'
        f'</style>', unsafe_allow_html=True)
    with st.container(key=container_key, border=True):
        for gi, group in enumerate(groups):
            # Group headers go through the SAME [0.6, 9.4] column split as rows below, purely so
            # the header text lines up flush with row labels -- two independently-sized layouts
            # (a full-width header vs a button+content row) would drift apart at different screen
            # widths since Streamlit's column fractions aren't a fixed pixel offset.
            hl, hr = st.columns([0.6, 9.4])
            hr.markdown(
                f'<div style="font-family:ui-monospace,Consolas,monospace;font-size:0.8rem;'
                f'letter-spacing:0.06em;text-transform:uppercase;color:#87d1ff;font-weight:700;'
                f'background:#111c1d;padding:0.5rem 0.6rem 0.3rem;border-radius:4px'
                f'{"" if gi == 0 else ";margin-top:0.4rem"}">'
                f'{group["label"]}</div>', unsafe_allow_html=True)
            for ri, r in enumerate(group["rows"]):
                row_key = f"{container_key}_{gi}_{ri}_{r['label']}"
                left, right = st.columns([0.6, 9.4])
                expanded = st.session_state.get(row_key, False)
                if left.button("▾" if expanded else "▸", key=row_key + "_btn"):
                    st.session_state[row_key] = not expanded
                tip = glossary.help_for(r["label"]).replace('"', "'")
                title_attr = f' title="{tip}"' if tip else ""
                right.markdown(
                    '<div style="display:grid;grid-template-columns:1.3fr 3.2fr 1.1fr;'
                    f'align-items:center;gap:0.6rem;padding:0.3rem 0;border-top:1px solid #142020">'
                    f'<div style="font-size:0.85rem;font-weight:600"{title_attr}>{r["label"]}</div>'
                    f'<div style="font-size:0.78rem;color:#8b9a9d;overflow:hidden;text-overflow:ellipsis;'
                    f'white-space:nowrap">{r.get("context") or ""}</div>'
                    f'<div style="text-align:right"><span style="color:{r["color"]}">●</span> '
                    f'<span style="font-family:ui-monospace,Consolas,monospace;font-weight:700;'
                    f'color:{r["color"]}">{r["value"]}</span></div></div>', unsafe_allow_html=True)
                if st.session_state.get(row_key, False):
                    # Inside `right`, not a bare st.caption/st.button -- those render at the
                    # container's own left edge (under the arrow column), not aligned with the
                    # row content above, which is exactly the indentation bug this was reported as.
                    for e in r.get("evidence", []):
                        right.caption(f"· {e}")
                    if r.get("ticker"):
                        if right.button(f"🔭 Research {r['ticker']} →", key=row_key + "_research"):
                            st.session_state["tp_free"] = r["ticker"]
                            st.session_state.nav = "Ticker Page"
                            st.rerun()


def scenario_grid(spot, strike, dte, iv, is_call, entry_premium):
    """Est. P&L per contract (Black-Scholes) if the stock moves X% (rows) over time (cols) — the
    'sell early' what-if. IV held constant. Captures delta (the move) + theta (time)."""
    moves = [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20]
    pts = [("now", 0)]
    if dte >= 6:
        pts += [(f"+{dte // 3}d", dte // 3), (f"+{2 * dte // 3}d", 2 * dte // 3)]
    elif dte >= 2:
        pts += [(f"+{dte // 2}d", dte // 2)]
    pts.append((f"exp {dte}d", dte))
    cols = {}
    for lab, el in pts:
        Tp = max((dte - el) / 365, 1e-6)
        cols[lab] = [round((det.bs_price(spot * (1 + m), strike, Tp, 0.045, iv, call=is_call)
                            - entry_premium) * 100) for m in moves]
    return pd.DataFrame(cols, index=[f"{m * 100:+.0f}%" for m in moves])

FAVICON = Path(__file__).parent / "assets" / "favicon.png"   # absolute path — robust to CWD

# TradingView Lightweight Charts (Apache 2.0), vendored locally at assets/vendor/ -- PIIP audit
# 2026-08, per user request for a real trading-terminal-feel intraday chart (auto price-scale on
# zoom, native crosshair, smooth pan/zoom) that Altair/Vega-Lite (a statistical-charting grammar,
# not built for this) can't deliver well. Downloaded once via
# https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js and saved
# to the repo -- inlined into the HTML component at render time, NEVER fetched from a CDN at
# runtime, matching this project's local-first rule the same way Altair itself is a bundled
# Python dependency, not a runtime fetch.
LIGHTWEIGHT_CHARTS_JS = Path(__file__).parent / "assets" / "vendor" / "lightweight-charts.standalone.production.js"


@st.cache_resource(show_spinner=False)
def _load_lightweight_charts_js() -> str:
    """Read the vendored library once per server process -- it's a static ~200KB file that never
    changes at runtime, cache_resource (not cache_data) since this is a big constant string with
    no TTL/per-arg cache key needed.

    PIIP audit 2026-08: defined here, right after the LIGHTWEIGHT_CHARTS_JS constant, NOT down by
    whichever chart function needed it first -- a real bug this session: Streamlit scripts execute
    top-to-bottom as a flat script, `if nav == "X":` blocks run immediately as they're reached, so
    a helper called from an EARLIER page (Home) needs to already be defined before that page's
    block runs, not merely before the module finishes loading. Every chart-rendering function on
    every page shares this one definition."""
    return LIGHTWEIGHT_CHARTS_JS.read_text(encoding="utf-8")


def _et_seconds(ts) -> int:
    """PIIP audit 2026-08: Lightweight Charts renders numeric `time` values as UTC and displays
    them as-is -- no per-viewer timezone conversion. Reinterpreting the ET wall-clock numbers AS
    IF they were UTC (strip tz, treat the naive value as UTC) makes the chart display the correct
    ET hour:minute regardless of the viewer's own browser timezone, which is what matters here
    since market hours are inherently ET-based, not the viewer's locale. Only needed for INTRADAY
    (sub-daily) series -- daily-bar charts use a plain {year, month, day} time object instead,
    which has no timezone ambiguity to correct for."""
    naive = ts.tz_localize(None) if ts.tzinfo is not None else ts
    return int(naive.replace(tzinfo=timezone.utc).timestamp())


# "owner/repo" the in-app Feedback page opens pre-filled GitHub issues against. Update this once
# the repo is actually pushed -- placeholder until then.
GITHUB_REPO = "jacksonp-dev/PIIP"

st.set_page_config(page_title="PIIP — Personal Investment Intelligence", layout="wide",
                    page_icon=str(FAVICON))


def _write_env_file(anthropic_key: str, finnhub_key: str, contact_email: str) -> None:
    lines = ["# PIIP secrets -- gitignored, NEVER commit this file.\n"]
    if anthropic_key:
        lines.append(f'ANTHROPIC_API_KEY="{anthropic_key}"\n')
    if finnhub_key:
        lines.append(f'FINNHUB_API_KEY="{finnhub_key}"\n')
    if contact_email:
        lines.append(f'RESEARCH_CONTACT_EMAIL="{contact_email}"\n')
    Path(".env").write_text("".join(lines), encoding="utf-8")


# First-run-only setup gate for a portable-installer clone that has no .env yet (a normal
# developer checkout already has one, so this never shows there). st.stop() below means nothing
# else in this script -- nav, the welcome dialog, every page -- runs on THIS pass; once the form
# is submitted, .env exists and the very next rerun skips straight past this block entirely.
if not Path(".env").exists():
    st.title("👋 Welcome to PIIP — first-time setup")
    st.markdown(
        "**This app runs entirely on YOUR computer.** It is not a hosted website — there is no "
        "PIIP server anywhere else involved. Everything you enter below is saved to a local "
        "`.env` file on this machine only; nothing is sent anywhere by this platform.\n\n"
        "Every field below is **optional** — skip all of them and click Continue to explore the "
        "free research and paper-trading features right away. You can always add a key later by "
        "editing the `.env` file created in this folder.")
    # Link buttons live OUTSIDE the form -- st.form only allows st.form_submit_button as its
    # "clickable" element, not st.link_button, so the get-a-key actions sit right above each
    # field instead of inside the form itself.
    st.subheader("🤖 Anthropic API key")
    st.caption("Powers the optional AI-interpretation layer on top of the free deterministic "
              "engine — pay-as-you-go, a few cents per research run. Skip this and the app still "
              "works fully on free data; the LLM layer just stays off.")
    st.link_button("Get an Anthropic API key →", "https://console.anthropic.com/settings/keys")

    st.subheader("📰 Finnhub API key")
    st.caption("Powers the 0DTE Intelligence page's Catalyst Terminal news feed — free tier. Skip "
              "this and that one section shows a setup hint instead of headlines; everything "
              "else works normally.")
    st.link_button("Get a free Finnhub API key →", "https://finnhub.io/register")

    st.subheader("✉️ Research contact email")
    st.caption("Sent as a courtesy contact in requests to Wikipedia and SEC EDGAR — both sites' "
              "own fair-access policies ask automated tools to include a real way to reach "
              "whoever's running them. Not shared with anyone else, not used for anything but "
              "that. Any email works — it's just so those two sites have a real contact.")

    with st.form("first_run_setup"):
        anthropic_key = st.text_input("Paste your Anthropic API key here (optional)",
                                      type="password", key="setup_anthropic")
        finnhub_key = st.text_input("Paste your Finnhub API key here (optional)",
                                    type="password", key="setup_finnhub")
        contact_email = st.text_input("Your email (optional)", key="setup_email")
        submitted = st.form_submit_button("Continue →", type="primary")
    if submitted:
        _write_env_file(anthropic_key.strip(), finnhub_key.strip(), contact_email.strip())
        st.rerun()
    st.stop()


@st.dialog("📊 Welcome to PIIP")
def _welcome_disclaimer():
    """Every launch, forever, unless the user checks 'don't show again' -- that choice is
    persisted via appstate's SQLite key-value store, not st.session_state, since session state
    resets every browser session and can't remember a real "never again" preference across
    restarts. `welcome_dismissed_this_session` (session-only) covers the OTHER case -- clicking
    "Got it" WITHOUT the checkbox should still stop it from reopening for the rest of THIS
    session, without permanently silencing it for next time.

    Gating this call on an "already seen" flag set the FIRST time this runs (rather than only
    once the dialog is actually dismissed) was tried first and was a real bug: st.dialog reruns
    the whole script on every widget interaction INSIDE it too (e.g. ticking the checkbox), and
    if the outer `if` had already flipped to "seen" from that very first render, the checkbox's
    own rerun would find the gate false and stop calling this function -- the dialog would never
    correctly reflect the checkbox being ticked. The gate below only changes state once the user
    actually clicks the button, so every interaction before that keeps re-opening/redrawing the
    SAME dialog with its latest widget values, which is exactly what a dialog needs."""
    st.markdown(
        "**This app runs entirely on YOUR computer.** It is not a hosted website — there is no "
        "server anywhere else involved. Your data, your account, and any API keys you enter stay "
        "on this machine; nothing is uploaded anywhere by this platform itself.\n\n"
        "This is a **free, open-source research tool** — every data source it uses is free, and "
        "some real data (paywalled institutional feeds, true real-time quotes, some order-flow / "
        "dealer positioning) simply isn't accessible here. Where a gap like that exists, it's "
        "disclosed right on the page, not silently filled in or estimated without a label.\n\n"
        "This platform is for **research and paper trading only** — helping you make more "
        "educated decisions. It does **not** execute real trades, and nothing here is financial "
        "advice.")
    dont_show = st.checkbox("Don't show this again", key="welcome_dont_show")
    if st.button("Got it, let's go", type="primary"):
        if dont_show:
            appstate.set(DB, "disclaimer_dismissed", "1")
        st.session_state["welcome_dismissed_this_session"] = True
        st.rerun()


if (appstate.get(DB, "disclaimer_dismissed") != "1"
        and not st.session_state.get("welcome_dismissed_this_session")):
    _welcome_disclaimer()


@st.cache_data(ttl=900, show_spinner=False)
def load_prices(tk):
    return data.get_prices(tk, "2y")


@st.cache_data(ttl=900, show_spinner=False)
def load_chain(tk, days):
    return data.get_option_chain(tk, data.nearest_expiry(tk, days))


@st.cache_data(ttl=900, show_spinner=False)
def _cached_expiries(tk):
    return data.list_expiries(tk)


@st.cache_data(ttl=900, show_spinner=False)
def _cached_chain(tk, expiry):
    return data.get_option_chain(tk, expiry)


@st.cache_data(ttl=120, show_spinner=False)   # short cache — intraday is time-sensitive
def get_intraday_snapshot(tk):
    """Fetches the bars, then delegates the VWAP/range/day-change math to det.intraday_snapshot()
    -- this used to reimplement that math inline (a duplicate this session's loose-ends audit
    flagged), adding only the chart DataFrame this caller needs on top, same pattern zero_dte.py
    already uses correctly."""
    df = data.get_intraday(tk, "5m", "1d")
    if df is None or df.empty:
        df = data.get_intraday(tk, "15m", "5d")   # fallback if today's session is empty
    snap = det.intraday_snapshot(df)
    if snap is None:
        return None
    c, v = df["Close"], df["Volume"]
    snap["chart"] = pd.DataFrame({"price": c, "VWAP": (c * v).cumsum() / v.cumsum().replace(0, 1)})
    return snap


@st.cache_data(ttl=900, show_spinner=False)
def run_scan():
    return scanner.scan()


# 12h TTL (PIIP audit 2026-08) -- a version check has no reason to hit the network more than a
# couple times a day; iip/update_check.py already swallows every failure (offline, GitHub down,
# repo not public) and returns None rather than raising, so caching a None result just means the
# next real attempt happens on the next cache expiry, never a stuck error state.
@st.cache_data(ttl=43200, show_spinner=False)
def check_for_piip_update():
    return update_check.check_for_update(GITHUB_REPO)


# 5min TTL, not the 30s the rest of 0DTE Intelligence refreshes at -- yields/DXY/oil/equal-weight
# breadth are genuinely slower-moving context (the whole reason this section is deprioritized to
# the bottom of the page), so refetching them every single 30s fragment rerun would just be wasted
# network calls for numbers that haven't meaningfully changed.
@st.cache_data(ttl=300, show_spinner=False)
def get_macro_batch():
    return macro.fetch_macro_batch()


@st.cache_data(ttl=300, show_spinner=False)
def get_rsp_vs_spy():
    return zd.fetch_daily_batch(tickers=["RSP", "SPY"], period="5d")


# 1-hour TTL, not 300s like the price-driven macro batch above -- TGA/RRP/bank reserves update at
# most daily and CPI/unemployment/payrolls only once a month, so refetching every 5 minutes would
# just be repeated network calls for numbers that provably haven't changed since the last release.
@st.cache_data(ttl=3600, show_spinner=False)
def get_liquidity_snapshot():
    return macro.liquidity_snapshot()


@st.cache_data(ttl=3600, show_spinner=False)
def get_econ_releases():
    return macro.economic_releases_snapshot()


@st.cache_data(ttl=1800, show_spinner=False)
def get_company_info(tk):
    info = fundamentals.company_info(tk)
    if not info.get("name"):
        info["name"] = scanner.NAMES.get(tk, tk)
    return info


@st.cache_data(ttl=900, show_spinner=False)
def get_full_news(tk):
    info = get_company_info(tk)
    return scanner.full_news(tk, sector=info.get("sector"), industry=info.get("industry"))


@st.cache_data(ttl=86400, show_spinner=False)   # S&P 500 membership barely changes -- 24h is plenty
def get_sp500_list():
    return screener.sp500_constituents()


@st.cache_data(ttl=900, show_spinner=False)
def get_wsb():
    return social.get_wsb_data()


@st.cache_data(ttl=900, show_spinner=False)
def get_reddit_multi():
    return social.get_multi_sub_data()


@st.cache_data(ttl=1800, show_spinner=False)
def get_rm_catalyst(ticker):
    return rm.catalyst_summary(ticker)


def _catalyst_digest(tickers: list[str], days_ahead: int = 7) -> list[dict]:
    """Upcoming EARNINGS within `days_ahead` days across a set of tickers (Watchlist + open
    Paper positions) -- the most common, reliably-dated catalyst type, sorted soonest-first.
    Reuses fundamentals.next_earnings() (already used by Deep Research/Reddit Momentum), no new
    data source or network calls beyond what's already cached elsewhere.

    Deliberately earnings-only: catalyst_summary() also covers biotech trial-completion dates, but
    those don't come with a clean numeric day-count to filter/sort by without building the full
    Deep Research dossier per ticker -- shown on that ticker's own page instead, not folded in here.

    Uses the cached get_earnings() wrapper (the SAME one the Catalysts page and Feed tab already
    use for this exact call), not fundamentals.next_earnings() directly -- calling the uncached
    function meant re-fetching from yfinance on EVERY single rerun of the Watchlist page (found in
    this session's perf audit). An earlier version of this fix added a separate get_next_earnings()
    wrapper instead of reusing get_earnings() -- caught immediately after: two cache namespaces for
    the identical call meant Watchlist and Catalysts never shared a cache entry for the same
    ticker, so consolidated onto the one that already existed."""
    out = []
    for tk in sorted({t.upper() for t in tickers if t}):
        earn = get_earnings(tk)
        if earn and earn.get("days") is not None and 0 <= earn["days"] <= days_ahead:
            out.append({"ticker": tk, "days": earn["days"], "date": earn["date"]})
    return sorted(out, key=lambda d: d["days"])


@st.cache_data(ttl=25, show_spinner=False)
def zd_fetch_intraday():
    return zd.fetch_intraday_batch()


@st.cache_data(ttl=25, show_spinner=False)
def zd_fetch_daily():
    return zd.fetch_daily_batch()


@st.cache_data(ttl=25, show_spinner=False)
def zd_fetch_chain(ticker):
    return zd.nearest_dated_chain(ticker)


@st.cache_data(ttl=14400, show_spinner=False)
def zd_fetch_historical_5m(ticker):
    # 4h TTL, not 25s -- this is a slower-moving reference dataset (up to 60 days of 5m bars) for
    # the time-of-day-adjusted relative-volume baseline, not a live quote. Re-fetching it every
    # 30s refresh would be pointless load for data that's the same all session.
    return zd.fetch_historical_5m(ticker)


@st.cache_data(ttl=30, show_spinner=False)
def pt_fetch_futures():
    # Same 30s-ish cadence as the rest of the live 0DTE fetches (zd_fetch_intraday is 25s) --
    # futures are the freshest-moving input to the Premarket Thesis, so this shouldn't lag behind
    # the other quotes on the page.
    return pt.fetch_futures_batch()


@st.cache_data(ttl=1800, show_spinner=False)
def pt_backtest_tier1(direction, risk_level, gap_bkt, trend_align):
    # 30min TTL -- this re-fetches ~60 days of 5m bars across 5 tickers plus 1y of daily bars,
    # real work that doesn't need to re-run every 30s like the rest of this fragment.
    return pbt.run_tier1_backtest(direction, risk_level, gap_bkt, trend_align)


@st.cache_data(ttl=1800, show_spinner=False)
def pt_backtest_tier2(direction, gap_bkt, trend_align, vix_bkt):
    return pbt.run_tier2_backtest(direction, gap_bkt, trend_align, vix_bkt)


@st.cache_data(ttl=300, show_spinner=False)
def zd_regime_stats(ticker):
    # 5-min TTL (per user request, promoted from a manual on-demand button to an always-visible
    # "headline" table) -- reads the whole Signal Calibration Log via a SQL query + groupby, real
    # but not free work, so this still respects the project's own rule of separating live state
    # (30s refresh) from historical validation (this) rather than recomputing it every fragment
    # rerun. 5 minutes is a reasonable "how current does a statistics table need to be" cadence --
    # nothing about market state itself is read from this, only past-outcome context.
    return zdlog.regime_stats(ticker)


@st.cache_data(ttl=900, show_spinner=False)
def get_lottery(ticker, spot, days=90):
    """A ~2×-expected-move OTM call + put at the expiry nearest `days` — the 'lottery ticket' legs.
    Expected move is recomputed for the CHOSEN expiry (it grows with time). Cached, best-effort."""
    from datetime import date, datetime
    exp = data.nearest_expiry(ticker, days)
    chain = data.get_option_chain(ticker, exp)
    om = det.option_metrics(spot, chain)
    em_pct = om.get("expected_move_straddle_pct") or om.get("expected_move_iv_pct") or 10.0
    em = spot * (em_pct / 100)
    dte = max((datetime.strptime(exp, "%Y-%m-%d").date() - date.today()).days, 1)
    T = dte / 365

    def leg(direction):
        side = chain["calls"] if direction == "call" else chain["puts"]
        target = spot + 2 * em if direction == "call" else spot - 2 * em
        s = side.copy()
        s["_d"] = (s["strike"] - target).abs()
        row = s.loc[s["_d"].idxmin()]
        bidv, askv = row.get("bid"), row.get("ask")
        bid = float(bidv) if bidv == bidv and bidv else 0.0        # NaN / None / 0 -> 0
        ask = float(askv) if askv == askv and askv else 0.0
        mid = det._mid(row)                                        # falls back to lastPrice if no bid/ask
        # You BUY a lottery ticket at the ASK. If there's no live ask (weekend / super-illiquid strike),
        # fall back to the stale last-trade and FLAG it — that's the exact case where our old number
        # ($1.50 mid) disagreed with the broker's real ask ($2.80). Price off the ask so cost isn't rosy.
        stale = ask <= 0                                          # no live offer -> this is a stale print
        buy = ask if ask > 0 else mid
        if not buy or buy != buy or buy <= 0:
            return None
        strike = float(row["strike"])
        be = (strike + buy) if direction == "call" else (strike - buy)
        pop = None
        if em > 0:
            z = (be - spot) / em
            pop = (1 - det._ncdf(z)) if direction == "call" else det._ncdf(z)
        iv = row.get("impliedVolatility")
        iv = float(iv) if iv and iv == iv else 0.5
        return {"option_type": direction, "option_strike": strike, "option_expiry": exp,
                "option_entry_premium": round(buy, 2), "option_iv_pct": round(iv * 100, 1),
                "option_greeks": det.bs_greeks(spot, strike, T, 0.045, iv, call=(direction == "call")),
                "be": round(be, 2), "pct_needed": round(abs(be / spot - 1) * 100, 1),
                "pop": pop, "cost": round(buy * 100, 0), "stale": stale, "has_bid": bid > 0}
    return {"exp": exp, "dte": dte, "em_pct": em_pct, "call": leg("call"), "put": leg("put")}


@st.cache_data(ttl=1800, show_spinner=False)
def get_earnings(ticker):
    return fundamentals.next_earnings(ticker)


@st.cache_data(ttl=1800, show_spinner=False)
def get_dossier(ticker):
    return dr.dossier(ticker)


def _parse_date(s):
    for fmt in ("%Y-%m-%d", "%Y-%m"):
        try:
            return datetime.strptime(s, fmt).date()
        except (ValueError, TypeError):
            continue
    return None


@st.cache_data(ttl=3600 * 4, show_spinner=False)
def get_etf_holdings(ticker: str, n: int = 10) -> list[str]:
    # 4h TTL -- fund holdings drift slowly (daily rebalancing at most), no need to re-fetch
    # every page load the way live quotes do.
    return data.get_etf_top_holdings(ticker, n)


@st.cache_data(ttl=1800, show_spinner=False)
def get_catalyst_rows(tickers: tuple) -> list:
    """One row per dated event (earnings, 8-K, Form 4, trial completion) across `tickers`, for the
    📡 Catalyst Radar tab. `days` is signed: positive = upcoming, negative = already happened."""
    today = date.today()
    rows = []
    for tk in tickers:
        earn = get_earnings(tk)
        if earn:
            rows.append({"ticker": tk, "kind": "Earnings", "date": earn["date"],
                        "days": earn["days"], "url": None})

        filings = sec_edgar.recent_filings(tk, forms=("8-K", "4"), lookback_days=30)
        for f in filings[:5]:
            kind = "8-K filing" if f["form"] == "8-K" else "Form 4 (insider)"
            rows.append({"ticker": tk, "kind": kind, "date": f["filed"],
                        "days": -f["days_ago"], "url": f["url"]})

        info = get_company_info(tk)
        sector, industry = (info.get("sector") or ""), (info.get("industry") or "")
        is_biotech = "Healthcare" in sector or any(
            k in industry.lower() for k in ("biotech", "drug", "pharma"))
        if is_biotech:
            trials = dr.clinical_trials(info.get("name") or tk, limit=5)
            near = [t for t in trials if t.get("primary_completion")]
            if near:
                soonest = min(near, key=lambda t: t["primary_completion"])
                d = _parse_date(soonest["primary_completion"])
                rows.append({"ticker": tk, "kind": "Trial completion",
                            "date": soonest["primary_completion"],
                            "days": (d - today).days if d else None, "url": None})
    rows.sort(key=lambda r: (r["days"] is None, abs(r["days"]) if r["days"] is not None else 0))
    return rows


_CONF_DOT = {"High": "🟢", "Medium": "🟡", "Low": "🟠", "UNKNOWN": "⚪"}


def _col_help(columns):
    """Build a column_config that gives each known column a hover-tooltip (ⓘ) from the glossary."""
    cfg = {}
    for c in columns:
        tip = glossary.help_for(c)
        if tip:
            cfg[c] = st.column_config.Column(c, help=tip)
    return cfg


def render_greeks_card(g: dict):
    """'The Greeks' as a real table (Delta/Gamma/Theta/Vega/Rho as their own columns, each with a
    glossary tooltip) instead of a cramped inline caption string. `g` is a dict from
    `det.bs_greeks()`; missing/empty values show as '—' rather than breaking."""
    def _fmt(v, nd=4):
        return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"

    with st.container(border=True):
        st.markdown("**The Greeks**")
        gc = st.columns(5)
        gc[0].metric("Delta", _fmt(g.get("delta")), help=glossary.help_for("Delta"))
        gc[1].metric("Gamma", _fmt(g.get("gamma"), 6), help=glossary.help_for("Gamma"))
        gc[2].metric("Theta", _fmt(g.get("theta_per_day")), help=glossary.help_for("Theta"))
        gc[3].metric("Vega", _fmt(g.get("vega")), help=glossary.help_for("Vega"))
        gc[4].metric("Rho", _fmt(g.get("rho")), help=glossary.help_for("Rho"))


def _dossier_section_df(fields: dict) -> pd.DataFrame:
    """One row per fact — plain-English 'What it means' is the star column, next to the raw value.
    Values are stringified (the column mixes numbers + text, which Arrow can't serialize as-is)."""
    def _fmt(x):
        if isinstance(x, bool):
            return str(x)
        if isinstance(x, float):
            return f"{x:,.2f}"
        return str(x)
    return pd.DataFrame([{
        "Field": k, "Value": _fmt(v["value"]),
        "What it means": v.get("read", "") or v.get("why", ""),
        "Conf": f"{_CONF_DOT.get(v['conf'], '')} {v['conf']}",
        "Source": v.get("source", ""),
    } for k, v in fields.items()])


@st.fragment
def render_dossier(D: dict, use_expanders: bool = True):
    """Render a Deep Research dossier. use_expanders=True for the standalone tab; False for INSIDE the
    card detail modal (Streamlit forbids nested expanders, so there we use plain headers).

    @st.fragment so the 'Needs your attention' jump buttons (use_expanders=True path only) only
    rerun this dossier's own subtree, not the whole app -- without it, every click triggered
    Streamlit's full-page 'running' fade for what should be an instant local expand, and since the
    target expander lives below the fold, the fade with no visible on-screen change read as
    "nothing happened" (reported by the user). The other two call sites (use_expanders=False,
    inside the trade-drawer dialog and inside Ticker Page's own expander) never hit that button at
    all, so fragment-scoping them too is harmless -- just means their own internal expanders also
    get cheaper reruns."""
    sc = D["scorecard"]
    st.markdown(f"#### Research Completeness: {sc['pct']}%")
    st.progress(sc["pct"] / 100)
    gc = st.columns(3)
    gc[0].markdown("**✓ Strong evidence**\n" + ("".join(f"\n- {a}" for a in sc["strong"]) or "\n- —"))
    gc[1].markdown("**⚠ Weak evidence**\n" + ("".join(f"\n- {a}" for a in sc["weak"]) or "\n- —"))
    gc[2].markdown("**✗ Missing**\n" + ("".join(f"\n- {a}" for a in sc["missing"]) or "\n- —"))
    st.success(f"**Highest-leverage next step →** {sc['next_step']}")
    st.caption("Completeness = how much we could **verify** (not a buy score). The 'What it means' column "
               "explains every number in plain English. Low areas = where YOUR next research pays off most.")

    tk = D.get("ticker", "")
    SECTION_NAMES = ["Executive summary", "Catalysts", "Financial health", "Valuation",
                     "Technicals", "Options", "Ownership / smart money"]

    # Specific WEAK/MISSING fields across every section, not just section-level buckets -- click
    # one and it force-opens that exact section instead of "open each one, scan, hope you spot it."
    gaps = []
    for name in SECTION_NAMES:
        for fname, f in D["sections"].get(name, {}).items():
            if f["conf"] == "UNKNOWN":
                gaps.append((name, fname, "missing"))
            elif f["conf"] == "Low":
                gaps.append((name, fname, "weak"))
    if gaps and use_expanders:
        # Collapsed by default, not a bare markdown header + open list -- user feedback: with a
        # dossier that's mostly UNKNOWN/Low-confidence fields (common right after building one),
        # this list can run 10-20+ items and dominated the top of the page before any of the
        # actual dossier content was visible. Now it's one line to open on demand.
        with st.expander(f"🎯 Needs your attention ({len(gaps)} item{'s' if len(gaps) != 1 else ''})"):
            # Same fix as the 0DTE/Reddit Momentum list-views: plain st.button in a loop stacks
            # each widget's own default block margin into ~1.5rem of dead space per row, and
            # centers text in a big pill instead of reading like a list -- scoped CSS via
            # st.container(key=...)'s auto-generated `st-key-...` class collapses that margin and
            # left-aligns/tightens the buttons so this reads as a compact row list.
            djump_key = f"djump_wrap_{tk}"
            st.markdown(
                f'<style>'
                f'.st-key-{djump_key} {{ background:#0c1414 !important; }}'
                f'.st-key-{djump_key} [data-testid="stElementContainer"] {{ margin-bottom: 0 !important; }}'
                f'.st-key-{djump_key} [data-testid="stButton"] button {{ text-align: left !important; '
                f'justify-content: flex-start !important; min-height: 2.1rem !important; '
                f'padding: 0.4rem 0.8rem !important; border-radius: 0 !important; border: none !important; '
                f'border-top: 1px solid #142020 !important; background: transparent !important; '
                f'width: 100% !important; }}'
                f'.st-key-{djump_key} [data-testid="stButton"]:first-of-type button {{ border-top: none !important; }}'
                f'.st-key-{djump_key} [data-testid="stButton"] button:hover {{ background: #111c1d !important; '
                f'border-color: #142020 !important; color: #e8ecec !important; }}'
                f'</style>', unsafe_allow_html=True)
            with st.container(key=djump_key, border=True):
                for i, (sec, fname, kind) in enumerate(gaps):
                    icon = "🔴" if kind == "missing" else "🟡"
                    if st.button(f"{icon} {sec} → {fname}", key=f"djump_{tk}_{i}"):
                        st.session_state[f"dexp_{tk}_{sec}"] = True
                        st.rerun()   # inside @st.fragment, a bare rerun already scopes to the fragment
        st.divider()

    def _block(title, name, body):
        if use_expanders:
            force_open = st.session_state.pop(f"dexp_{tk}_{name}", False)
            with st.expander(title, expanded=force_open):
                body()
        else:
            st.markdown(f"**{title}**")
            body()

    for name in SECTION_NAMES:
        fields = D["sections"].get(name, {})
        if not fields:
            continue
        known = sum(1 for f in fields.values() if f["conf"] != "UNKNOWN")
        total = len(fields)
        ratio = known / total if total else 0.0
        dot = "🟢" if ratio >= 0.999 else "🟡" if ratio >= 0.5 else "🔴"
        _block(f"{dot} {name}  ·  {known}/{total} verified", name,
               lambda fields=fields: st.dataframe(
                   _dossier_section_df(fields), width="stretch", hide_index=True,
                   column_config=_col_help(["Field", "Value", "What it means", "Conf", "Source"])))

    if D["is_biotech"]:
        def _clin():
            if D["trials"]:
                st.dataframe(pd.DataFrame([{
                    "Phase": t["phase"], "Status": t["status"], "Trial": t["brief"],
                    "Conditions": t["conditions"], "Enroll": t.get("enrollment"),
                    "Primary completion": t.get("primary_completion"), "NCT": t["nct"],
                } for t in D["trials"]]), width="stretch", hide_index=True,
                    column_config=_col_help(["Phase", "Status", "Enroll", "Primary completion", "NCT"]))
                st.caption("**Primary completion ≈ when trial data is collected** — the readout/PR (the real "
                           "catalyst) usually follows within weeks/months. Not an FDA decision date.")
            else:
                st.info("No active interventional trials found for this sponsor name.")
        _block(f"🧬 Clinical trials  ·  {len(D['trials'])} active (ClinicalTrials.gov)", "Clinical trials", _clin)

    _block("❓ Unknowns — what we CAN'T verify for free (exposed, not hidden)", "Unknowns",
           lambda: st.dataframe(pd.DataFrame(D["unknowns"], columns=["Item", "Status", "How to get it"]),
                                width="stretch", hide_index=True))
    st.caption("Phase 2 (opt-in, paid AI) would add a grounded bull thesis, counter-thesis, and "
               "'questions you didn't think to ask' — built ONLY from the verified data above.")


def _jesc(s):
    return str(s).replace("$", "\\$")


def _jbullets(text):
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return "\n".join(f"- {_jesc(ln)}" for ln in lines) if lines else "_none given_"


def _render_open_entry(e: dict, due_ids: set, key_prefix: str = ""):
    """One OPEN journal entry: thesis/falsifiers + the review form + delete. Shared by the 📔
    Journal tab (all tickers) and the 🔭 Ticker Page tab (one ticker) so both stay in sync.
    Key info (ticker, decision, confidence, dates) shows in a colored tile without expanding --
    tile color flags whether a review is due, not a win/loss call. Thesis detail + the review
    form (needs real interactivity, can't be a static tile) sit behind a smaller expander below."""
    is_due = e["id"] in due_ids
    tile = _tile_style("🟡" if is_due else "⚪")
    conf = f"{e.get('confidence')}%" if e.get("confidence") is not None else "?"
    st.markdown(
        f'<div style="background:{tile["bg"]};border:1px solid {tile["border"]};'
        'border-radius:8px;padding:0.7rem 0.9rem;margin-bottom:0.3rem">'
        f'<div style="font-weight:700;font-size:0.92rem">{"⏰ " if is_due else ""}{e["ticker"]} · {e["decision"]}'
        + (f' · {e["direction"]}' if e.get("direction") else "") + '</div>'
        f'<div style="font-size:0.78rem;color:#8b9a9d;margin-top:0.15rem">'
        f'logged {e["created"]} · review {e.get("review_date") or "—"}</div>'
        f'<div style="font-size:0.8rem;margin-top:0.4rem">'
        f'Confidence <b>{conf}</b> · Entry '
        + (f'${e["spot_at_entry"]:.2f}' if e.get("spot_at_entry") else "—") + ' · Target '
        + (f'${e["target"]:.2f}' if e.get("target") else "—") + ' · Max loss '
        + (f'${e["max_loss"]:,.0f}' if e.get("max_loss") else "—") + '</div>'
        '</div>', unsafe_allow_html=True)
    with st.expander("Thesis, falsifiers & review ▸", expanded=is_due):
        st.markdown(f"**Thesis:** {_jesc(e.get('thesis') or '')}")
        st.markdown("**What would prove it wrong:**")
        st.markdown(_jbullets(e.get("falsifiers")))
        if (e.get("assumptions") or "").strip():
            st.markdown("**Assumptions:**")
            st.markdown(_jbullets(e.get("assumptions")))
        meta = " · ".join(x for x in [
            f"Catalyst: {_jesc(e['catalyst'])}" if e.get("catalyst") else "",
            f"Exit: {_jesc(e['exit_plan'])}" if e.get("exit_plan") else "",
            f"Size: {_jesc(e['position_size'])}" if e.get("position_size") else ""] if x)
        if meta:
            st.caption(meta)
        st.divider()
        st.markdown("**Close the loop — review the outcome honestly:**")
        with st.form(f"{key_prefix}review_{e['id']}"):
            r_out = st.text_area("What actually happened?", key=f"{key_prefix}rout_{e['id']}")
            r_ver = st.selectbox("Verdict — separate PROCESS from LUCK", journal.VERDICTS,
                                 key=f"{key_prefix}rver_{e['id']}")
            r_rep = st.radio("Would you make the same decision again?", ["Yes", "No", "Unsure"],
                             horizontal=True, key=f"{key_prefix}rrep_{e['id']}")
            r_les = st.text_input("Lessons learned", key=f"{key_prefix}rles_{e['id']}")
            if st.form_submit_button("✔ Save review", type="primary"):
                journal.review(DB, e["id"], r_out, r_ver, r_rep, r_les)
                st.rerun()
        if st.button("🗑 Delete this decision", key=f"{key_prefix}jdel_{e['id']}"):
            journal.delete(DB, e["id"])
            st.rerun()


# Verdict emoji (journal.VERDICTS) -> tile color, driven by whether the REASONING was right, not
# whether the outcome was good -- matching this journal's own stated philosophy of separating
# process from luck. "Right but unlucky" gets a neutral/blue tile, not red, since the process
# wasn't the problem; "wrong but lucky" gets amber as a caution not to mistake luck for skill.
_VERDICT_TILE = {"✅": "🟢", "🌧️": "🔵", "🍀": "🟡", "❌": "🔴"}


def _verdict_tile_emoji(verdict: str) -> str:
    for prefix, tile_emoji in _VERDICT_TILE.items():
        if verdict.startswith(prefix):
            return tile_emoji
    return "⚪"


def _render_reviewed_entry(e: dict, key_prefix: str = ""):
    verdict = e.get("verdict", "")
    tile = _tile_style(_verdict_tile_emoji(verdict))
    lesson_html = (f'<div style="font-size:0.8rem;margin-top:0.4rem;color:#e8ecec">'
                   f'<b>Lesson:</b> {_jesc(e["lessons"])}</div>' if (e.get("lessons") or "").strip() else "")
    st.markdown(
        f'<div style="background:{tile["bg"]};border:1px solid {tile["border"]};'
        'border-radius:8px;padding:0.7rem 0.9rem;margin-bottom:0.3rem">'
        f'<div style="font-weight:700;font-size:0.92rem">{e["ticker"]} · {e["decision"]}</div>'
        f'<div style="font-size:0.78rem;color:#8b9a9d;margin-top:0.15rem">logged {e["created"]}</div>'
        f'<div style="font-size:0.8rem;margin-top:0.4rem">{verdict}</div>'
        f'<div style="font-size:0.78rem;color:#8b9a9d;margin-top:0.2rem">'
        f'Repeat again? <b>{e.get("repeat_again", "")}</b></div>'
        + lesson_html + '</div>', unsafe_allow_html=True)
    with st.expander("Thesis & outcome ▸"):
        st.markdown(f"**Thesis:** {_jesc(e.get('thesis') or '')}")
        st.markdown(f"**Outcome:** {_jesc(e.get('outcome') or '')}")


def _render_decision_form(heading: str, default_ticker: str | None = None, key_suffix: str = ""):
    """The '+ New decision' form. Ticker is free-text (📔 Journal tab) or locked to one ticker
    (🔭 Ticker Page). Shared so both tabs log through the exact same path into `journal.add`."""
    with st.container(border=True):
        st.subheader(heading)
        with st.form(f"new_decision{key_suffix}", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            if default_ticker:
                c1.text_input("Ticker", value=default_ticker, disabled=True, key=f"jtk{key_suffix}")
                jtk = default_ticker.upper()
            else:
                jtk = c1.text_input("Ticker", key=f"jtk{key_suffix}").strip().upper()
            jdec = c2.selectbox("Decision", ["BUY", "WATCH", "PASS", "SELL"], key=f"jdec{key_suffix}")
            jdir = c3.selectbox("Direction (for scoring)", ["bullish", "bearish", "none"], key=f"jdir{key_suffix}")
            jconf = st.slider("Confidence in the thesis", 0, 100, 50, key=f"jconf{key_suffix}",
                              help="Your honest read, logged BEFORE the outcome. Feeds the Scorecard.")
            jthesis = st.text_area("Thesis — WHY you think this (the case)", key=f"jthesis{key_suffix}")
            jfals = st.text_area("What would prove me WRONG? (one per line — commit to these NOW)",
                                 key=f"jfals{key_suffix}",
                                 help="Pre-committed falsifiers stop 'thesis drift' — quietly changing your reasons later.")
            jassum = st.text_area("Key assumptions this depends on (one per line)", key=f"jassum{key_suffix}")
            c4, c5, c6 = st.columns(3)
            jcat = c4.text_input("Key catalyst / event", key=f"jcat{key_suffix}")
            jhz = c5.selectbox("Horizon (days) for scoring", [7, 30, 90], index=1, key=f"jhz{key_suffix}")
            jrev = c6.date_input("Review on", key=f"jrev{key_suffix}")
            st.markdown("**Exit plan** — decide before you're in it:")
            c7, c8, c9, c10 = st.columns(4)
            jtgt = c7.number_input("Target $", value=0.0, step=0.5, key=f"jtgt{key_suffix}")
            jstop = c8.number_input("Stop $", value=0.0, step=0.5, key=f"jstop{key_suffix}")
            jmax = c9.number_input("Max loss $", value=0.0, step=10.0, key=f"jmax{key_suffix}")
            jsize = c10.text_input("Position size", key=f"jsize{key_suffix}")
            jexit = st.text_input("Exit plan in words (e.g. 'sell before earnings if up 2×')", key=f"jexit{key_suffix}")
            if st.form_submit_button("📔 Log decision", type="primary"):
                if not jtk or not jthesis.strip():
                    st.error("Ticker and a thesis are required.")
                else:
                    try:
                        info = get_company_info(jtk)
                        spot0 = data.last_valid_close(load_prices(jtk))
                    except Exception:
                        info, spot0 = {}, None
                    journal.add(DB, {
                        "ticker": jtk, "company": (info or {}).get("name") or jtk, "decision": jdec,
                        "direction": None if jdir == "none" else jdir, "confidence": jconf,
                        "thesis": jthesis, "falsifiers": jfals, "assumptions": jassum, "catalyst": jcat,
                        "target": jtgt or None, "stop": jstop or None, "max_loss": jmax or None,
                        "position_size": jsize, "exit_plan": jexit, "horizon_days": jhz,
                        "review_date": jrev.isoformat(), "spot_at_entry": spot0})
                    st.session_state["journal_msg"] = (
                        f"Logged decision on {jtk}. "
                        + ("A prediction was added to the 🎯 Scorecard." if jdir != "none"
                           else "No direction → not scored (fine for WATCH/PASS)."))
                    st.rerun()


def lottery_why(c, lp, earn, buzz):
    """Honest 'why a big move might come' reasoning + the data-suggested (coin-flip) direction."""
    reasons = [f"Options are pricing a **±{lp['em_pct']:.0f}% move** by {lp['exp']} — the market "
               "already expects a big swing."]
    if c.get("iv_pct"):
        reasons.append(f"IV is **{c['iv_pct']:.0f}%** — elevated IV means a move is being priced in.")
    if earn and earn.get("days") is not None and 0 <= earn["days"] <= lp["dte"]:
        reasons.append(f"⚡ **Earnings {earn['date']}** ({earn['days']}d out) — a known catalyst inside this window.")
    if buzz:
        chg = f", +{buzz['change_pct']:.0f}% 24h" if buzz.get("change_pct") is not None else ""
        reasons.append(f"🔥 Trending on WSB (#{buzz['rank']}, {buzz['mentions']} mentions{chg}) — attention spiking.")
    reasons.append("_Read the articles below to find the **actual** catalyst — the tool sees that a move "
                   "is priced in, not why._")
    lean = c.get("direction_lean")
    side_txt = {"bullish": "📈 leans **CALL** (up)", "bearish": "📉 leans **PUT** (down)"}.get(
        lean, "🤷 **no clear direction** — market expects a big move *either way*")
    return reasons, side_txt, lean




@st.dialog("Confirm your trade")
def confirm_contract_dialog(ticker, spot, opt, budget):
    """Confirm a SPECIFIC contract (chosen from the chain) before it hits the paper account."""
    per = opt["option_entry_premium"] * 100
    cash = portfolio.get_cash(DB)
    spend = min(budget, cash)
    n = int(spend // per) if per > 0 else 0
    strike = opt["option_strike"]
    be = (strike + opt["option_entry_premium"]) if opt["option_type"] == "call" else (strike - opt["option_entry_premium"])
    g = opt.get("option_greeks", {})
    st.markdown(f"### {ticker} — {opt['option_type'].upper()} \\${strike:g}")
    st.markdown(
        f"- exp {opt['option_expiry']} · premium \\${opt['option_entry_premium']}/sh → **\\${per:,.0f}/contract**\n"
        f"- breakeven **\\${be:.2f}**  ·  IV {opt.get('option_iv_pct')}%\n"
        f"- **{n} contract(s) = \\${n * per:,.0f}** total\n"
        f"- cash \\${cash:,.0f} → **\\${cash - n * per:,.0f}** after"
    )
    render_greeks_card(g)
    st.caption("Paper trade · fills at mid, no commission (real fills are worse).")
    with st.expander("📊 What-if: sell early (est. P&L per contract if the stock moves)"):
        from datetime import date, datetime
        iv_frac = (opt.get("option_iv_pct") or 50) / 100
        dte = max((datetime.strptime(opt["option_expiry"], "%Y-%m-%d").date() - date.today()).days, 1)
        grid = scenario_grid(spot, strike, dte, iv_frac, opt["option_type"] == "call",
                             opt["option_entry_premium"])
        st.dataframe(grid.style.map(_pnl_color).format("${:+,.0f}"))
        st.caption("Rows = stock move · columns = time from now. Est. P&L on **one contract** if you "
                   "**sell then** — you profit on the *move*, you don't have to hold to expiry. "
                   "Watch theta bleed left→right and delta pay on the move. (Black-Scholes, **IV held "
                   "constant** — a real IV drop would make it worse.)")
    if n < 1:
        st.warning(f"One contract costs \\${per:,.0f} — more than your \\${spend:,.0f}.")
        if st.button("Close"):
            st.rerun()
        return
    a, b = st.columns(2)
    if a.button("✅ Confirm paper buy", type="primary", width="stretch"):
        portfolio.buy(DB, ticker, opt["option_type"], strike, opt["option_expiry"],
                      n, opt["option_entry_premium"], spot)
        st.session_state["last_buy_msg"] = (
            f"Bought {n} {ticker} {opt['option_type'].upper()} ${strike:g} for ${n * per:,.0f}. See 💰 Paper.")
        st.rerun()
    if b.button("Cancel", width="stretch"):
        st.rerun()


def render_option_chain(ticker, spot, confirm=True, ctx="research", target_dte=30):
    """Robinhood-style, research-backed options chain you can paper-buy from — with breakeven,
    % to breakeven, cost/contract, and an ODDS flag (🟢 within the expected move / 🔴 moonshot).
    `target_dte` picks which listed expiry opens by default (e.g. 0 for a 0DTE-focused page)."""
    from datetime import date, datetime
    with st.spinner(f"Loading {ticker} option chain…"):
        exps = _cached_expiries(ticker)
    if not exps:
        st.info("No options listed for this ticker.")
        return

    def _dte(e):
        return (datetime.strptime(e, "%Y-%m-%d").date() - date.today()).days

    cc = st.columns([1.2, 2, 1])
    typ = cc[0].radio("Type", ["Call", "Put"], horizontal=True, key=f"ctype_{ticker}_{ctx}")
    default_idx = min(range(len(exps)), key=lambda i: abs(_dte(exps[i]) - target_dte))
    exp = cc[1].selectbox("Expiry", exps, index=default_idx,
                          format_func=lambda e: f"{e}  ({_dte(e)}d)", key=f"cexp_{ticker}_{ctx}")
    budget = cc[2].number_input("Budget ($)", value=int(st.session_state.get("feed_budget", 500)),
                                step=50, min_value=1, key=f"cbud_{ticker}_{ctx}")

    with st.spinner(f"Loading {exp} chain…"):
        chain = _cached_chain(ticker, exp)
    om = det.option_metrics(spot, chain)
    em_pct = om.get("expected_move_straddle_pct") or om.get("expected_move_iv_pct")   # straddle = robust
    em = spot * (em_pct / 100) if em_pct else None
    st.markdown(f"**Share price \\${spot:.2f}**"
                + (f"  ·  expected move to {exp}: ±{em_pct:.1f}% (±\\${em:.2f})" if em else ""))

    side = (chain["calls"] if typ == "Call" else chain["puts"]).copy()
    side = side[(side["strike"] >= spot * 0.75) & (side["strike"] <= spot * 1.35)]
    side = side.sort_values("strike", ascending=(typ == "Put"))

    # Same dark-card/shading/color language as the 0DTE list-view, but keeping every column
    # side-by-side (native st.columns per row) rather than collapsing to click-to-expand rows --
    # comparing Strike/Premium/Breakeven/Odds/P(profit) across many strikes at once is the whole
    # point of a chain, so hiding columns behind a tap would be a functional regression here.
    occ_key = f"occ_{ticker}_{ctx}"
    st.markdown(
        f'<style>'
        f'.st-key-{occ_key} {{ background:#0c1414 !important; }}'
        f'.st-key-{occ_key} [data-testid="stElementContainer"] {{ margin-bottom: 0 !important; }}'
        f'.st-key-{occ_key} [data-testid="stHorizontalBlock"] {{ gap: 0.5rem !important; '
        f'align-items: center !important; padding: 0.3rem 0.3rem; }}'
        f'.st-key-{occ_key} [data-testid="stHorizontalBlock"]:not(:first-of-type) '
        f'{{ border-top: 1px solid #142020; }}'
        f'.st-key-{occ_key} [data-testid="stHorizontalBlock"]:first-of-type '
        f'{{ background:#111c1d; border-radius:4px; }}'
        f'</style>', unsafe_allow_html=True)
    occ_ctx = st.container(key=occ_key, border=True)
    hdr = occ_ctx.columns([0.9, 0.9, 1.1, 0.9, 0.9, 1.3, 0.9, 0.9])
    for i, lab in enumerate(["Strike", "Premium", "Breakeven", "% to B/E", "Cost", "Odds", "P(profit)", "Buy"]):
        tip = glossary.help_for(lab)
        label_html = (f'<span style="color:#87d1ff;font-weight:600;font-size:0.88rem;'
                      f'letter-spacing:0.01em;white-space:nowrap" title="{tip}">{lab}'
                      + (' ⓘ' if tip else '') + '</span>')
        hdr[i].markdown(label_html, unsafe_allow_html=True)

    T = max(_dte(exp), 1) / 365
    # pass 1: compute each row + probability of profit (P underlying is past breakeven at expiry,
    # normal approx with sigma = the expected move). Used to mark the highest-probability contract.
    rows_data = []
    for _, row in side.iterrows():
        prem = det._mid(row)
        if not prem or prem != prem:
            continue
        strike = float(row["strike"])
        intrinsic = max(0.0, (spot - strike) if typ == "Call" else (strike - spot))
        bid = row.get("bid")
        has_bid = bid is not None and bid == bid and float(bid) > 0     # no bid -> can't exit
        stale = prem < intrinsic - 0.005            # mid/last below intrinsic is impossible -> stale quote
        eff = max(prem, intrinsic)                  # you can't buy below intrinsic; real ask is >= intrinsic
        liquid = has_bid and not stale
        per = eff * 100
        be = (strike + eff) if typ == "Call" else (strike - eff)
        req = (be - spot) if typ == "Call" else (spot - be)
        pct_be = (be / spot - 1) * 100
        pop = None
        if em and em > 0 and liquid:                # no probability shown on an untradeable/stale quote
            z = (be - spot) / em
            pop = (1 - det._ncdf(z)) if typ == "Call" else det._ncdf(z)
        rows_data.append({"row": row, "strike": strike, "prem": eff, "per": per, "be": be, "req": req,
                          "pct_be": pct_be, "pop": pop, "liquid": liquid, "stale": stale, "has_bid": has_bid})
    best_pop = max((r["pop"] for r in rows_data if r["pop"] is not None), default=None)
    best_pop_budget = max((r["pop"] for r in rows_data                    # best odds you can actually afford
                           if r["pop"] is not None and r["per"] <= budget), default=None)

    _price_drawn = False
    _prev_above = None
    with occ_ctx:
        for rd in rows_data:
            _above = rd["strike"] >= spot                 # strike at/above the current share price?
            if not _price_drawn and _prev_above is not None and _above != _prev_above:
                st.markdown(                               # Robinhood-style ITM/OTM divider at spot
                    f"<div style='border-top:2px solid #e8a400;text-align:center;color:#e8a400;"
                    f"font-weight:600;font-size:0.8rem;padding-top:2px;margin:1px 0;'>"
                    f"share price &#36;{spot:.2f}</div>", unsafe_allow_html=True)
                _price_drawn = True
            _prev_above = _above
            strike, prem, per = rd["strike"], rd["prem"], rd["per"]
            be, req, pct_be, pop = rd["be"], rd["req"], rd["pct_be"], rd["pop"]
            if em and req <= em:
                dot, word = "🟢", "within move"
            elif em and req <= 2 * em:
                dot, word = "🟡", "stretch"
            else:
                dot, word = "🔴", "moonshot"
            marker = ""
            if rd["liquid"] and pop is not None:
                if best_pop is not None and pop >= best_pop - 1e-9:
                    marker = "⭐"                         # highest odds in the whole list (often over budget)
                elif (best_pop_budget is not None and per <= budget
                      and pop >= best_pop_budget - 1e-9):
                    marker = "<span style='color:#22c55e'>★</span>"   # green star = best odds you can afford
            n = int(min(budget, portfolio.get_cash(DB)) // per) if per > 0 else 0
            rc = st.columns([0.9, 0.9, 1.1, 0.9, 0.9, 1.3, 0.9, 0.9])
            rc[0].markdown(f"**\\${strike:g}**")
            rc[1].write(f"≥\\${prem:.2f}" if rd["stale"] else f"\\${prem:.2f}")
            rc[2].write(f"\\${be:.2f}")
            rc[3].write(f"{pct_be:+.1f}%")
            rc[4].write(f"\\${per:,.0f}")
            rc[5].markdown(f'<span style="color:{_TEXT_BY_EMOJI[dot]}">{dot} <b>{word}</b></span>',
                           unsafe_allow_html=True)
            if not rd["liquid"]:
                rc[6].markdown(f'<span style="color:#8b9a9d">⚠️ {"no bid" if not rd["has_bid"] else "stale"}</span>',
                               unsafe_allow_html=True)
            else:
                pop_color = _score_color(pop * 100, low=15, high=40) if pop is not None else "#8b9a9d"
                rc[6].markdown(f'<span style="color:{pop_color};font-weight:700">'
                               f'{marker} {pop * 100:.0f}%</span>' if pop is not None
                               else '<span style="color:#8b9a9d">—</span>', unsafe_allow_html=True)
            if rc[7].button(f"Buy {n}", key=f"cbuy_{ticker}_{typ}_{strike}_{exp}_{ctx}"):
                row = rd["row"]
                iv = row.get("impliedVolatility")
                iv = float(iv) if iv and iv == iv else ((om.get("atm_iv_pct") or 0) / 100)
                opt = {"option_type": typ.lower(), "option_strike": strike, "option_expiry": exp,
                       "option_entry_premium": round(prem, 2), "option_iv_pct": round(iv * 100, 1),
                       "option_greeks": det.bs_greeks(spot, strike, T, 0.045, iv, call=(typ == "Call"))}
                if confirm:                               # Research page: open a confirm dialog
                    confirm_contract_dialog(ticker, spot, opt, budget)
                elif n < 1:                               # inside the modal: buy directly (no nested dialog)
                    st.warning(f"1 contract costs \\${per:,.0f} — more than your "
                               f"\\${min(budget, portfolio.get_cash(DB)):,.0f}.")
                else:
                    portfolio.buy(DB, ticker, opt["option_type"], strike, exp, n, opt["option_entry_premium"], spot)
                    st.session_state["last_buy_msg"] = (
                        f"Bought {n} {ticker} {typ.upper()} \\${strike:g} for \\${n * per:,.0f}. See 💰 Paper.")
                    st.rerun()
    st.markdown(
        "<div style='color:#888;font-size:0.85em'>"
        "<b>Odds:</b> 🟢 breakeven within ~1 expected move · 🟡 up to 2× · 🔴 moonshot. "
        "<b>P(profit)</b> = est. chance the stock is past breakeven at expiry. "
        "<b>⭐ = highest odds in the whole list</b> (usually deep ITM + expensive — you <i>pay</i> for the "
        "odds, no free lunch). <b><span style='color:#22c55e'>★</span> = best odds you can afford</b> "
        "(highest P(profit) among strikes &le; your budget) — your realistic 'best way in' for this side. "
        "<b>⚠️ no bid</b> = nothing to sell into, you can't exit; <b>⚠️ stale</b> = quoted below intrinsic "
        "value (an untradeable phantom) — both excluded from markers, shown as &ge; a floored premium. "
        "Neither marker is a direction call (that's a coin flip). Premium = mid; real fills at the ask are worse."
        "</div>", unsafe_allow_html=True)


def render_trade_drawer():
    """Global Trade + Positions popovers -- available from every nav section without navigating
    away. Both sit together on the right (Positions, then Trade), both small/content-sized (NOT
    stretched to fill a column -- that's what made them "huge" the first time). Panel width is
    controlled directly via [data-testid="stPopoverBody"] (the panel's own DOM node, confirmed via
    the installed frontend bundle -- decoupled from the trigger button's size entirely), so
    shrinking the buttons doesn't shrink the option chain / positions table back into wrapping.
    Each popover is independent -- both can be open at once, nothing ties their state together.

    Dropped the scroll-linked shrink (animation-timeline: scroll(...)) that shipped earlier --
    user confirmed it didn't animate in their browser. Rather than debug a CSS feature I can't
    watch render, kept only the effects that are guaranteed to work everywhere: idle pulse ring +
    shimmer (ambient, ::after/::before), hover scale/brighten (:hover) -- ordinary CSS, no
    browser-support gamble."""
    st.markdown(
        '<style>'
        '[data-testid="stPopoverBody"] { min-width: 820px !important; width: 820px !important; '
        '  max-width: 95vw !important; }'
        '.st-key-float_btn_row [data-testid="stPopover"] { position:sticky !important; top:0.6rem; z-index:5; }'
        '.st-key-float_btn_row [data-testid="stLinkButton"] { position:sticky !important; top:0.6rem; z-index:5; }'
        # Sponsor sits alone on the far left (deliberately separated from Positions/Trade so it
        # can't be an accidental click while reaching for those), while Positions+Trade stay
        # packed together on the far right. space-between on the OUTER row splits the two groups
        # apart; the more specific .st-key-float_btn_actions rule below (later in this stylesheet,
        # so it wins the cascade tie) overrides back to flex-end just for the inner Positions/
        # Trade pair so THEY stay tight against each other instead of also splitting apart.
        # shrink-to-fit columns (flex:0 0 auto) replace Streamlit's default even column split,
        # which stretched each button across a third of the page width and left dead gaps between
        # them since each button is left-aligned within its own wide column.
        '.st-key-float_btn_row [data-testid="stHorizontalBlock"] { '
        '  display:flex !important; justify-content:space-between !important; gap:0.5rem !important; }'
        '.st-key-float_btn_row [data-testid="stHorizontalBlock"] > div { '
        '  flex:0 0 auto !important; width:auto !important; min-width:0 !important; }'
        '.st-key-float_btn_actions [data-testid="stHorizontalBlock"] { '
        '  display:flex !important; justify-content:flex-end !important; gap:0.5rem !important; }'
        # Same base look for the Sponsor link button as the Positions/Trade popovers -- shares
        # every selector below via a comma list so the three stay pixel-identical, not a
        # separately-maintained near-copy that could silently drift apart later.
        '.st-key-float_btn_row [data-testid="stPopover"] button, '
        '.st-key-float_btn_row [data-testid="stLinkButton"] a {'
        '  border-radius:999px !important; font-weight:700 !important; font-size:0.82rem !important; '
        '  padding:0.35rem 0.9rem !important; width:auto !important; '
        '  background:#12181a !important; border:1px solid #2a3a3c !important; '
        '  box-shadow:0 2px 6px rgba(0,0,0,0.35) !important; position:relative; overflow:hidden; '
        '  transition:transform 150ms ease, border-color 150ms ease; }'
        '.st-key-float_btn_row [data-testid="stPopover"] button:hover, '
        '.st-key-float_btn_row [data-testid="stLinkButton"] a:hover {'
        '  transform:scale(1.06) !important; border-color:#7cc4ea !important; }'
        # idle pulse ring
        '.st-key-float_btn_row [data-testid="stPopover"] button::after, '
        '.st-key-float_btn_row [data-testid="stLinkButton"] a::after {'
        '  content:""; position:absolute; inset:0; border-radius:999px; border:1.5px solid #7cc4ea; '
        '  opacity:0; animation:floatBtnPulse 2.6s ease-out infinite; pointer-events:none; }'
        # idle shimmer sweep
        '.st-key-float_btn_row [data-testid="stPopover"] button::before, '
        '.st-key-float_btn_row [data-testid="stLinkButton"] a::before {'
        '  content:""; position:absolute; top:0; left:-60%; width:40%; height:100%; pointer-events:none; '
        '  background:linear-gradient(100deg, transparent, rgba(124,196,234,0.28), transparent); '
        '  animation:floatBtnShimmer 3.6s ease-in-out infinite; }'
        '.st-key-float_btn_row [data-testid="stPopover"] button:hover::after, '
        '.st-key-float_btn_row [data-testid="stPopover"] button:hover::before, '
        '.st-key-float_btn_row [data-testid="stLinkButton"] a:hover::after, '
        '.st-key-float_btn_row [data-testid="stLinkButton"] a:hover::before { animation-play-state:paused; opacity:0; }'
        '@keyframes floatBtnPulse { 0% { transform:scale(1); opacity:0.5; } '
        '  70% { transform:scale(1.35); opacity:0; } 100% { transform:scale(1.35); opacity:0; } }'
        '@keyframes floatBtnShimmer { 0% { left:-60%; } 45% { left:130%; } 100% { left:130%; } }'
        '@media (prefers-reduced-motion: reduce) {'
        '  .st-key-float_btn_row [data-testid="stPopover"] button, '
        '  .st-key-float_btn_row [data-testid="stPopover"] button::after, '
        '  .st-key-float_btn_row [data-testid="stPopover"] button::before, '
        '  .st-key-float_btn_row [data-testid="stLinkButton"] a, '
        '  .st-key-float_btn_row [data-testid="stLinkButton"] a::after, '
        '  .st-key-float_btn_row [data-testid="stLinkButton"] a::before { animation:none !important; } }'
        '</style>', unsafe_allow_html=True)

    with st.container(key="float_btn_row"):
        sponsor_col, actions_col = st.columns(2)
        with sponsor_col:
            st.link_button("💖 Sponsor", "https://github.com/sponsors/jacksonp-dev")
        with actions_col, st.container(key="float_btn_actions"):
            pos_col, trade_col = st.columns(2)
            with pos_col:
                live_n = len(portfolio.open_positions(DB))
                with st.popover(f"📋 Positions ({live_n})" if live_n else "📋 Positions"):
                    st.markdown("**Open positions**")
                    try:
                        s = _fetch_paper_summary()
                        _render_positions_block(s, "drawer")
                    except Exception as e:
                        st.error(f"Error: {e}")
            with trade_col:
                with st.popover("📊 Trade"):
                    st.markdown("**Quick paper trade**")
                    on_zero_dte = nav == "0DTE Intelligence"
                    default_tk = (st.session_state.get("zd_ticker") if on_zero_dte
                                 else st.session_state.get("drawer_ticker")) or "SPY"
                    dtk = st.text_input("Ticker", default_tk, key="drawer_tk_input").strip().upper()
                    if dtk:
                        st.session_state["drawer_ticker"] = dtk
                        try:
                            with st.spinner(f"Loading {dtk}…"):
                                dspot = data.last_valid_close(load_prices(dtk))
                            if dspot is None:
                                raise ValueError("no valid close price available")
                        except Exception as e:
                            st.error(f"Couldn't load {dtk}: {e}")
                        else:
                            render_option_chain(dtk, dspot, confirm=True, ctx="drawer",
                                                target_dte=0 if on_zero_dte else 30)


@st.cache_data(ttl=900, show_spinner=False)
def get_assessment(ticker):
    return assess.full(ticker)


@st.dialog("Stock detail", width="large")
def stock_detail_dialog(ticker):
    """Full drill-in: color-coded state flags (incl. earnings + analysts), news, Reddit, and a
    buyable options chain — all in one modal."""
    try:
        with st.spinner(f"Loading {ticker}…"):
            a = get_assessment(ticker)
    except Exception as e:
        st.error(f"Couldn't load {ticker}: {e}")
        return
    spot, om = a["spot"], a["om"]
    st.markdown(f"## {ticker} — {get_company_info(ticker).get('name') or ticker}")
    _emh = om.get("expected_move_straddle_pct") or om.get("expected_move_iv_pct")   # straddle-first, matches feed
    st.caption(f"${spot:,.2f} · expected move ±{_emh}% (~30d) · IV {om.get('atm_iv_pct')}% · "
               f"put/call OI {om.get('put_call_oi_ratio')}")
    lean = a.get("direction", "neutral")
    r = a.get("reasoning", {})
    sug = {"bullish": "📈 CALL (up)", "bearish": "📉 PUT (down)"}.get(lean, "🤷 no clear direction")
    bull, bear = r.get("bull_factors", []), r.get("bear_factors", [])
    because = f"for: {', '.join(bull)}" if lean == "bullish" else (
        f"for: {', '.join(bear)}" if lean == "bearish" else "signals are mixed")
    st.info(f"**Data-suggested play: {sug}**  ·  conf {a.get('confidence')}  \n"
            f"{because}  \n_⚠️ ~a coin flip and ONE input — weigh the flags below and decide yourself._")
    flags = list(a["flags"].items())
    for i in range(0, len(flags), 3):
        cols = st.columns(3)
        for j, (name, fl) in enumerate(flags[i:i + 3]):
            with cols[j].container(border=True):
                st.markdown(f"**{fl['color']} {name}** — {fl['label']}")
                st.caption(fl["detail"])
    buzz = social.buzz_for(ticker, get_wsb())
    if buzz:
        chg = f", {buzz['change_pct']:+.0f}% 24h" if buzz["change_pct"] is not None else ""
        st.markdown(f"🔥 **WSB #{buzz['rank']}** — {buzz['mentions']} mentions{chg}  _(context, often contrarian)_")
    with st.expander("📰 News"):
        news = get_full_news(ticker)
        for nsi in news[:8]:
            meta = " · ".join(x for x in (nsi.get("publisher", ""), nsi.get("date", "")) if x)
            st.markdown(f"• [{nsi['title']}]({nsi['link']})  \n*{meta}*")
        if not news:
            st.write("No articles found.")
    st.markdown("### 🧮 Options chain — pick a strike & paper-buy")
    render_option_chain(ticker, spot, confirm=False, ctx="modal")

    # ── Deep Research, on the fly (loaded on demand; cached; no nested expanders in a dialog) ──
    st.markdown("---")
    if st.button("🔬 Load Deep Research dossier", key=f"loaddeep_{ticker}"):
        st.session_state[f"deep_in_modal_{ticker}"] = True
    if st.session_state.get(f"deep_in_modal_{ticker}"):
        st.markdown("### 🔬 Deep Research — data-first, plain-English, reduces uncertainty")
        try:
            with st.spinner(f"Researching {ticker} (Yahoo, ClinicalTrials.gov)…"):
                Dm = get_dossier(ticker)
            render_dossier(Dm, use_expanders=False)
        except Exception as e:
            st.error(f"Deep research failed: {e}")


st.markdown("""
<style>
  /* ── readability polish (dark-friendly, conservative) ── */
  .block-container { padding-top: 2.2rem; }   /* keep layout=wide full width for tables */
  [data-testid="stMarkdownContainer"] p,
  [data-testid="stMarkdownContainer"] li { line-height: 1.6; font-size: 0.96rem; }
  /* brighter, roomier captions (the old dim gray was the main 'hard to read' culprit) */
  [data-testid="stCaptionContainer"] p,
  [data-testid="stCaptionContainer"] { color: #9fb0c3 !important; font-size: 0.86rem; line-height: 1.5; }
  /* sidebar nav group headers (PIIP Navigation / RESEARCH / PORTFOLIO / TOOLS) — accent + bold */
  [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
  [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #87d1ff !important; font-weight: 700; letter-spacing: 0.03em;
  }
  h1, h2, h3, h4 { letter-spacing: -0.01em; margin-top: 0.5rem; }   /* section headers stay white */
  /* metric LABELS (ATM IV, Exp move, Put/Call OI, Expected move…) get the accent color; values stay white.
     !important + child <p> so Streamlit's more-specific theme CSS doesn't override it. */
  [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p, [data-testid="stMetricLabel"] * {
    color: #87d1ff !important; opacity: 1; font-weight: 600;
  }
  [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; letter-spacing: -0.01em; }
  [data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
  /* bordered containers (feed cards, glossary) get a touch of polish */
  [data-testid="stVerticalBlockBorderWrapper"] { border-radius: 12px; }
  /* no leftover top gap above the sidebar logo from the sidebar's first-element padding */
  [data-testid="stSidebar"] > div:first-child { padding-top: 0.75rem; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<style>'
    '.piip-cursor { display:inline-block; width:0.55rem; height:1.1rem; background:#7cc4ea; '
    'margin-left:0.15rem; vertical-align:-0.15rem; animation:piipCursorBlink 1.1s steps(2) infinite; }'
    '@keyframes piipCursorBlink { 50% { opacity:0; } }'
    '@media (prefers-reduced-motion: reduce) { .piip-cursor { animation:none !important; } }'
    '</style>'
    '<div style="font-family:ui-monospace,Consolas,monospace;font-size:1.5rem;font-weight:700;'
    'color:#e8ecec;letter-spacing:-0.01em;margin:0.3rem 0 0.4rem">'
    '<span style="color:#79ed8e">&gt;</span> Personal Investment Intelligence Platform'
    '<span class="piip-cursor"></span></div>', unsafe_allow_html=True)
st.caption("Calibrated forecasting for US equities & options — **research & education only, not advice.** "
           "The deterministic engine *computes*; the LLM *interprets* and is on trial vs the baseline. "
           "Today's honest baseline is ~a coin flip — this tool tells you the truth, it doesn't flatter.")

if st.session_state.get("last_buy_msg"):
    st.success(st.session_state.pop("last_buy_msg"))

NAV_GROUPS = [
    ("", [("🏠", "Home")]),
    ("Research", [("⭐", "Watchlist"), ("🧮", "Screener"), ("📰", "Feed"), ("📣", "Reddit Momentum"),
                  ("📟", "0DTE Intelligence"), ("🎰", "Lottery"), ("🔎", "Research"),
                  ("🔬", "Deep Research"), ("🔭", "Ticker Page"), ("📡", "Catalysts")]),
    ("Portfolio", [("💰", "Paper"), ("📔", "Journal"), ("🎯", "Scorecard")]),
    ("Tools", [("⏳", "Backtest"), ("📖", "Glossary"), ("🐛", "Feedback")]),
]
st.session_state.setdefault("nav", "Home")
with st.sidebar:
    _logo_b64 = base64.b64encode(FAVICON.read_bytes()).decode()
    st.markdown(
        f'<div style="text-align:center"><img src="data:image/png;base64,{_logo_b64}" '
        f'width="100" style="display:inline-block"></div>',
        unsafe_allow_html=True)
    st.caption("PIIP Navigation")
    for gi, (group_name, items) in enumerate(NAV_GROUPS):
        if group_name:
            st.caption(group_name)
        for icon, label in items:
            active = st.session_state.nav == label
            if st.button(f"{icon}  {label}", key=f"nav_{label}", width="stretch",
                         type="primary" if active else "secondary"):
                st.session_state.nav = label
                st.rerun()
        if gi == 0:
            # Global ticker search -- every other page makes you navigate to a specific tab and
            # re-type the ticker; this jumps straight to Ticker Page from anywhere. Collapsed by
            # default: a REAL secondary nav st.button (not a styled lookalike), so it's a
            # guaranteed exact match to Watchlist/Feed/etc. around it -- no color-approximation
            # needed for that state at all. Clicking it swaps in the actual input row (styled to
            # match that same secondary-button color family) -- the one thing this can't do that
            # the design mockup could: no auto-focus on expand, since Streamlit can't run injected
            # JS (st.markdown-injected <script> tags don't execute), so you still have to click
            # into the box once it appears.
            st.session_state.setdefault("search_expanded", False)
            if not st.session_state["search_expanded"]:
                if st.button("🔎  Jump to ticker", key="search_trigger", width="stretch"):
                    st.session_state["search_expanded"] = True
                    st.rerun()
            else:
                # Plain `button` tag selector, not [data-testid="stButton"] -- that testid turned
                # out not to match a form_submit_button's actual DOM (confirmed by the CSS visibly
                # not applying: the arrow button kept its own default box/border in the live app).
                # Scoped to .st-key-global_search_wrap, so this can't leak out and restyle buttons
                # anywhere else on the page. Also dropped the separate "✕" collapse button from the
                # last version -- wasn't part of the mockup the user actually approved; jumping
                # already auto-collapses, and adding a second, unapproved control was the other
                # half of why this didn't match what was shown.
                st.markdown(
                    '<style>'
                    # Several rounds of guessing exact hex values to match the nav buttons failed
                    # (wrong shade, then a focus effect that grew/stuck) because I can't see the
                    # actual render. Stopped guessing colors entirely: no custom background/border
                    # on the input or its wrapper anymore, just Streamlit's own native dark-theme
                    # styling for a text_input, which draws from the SAME theme as the secondary
                    # buttons around it -- letting the two stay consistent by construction instead
                    # of by manual color-matching. Only kept: hiding the redundant "Press Enter to
                    # submit form" hint, and tightening the row's own layout spacing/gap.
                    '.st-key-global_search_wrap [data-testid="InputInstructions"] { display:none !important; }'
                    '.st-key-global_search_wrap small { display:none !important; }'
                    '.st-key-global_search_wrap [data-testid="stHorizontalBlock"] { '
                    '  gap:0.3rem !important; align-items:center !important; }'
                    '.st-key-global_search_wrap [data-testid="stElementContainer"] { margin-bottom:0 !important; }'
                    # The extra height causing the page to shift down on expand: st.form(border=False)
                    # hides the visible border but Streamlit still reserves the SAME internal padding
                    # a bordered form would have (so toggling the parameter alone doesn't reflow other
                    # things) -- that reserved padding is what was adding the height, not the input
                    # itself. Zero it out specifically, still no color/background touched.
                    '.st-key-global_search_wrap [data-testid="stForm"] { '
                    '  padding:0 !important; margin:0 !important; gap:0 !important; }'
                    '.st-key-global_search_wrap [data-testid="stVerticalBlock"] { gap:0 !important; }'
                    '.st-key-global_search_wrap { margin-bottom:0.5rem; }'
                    '</style>', unsafe_allow_html=True)
                with st.container(key="global_search_wrap"):
                    with st.form("global_search_form", clear_on_submit=True, border=False):
                        jc1, jc2 = st.columns([6, 1])
                        jump_tk = jc1.text_input("Jump to ticker", key="global_search",
                                                 placeholder="🔎 Type a ticker…", label_visibility="collapsed")
                        jump_go = jc2.form_submit_button("→")
                if jump_go and jump_tk.strip():
                    st.session_state["tp_free"] = jump_tk.strip().upper()
                    st.session_state.nav = "Ticker Page"
                    st.session_state["search_expanded"] = False
                    st.rerun()

    # Update notice (PIIP audit 2026-08): sidebar footer, visible on every page. Checks GitHub's
    # raw-content CDN for a newer VERSION file, cached 12h so this never hits the network on every
    # rerun -- any failure (offline, GitHub down, repo not public) is swallowed silently inside
    # check_for_update(), never blocking or erroring the app. Shows the current version either way
    # so there's always an easy answer to "what version am I even running."
    st.divider()
    _update_info = check_for_piip_update()
    if _update_info and _update_info["update_available"]:
        st.success(f"🎉 Update available: v{_update_info['remote_version']} "
                  f"(you're on v{_update_info['local_version']})")
        st.link_button("⬇️ Get the latest version", f"https://github.com/{GITHUB_REPO}",
                       width="stretch")
        if _update_info.get("changelog"):
            with st.expander("📋 What's new"):
                st.markdown(_update_info["changelog"])
    else:
        st.caption(f"PIIP v{update_check.local_version()}")
nav = st.session_state.nav
if "prev_last_visit" not in st.session_state:
    # Once per browser SESSION, not per rerun -- otherwise "since last visit" would read back as
    # "a few seconds ago" on every single click instead of spanning real time away from the app.
    try:
        st.session_state["prev_last_visit"] = appstate.get_and_bump_last_visit(DB)
    except Exception:
        st.session_state["prev_last_visit"] = None
try:
    _early_summary = _fetch_paper_summary()   # cached -- reuses whatever's already fresh this rerun
    _expired = portfolio.close_expired(DB, _early_summary["positions"])
    if _expired:
        # position count changed -- _fetch_paper_summary()'s own staleness check (n_open mismatch)
        # will re-mark on its next call, so no manual "stale" flag needed here.
        names = ", ".join(f"{p['ticker']} {p['opt_type'].upper()} ${p['strike']:g}" for p in _expired)
        st.toast(f"⏰ Auto-closed {len(_expired)} expired position(s): {names}")
        _early_summary = _fetch_paper_summary()   # refresh once so log_snapshot logs the post-close mark
    portfolio.log_snapshot(DB, current_equity=_early_summary["equity"], current_cash=_early_summary["cash"])
except Exception:
    pass   # best-effort background telemetry — a transient data-fetch failure must never crash the page
render_trade_drawer()        # global "🧮 Trade" popover — available on every page, not just one tab

@st.fragment
def _render_equity_chart(curve: list[dict]):
    """Isolated so picking a different Range doesn't trigger a full-app rerun -- re-slicing an
    already-fetched `curve` list is cheap, but without a fragment every click still reran the
    ENTIRE script (nav, both popovers, this page's other sections), which is what actually made
    switching ranges feel like a slow reload."""
    RANGES = {"1D": 1, "1W": 7, "1M": 30, "3M": 90,
              "YTD": (date.today() - date(date.today().year, 1, 1)).days + 1,
              "1Y": 365, "ALL": None}
    rng = st.radio("Range", list(RANGES.keys()), index=6, horizontal=True,
                   key="home_chart_range", label_visibility="collapsed")
    days = RANGES[rng]
    shown = curve if days is None else [
        p for p in curve if (date.today() - date.fromisoformat(p["date"])).days < days]
    if rng == "1D":
        st.caption("Intraday tracking isn't wired up yet — this shows only today's single "
                   "end-of-day-so-far point, not a live intraday curve. **1W or wider is the "
                   "honest view right now.**")
    if len(shown) >= 2:
        # PIIP audit 2026-08: rebuilt on TradingView's Lightweight Charts (see
        # LIGHTWEIGHT_CHARTS_JS / _load_lightweight_charts_js(), first used for the 0DTE intraday
        # chart) instead of Altair. This chart was ALREADY hand-building a TradingView look
        # (right-axis price scale, dashed rule + floating tag pinned at the last value) out of 4
        # layered Altair marks -- lastValueVisible + priceLineVisible are native, built-in
        # features of the actual TradingView library, so this is a straightforward, better-
        # fitting swap, not just consistency for its own sake. Colored by net direction over the
        # SHOWN window, same green/red semantics as the Watchlist/Screener sparklines (those stay
        # Altair -- tiny inline table-cell charts, a full JS chart per row isn't a good trade).
        color = "#79ed8e" if shown[-1]["equity"] >= shown[0]["equity"] else "#ff8080"
        points = [{"time": {"year": int(p["date"][:4]), "month": int(p["date"][5:7]),
                            "day": int(p["date"][8:10])}, "value": round(float(p["equity"]), 2)}
                 for p in shown]
        chart_id = f"home_equity_{rng}".replace(" ", "_")
        payload = json.dumps({"points": points, "color": color})
        html = f"""
<div id="{chart_id}" style="width:100%;height:280px;"></div>
<script>{_load_lightweight_charts_js()}</script>
<script>
(function() {{
  const data = {payload};
  const container = document.getElementById("{chart_id}");
  const chart = LightweightCharts.createChart(container, {{
    autoSize: true,
    layout: {{ background: {{ type: "solid", color: "transparent" }}, textColor: "#8b9a9d" }},
    grid: {{ vertLines: {{ color: "#1c2426" }}, horzLines: {{ color: "#1c2426" }} }},
    rightPriceScale: {{ borderColor: "#232b2d" }},
    timeScale: {{ borderColor: "#232b2d" }},
    crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
  }});
  const series = chart.addSeries(LightweightCharts.LineSeries, {{
    color: data.color, lineWidth: 2,
    lastValueVisible: true, priceLineVisible: true, priceLineStyle: 2,
    priceFormat: {{ type: "price", precision: 2, minMove: 0.01 }},
  }});
  series.setData(data.points);
  chart.timeScale().fitContent();
}})();
</script>
"""
        st.iframe(html, height=290)
    else:
        st.caption(f"Only one data point in the {rng} window — try a wider range.")
    st.caption("Historical points before daily snapshot-logging began are reconstructed from "
               "your actual recorded trades (start balance + realized P&L at each exit) — real "
               "data, but it excludes unrealized swings of positions that were still open at "
               "each past moment. Today's point is always live full equity.")


# ─────────────────────────── Home ───────────────────────────
if nav == "Home":
    # _fetch_paper_summary() is the SAME cache render_trade_drawer()'s Positions popover already
    # primed above -- portfolio.summary() does a live network round-trip PER open position
    # (current_premium() -> get_spot()/get_option_chain()), so calling it again here (and AGAIN
    # inside the old uncached day_pnl()/equity_curve()) meant every Home render -- including
    # switching the chart's Range radio -- did 3-4x the necessary network calls.
    psum = _fetch_paper_summary()
    dpnl = portfolio.day_pnl(DB, current_equity=psum["equity"])

    # "Since your last visit" digest -- prev_last_visit is None on the very first-ever visit (no
    # prior timestamp to compare against), and is only set ONCE per browser session (see the
    # top-level gate above), so this reads as "since you were last here," not "since your last click."
    prev_visit = st.session_state.get("prev_last_visit")
    if prev_visit:
        try:
            prev_dt = datetime.fromisoformat(prev_visit)
            elapsed = datetime.now(timezone.utc) - prev_dt
            if elapsed.days >= 1:
                elapsed_str = f"{elapsed.days}d ago"
            elif elapsed.seconds >= 3600:
                elapsed_str = f"{elapsed.seconds // 3600}h ago"
            else:
                elapsed_str = f"{max(1, elapsed.seconds // 60)}m ago"
            curve_since = portfolio.equity_curve(DB, current_equity=psum["equity"])
            prior_points = [p for p in curve_since if p["date"] <= prev_dt.date().isoformat()]
            eq_then = prior_points[-1]["equity"] if prior_points else None
            graded = pred.graded_since(prev_visit, DB)
            bits = []
            if eq_then is not None:
                delta = psum["equity"] - eq_then
                arrow = "▲" if delta >= 0 else "▼"
                bits.append(f"{arrow} ${abs(delta):,.2f} equity change")
            if graded:
                hits = sum(1 for g in graded if g["hit"])
                bits.append(f"{len(graded)} Scorecard prediction(s) graded ({hits} correct)")
            if bits:
                st.info(f"**Since your last visit ({elapsed_str}):** " + " · ".join(bits))
        except Exception:
            pass

    _render_kpi_tiles(psum, dpnl)
    curve = portfolio.equity_curve(DB, current_equity=psum["equity"])
    if len(curve) >= 2:
        _render_equity_chart(curve)
    else:
        st.caption("📈 The equity chart fills in as daily snapshots accumulate — check back "
                   "after a few days of paper trading.")

    st.subheader("Open positions")
    if not psum["positions"]:
        st.write("None yet — open one from 📰 Feed, 🔎 Research, or 🎰 Lottery.")
    else:
        def _home_dte(exp):
            try:
                return (date.fromisoformat(exp) - date.today()).days
            except Exception:
                return None

        hdf = pd.DataFrame([{
            "Contract": f"{p['ticker']} {p['opt_type'].upper()} ${p['strike']:g}",
            "Expiry": p["expiry"] + (f"  ({d}d)" if (d := _home_dte(p["expiry"])) is not None else ""),
            "Qty": p["contracts"],
            "Entry": p["entry_premium"],
            "Now": p["current_premium"],
            "P&L $": p["unrealized_pnl"],
            "P&L %": p["unrealized_pct"],
        } for p in psum["positions"]])
        # Reuses the SAME _pnl_bar_style/_signed_money/_signed_pct helpers as the Paper tab's and
        # Positions popover's open-positions table (_render_positions_block above), not a separate
        # hand-rolled color scheme -- this table used to run its own local _home_pnl_color (plain
        # text color, different from the shared table's gradient bar) which had drifted into a
        # visibly different look for the exact same P&L data shown elsewhere in the app.
        max_pnl_dollar_h = _safe_abs_max(hdf["P&L $"])
        max_pnl_pct_h = _safe_abs_max(hdf["P&L %"])
        styled_h = (hdf.style
                    .apply(lambda col: [_pnl_bar_style(v, max_pnl_dollar_h) for v in col], subset=["P&L $"])
                    .apply(lambda col: [_pnl_bar_style(v, max_pnl_pct_h) for v in col], subset=["P&L %"])
                    .format({"Entry": "${:.2f}", "Now": "${:.2f}",
                             "P&L $": _signed_money, "P&L %": _signed_pct}))
        # width="stretch" (not "content") -- with only 7 columns below a full-width tile grid and
        # chart, "content" left a large empty gap next to the table instead of matching the rest
        # of the page (confirmed from a live screenshot). Explicit per-column widths -- "medium"
        # for the two text columns, "small" for the numeric ones -- so the fill concentrates into
        # Contract/Expiry instead of evenly padding every column including 1-2 digit Qty/P&L cells,
        # which is what made Watchlist/Screener look bad when they were left on plain "stretch".
        home_col_cfg = {c: st.column_config.Column(c, help=glossary.help_for(c),
                        width="medium" if c in ("Contract", "Expiry") else "small") for c in hdf.columns}
        st.dataframe(styled_h, width="stretch", hide_index=True,
                    height=(len(hdf) + 1) * 35 + 3, column_config=home_col_cfg)
        st.caption("Read-only snapshot — manage/close positions on the 💰 Paper page.")

@st.fragment
def _render_watchlist_table(watched: list[dict]):
    """The dense sparkline table + row-selection Open/Remove controls (Option A from
    watchlist_redesign_options.html). @st.fragment for the same reason as _render_positions_block
    above: st.dataframe's on_select="rerun" would otherwise trigger a full-app rerun (nav, popovers,
    everything) on every row click, for what should be an instant local selection. Add/nav actions
    (form submit, Open, Remove) still call plain st.rerun() deliberately -- those change the
    underlying watchlist or navigate away, so they need a real full-app rerun, not a fragment-scoped
    one that would leave stale state elsewhere on the page."""
    rmulti = get_reddit_multi()
    trend_by_ticker = {t["ticker"]: t for t in social.reddit_trending_multi(9999, data=rmulti)}
    rows = []
    for w in watched:
        tk = w["ticker"]
        try:
            info = get_company_info(tk)
            valid_close = load_prices(tk)["Close"].dropna()
            spot = float(valid_close.iloc[-1])
            day_chg = ((spot / float(valid_close.iloc[-2]) - 1) * 100
                      if len(valid_close) >= 2 else None)
            spark = [float(v) for v in valid_close.tail(20).tolist()]
        except Exception:
            info, spot, day_chg, spark = {}, None, None, []
        cat = get_rm_catalyst(tk)
        tbuzz = trend_by_ticker.get(tk)
        rows.append({
            "Symbol": tk,
            "Company": info.get("name", tk),
            "Chart": spark,
            "Price": spot,
            "1D %": day_chg,
            "Catalyst": cat.get("next_catalyst") or "No known dated catalyst found",
            "Reddit": (f"{tbuzz['total_mentions']} mentions / {len(tbuzz['subs'])} sub(s)"
                      if tbuzz else "No Reddit buzz found"),
        })
    wdf = pd.DataFrame(rows)
    max_abs_chg = _safe_abs_max(wdf["1D %"])
    styled_w = (wdf.style
                .apply(lambda col: [_pnl_bar_style(v, max_abs_chg) for v in col], subset=["1D %"])
                .format({"Price": lambda v: f"${v:,.2f}" if v is not None and v == v else "—",
                         "1D %": lambda v: f"{v:+.1f}%" if v is not None and v == v else "—"}))
    # width="stretch" + explicit per-column widths, not "content" -- same "sad open space" fix
    # already applied to Home/Paper's tables: "content" left a large empty gap beside the table
    # instead of filling the page, and plain "stretch" (no column_config) pads every column out
    # evenly including short Symbol/Price/1D % cells. "large" for Catalyst (longest text), "medium"
    # for Company/Reddit, "small" for the rest.
    wl_col_cfg = {
        "Symbol": st.column_config.Column("Symbol", width="small", help=glossary.help_for("Symbol")),
        "Company": st.column_config.Column("Company", width="medium", help=glossary.help_for("Company")),
        "Chart": st.column_config.LineChartColumn("Chart", width="small", color="auto",
                                                   help=glossary.help_for("Chart")),
        "Price": st.column_config.Column("Price", width="small", help=glossary.help_for("Price")),
        "1D %": st.column_config.Column("1D %", width="small", help=glossary.help_for("1D %")),
        "Catalyst": st.column_config.Column("Catalyst", width="large", help=glossary.help_for("Catalyst")),
        "Reddit": st.column_config.Column("Reddit", width="medium", help=glossary.help_for("Reddit")),
    }
    # Explicit height sized to every row -- same fix as the Screener table: without it Streamlit
    # caps the box to a fixed default height and scrolls internally even with room to spare below.
    event = st.dataframe(styled_w, width="stretch", hide_index=True,
                         column_config=wl_col_cfg, height=(len(rows) + 1) * 35 + 3,
                         on_select="rerun", selection_mode="single-row", key="wl_table")
    sel = list(event.selection.rows) if getattr(event, "selection", None) else []
    sel_tk = watched[sel[0]]["ticker"] if sel else None
    wa1, wa2, wa3 = st.columns([1, 1, 4])
    if wa1.button("🔭 Open", disabled=not sel_tk, key="wl_open_sel"):
        st.session_state["tp_free"] = sel_tk
        st.session_state.nav = "Ticker Page"
        st.rerun()
    if wa2.button("✕ Remove", disabled=not sel_tk, key="wl_remove_sel"):
        wl.remove(DB, sel_tk)
        st.rerun()
    wa3.caption("Select a row (click anywhere in it), then Open or Remove.")


# ─────────────────────────── Watchlist ───────────────────────────
if nav == "Watchlist":
    _info_header("⭐ Watchlist", (
        "Tickers you're tracking but don't (necessarily) own. Free data only — price/day "
        "change (Yahoo Finance), next catalyst (same lookup Reddit Momentum uses), Reddit "
        "buzz (ApeWisdom, all 6 tracked subreddits) — nothing paid, nothing fabricated."),
        size="1.3rem")
    with st.form("wl_add_form", clear_on_submit=True):
        ac1, ac2 = st.columns([4, 1])
        add_tk = ac1.text_input("Add ticker", placeholder="Add a ticker (e.g. AAPL)",
                                label_visibility="collapsed", key="wl_add_input")
        add_go = ac2.form_submit_button("+ Add", type="primary")
    if add_go and add_tk.strip():
        if wl.add(DB, add_tk):
            st.toast(f"Added {add_tk.strip().upper()} to your watchlist")
        else:
            st.toast(f"{add_tk.strip().upper()} is already on your watchlist")

    watched = wl.list_tickers(DB)
    watch_tickers = [w["ticker"] for w in watched]
    try:
        open_position_tickers = [p["ticker"] for p in _fetch_paper_summary()["positions"]]
    except Exception:
        open_position_tickers = []

    digest = _catalyst_digest(watch_tickers + open_position_tickers, days_ahead=7)
    if digest:
        with st.container(border=True):
            st.markdown("**🗓️ Upcoming this week**")
            for d in digest:
                when = "today" if d["days"] == 0 else f"in {d['days']}d"
                st.write(f"**{d['ticker']}** — earnings {when} ({d['date']})")
        st.caption("Earnings only (the one catalyst type with a reliable numeric date to sort by) "
                   "across your Watchlist + open Paper positions.")

    if not watched:
        st.info("Your watchlist is empty. Add a ticker above to start tracking it.")
    else:
        _render_watchlist_table(watched)


def _render_candlestick(ticker: str, key_prefix: str):
    """Expanded real-OHLC candlestick view for a selected table row -- the inline sparkline
    (a single line of closes) is fine for a table cell but was never meant to BE the price chart;
    clicking a row wants a genuine trading-app-style view, closer to what a real broker shows.
    Reuses the SAME cached load_prices() (2y daily bars) already used for the table's own day-
    change/sparkline calc -- no new data source.

    PIIP audit 2026-08: rebuilt on TradingView's Lightweight Charts (see
    LIGHTWEIGHT_CHARTS_JS / _load_lightweight_charts_js(), first used for the 0DTE intraday
    chart), same as every other price chart on the platform now -- real scroll/pinch zoom with
    price auto-fitting the visible range, plus the crosshair OHLC/date legend. Daily bars use a
    plain {year, month, day} time object (no time-of-day component to get wrong), unlike the
    intraday chart's ET-seconds handling."""
    try:
        df = load_prices(ticker)
    except Exception as e:
        st.error(f"Couldn't load price history for {ticker}: {e}")
        return
    RANGES = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "ALL": None}
    rng = st.radio("Range", list(RANGES.keys()), index=1, horizontal=True,
                   key=f"{key_prefix}_candle_range_{ticker}", label_visibility="collapsed")
    n = RANGES[rng]
    cdf = (df.tail(n) if n else df).reset_index()
    cdf.columns = ["Date"] + list(cdf.columns[1:])   # normalize whatever yfinance named the index
    cdf = cdf[cdf["Close"].notna()]   # drop any trailing all-NaN session row (yfinance sometimes
    # appends one for the most recent day before that session's data is fully available upstream)

    last_close = float(cdf["Close"].iloc[-1])
    day_chg = ((last_close / float(cdf["Close"].iloc[-2]) - 1) * 100) if len(cdf) >= 2 else 0.0
    chg_color = "#79ed8e" if day_chg >= 0 else "#ff8080"
    st.markdown(f'<div style="font-size:1.3rem;font-weight:800;margin:0.3rem 0">{ticker} '
               f'<span style="font-size:1.1rem">${last_close:,.2f}</span> '
               f'<span style="font-size:0.9rem;color:{chg_color}">{day_chg:+.2f}% (period)</span></div>',
               unsafe_allow_html=True)

    def _bday(ts):
        return {"year": int(ts.year), "month": int(ts.month), "day": int(ts.day)}
    candles = [{"time": _bday(r.Date), "open": round(float(r.Open), 4), "high": round(float(r.High), 4),
               "low": round(float(r.Low), 4), "close": round(float(r.Close), 4)}
              for r in cdf.itertuples()]
    chart_id = f"{key_prefix}_candle_{ticker}_{rng}".replace(" ", "_")
    payload = json.dumps({"candles": candles})
    html = f"""
<div id="{chart_id}_legend" style="font-family:ui-monospace,Consolas,monospace;font-size:0.78rem;
     color:#e8ecec;margin-bottom:0.3rem;min-height:1.2em;">Loading…</div>
<div id="{chart_id}" style="width:100%;height:365px;"></div>
<script>{_load_lightweight_charts_js()}</script>
<script>
(function() {{
  const data = {payload};
  const ticker = {json.dumps(ticker)};
  const container = document.getElementById("{chart_id}");
  const legend = document.getElementById("{chart_id}_legend");
  const chart = LightweightCharts.createChart(container, {{
    autoSize: true,
    layout: {{ background: {{ type: "solid", color: "transparent" }}, textColor: "#8b9a9d" }},
    grid: {{ vertLines: {{ color: "#1c2426" }}, horzLines: {{ color: "#1c2426" }} }},
    rightPriceScale: {{ borderColor: "#232b2d" }},
    timeScale: {{ borderColor: "#232b2d" }},
    crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
  }});
  const candleSeries = chart.addSeries(LightweightCharts.CandlestickSeries, {{
    upColor: "#79ed8e", downColor: "#ff8080", borderVisible: false,
    wickUpColor: "#79ed8e", wickDownColor: "#ff8080",
  }});
  candleSeries.setData(data.candles);

  function fmtDate(t) {{
    const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
    return `${{MONTHS[t.month - 1]}} ${{t.day}}, ${{t.year}}`;
  }}
  function updateLegend(param) {{
    let bar = (param && param.time && param.seriesData) ? param.seriesData.get(candleSeries) : null;
    if (!bar && data.candles.length) bar = data.candles[data.candles.length - 1];
    if (!bar) {{ legend.textContent = "No data"; return; }}
    const upDown = bar.close >= bar.open ? "#79ed8e" : "#ff8080";
    legend.innerHTML = `<b>${{ticker}}</b> &nbsp; ${{fmtDate(bar.time)}} &nbsp; `
      + `O <span style="color:${{upDown}}">${{bar.open.toFixed(2)}}</span> `
      + `H <span style="color:${{upDown}}">${{bar.high.toFixed(2)}}</span> `
      + `L <span style="color:${{upDown}}">${{bar.low.toFixed(2)}}</span> `
      + `C <span style="color:${{upDown}}">${{bar.close.toFixed(2)}}</span>`;
  }}
  chart.subscribeCrosshairMove(updateLegend);
  updateLegend(null);

  chart.timeScale().fitContent();
}})();
</script>
"""
    st.iframe(html, height=395)


# ─────────────────────────── Screener ───────────────────────────
@st.fragment
def _render_screener():
    """Fragment-scoped so adjusting a filter (price/change range, top-N) only reruns this section
    -- without it, every number_input tweak would trigger the SAME whole-page fade this session
    already fixed twice for Positions/Home (nav, popovers, everything re-executing for what should
    be an instant local re-filter of already-fetched results). The Scan button lives in here too,
    not just the filters -- scanning doesn't need to affect anything outside this page, so there's
    no reason for it to be a full-app rerun either."""
    _info_header("🧮 Screener", (
        "Screens the real S&P 500 (free — Wikipedia's own maintained constituent table, no "
        "paid data, no fabricated universe) plus the major index/sector ETFs (SPY, QQQ, IWM, "
        "DIA, and the 10 SPDR sector funds) by price and today's % move. Batched requests, "
        "not one-per-ticker — a full scan is a handful of network calls, not 500+."), size="1.3rem")
    # Narrow columns + a trailing spacer, not st.columns(3) -- on a wide layout page, 3 equal
    # columns each stretch a 2-3-digit number_input to a third of the full browser width (confirmed
    # from a live screenshot: gigantic dark boxes around tiny numbers). Grouping (min/max price,
    # min/max change, top-N) stays the same, just narrower + pushed left instead of spread full-width.
    sc1, sc2, sc3, _sp = st.columns([1, 1, 1, 3])
    price_min = sc1.number_input("Min price ($)", value=0.0, step=1.0, min_value=0.0, key="scr_pmin")
    price_max = sc1.number_input("Max price ($, 0 = no cap)", value=0.0, step=1.0, min_value=0.0, key="scr_pmax")
    chg_min = sc2.number_input("Min 1D change (%)", value=-100.0, step=0.5, key="scr_cmin")
    chg_max = sc2.number_input("Max 1D change (%)", value=100.0, step=0.5, key="scr_cmax")
    topn = sc3.slider("Show top N (by |1D change|)", 10, 200, 50, key="scr_topn")

    if st.button("🔄 Scan S&P 500 + ETFs", type="primary"):
        with st.spinner("Fetching the S&P 500 constituent list…"):
            constituents = get_sp500_list()
        if not constituents:
            st.error("Couldn't load the S&P 500 list right now (Wikipedia fetch failed) — try again shortly.")
        else:
            # ETFs appended after (not deduped against) the S&P 500 list -- none of SPY/QQQ/IWM/DIA
            # or the sector SPDRs are themselves S&P 500 constituents, so there's no overlap to worry about.
            constituents = constituents + screener.etf_universe()
            tickers = [c["ticker"] for c in constituents]
            with st.spinner(f"Scanning {len(tickers)} tickers (batched — a handful of requests, "
                            f"not {len(tickers)})…"):
                results = screener.batch_scan(tickers)
            st.session_state["screener_results"] = results
            st.session_state["screener_meta"] = {c["ticker"]: c for c in constituents}
            st.toast(f"Scanned {len(results)}/{len(tickers)} tickers (S&P 500 + ETFs).")

    results = st.session_state.get("screener_results")
    meta = st.session_state.get("screener_meta", {})
    if not results:
        st.info("Hit **Scan S&P 500 + ETFs** to build the screener results.")
    else:
        rows = []
        for tk, d in results.items():
            price = d.get("price")
            chg = d.get("day_change_pct")
            if price is None or price < price_min or (price_max and price > price_max):
                continue
            if chg is not None and not (chg_min <= chg <= chg_max):
                continue
            m = meta.get(tk, {})
            rows.append({"Symbol": tk, "Company": m.get("name", tk), "Sector": m.get("sector") or "—",
                        "Chart": d.get("spark") or [], "Price": price, "1D %": chg,
                        "Volume": d.get("volume")})
        rows.sort(key=lambda r: abs(r["1D %"]) if r["1D %"] is not None else -1, reverse=True)
        rows = rows[:topn]
        if not rows:
            st.info("No scanned tickers match these filters. Try widening the price/change range.")
        else:
            sdf = pd.DataFrame(rows)
            max_abs = _safe_abs_max(sdf["1D %"])
            styled_s = (sdf.style
                        .apply(lambda col: [_pnl_bar_style(v, max_abs) for v in col], subset=["1D %"])
                        .format({"Price": lambda v: f"${v:,.2f}" if v is not None and v == v else "—",
                                 "1D %": lambda v: f"{v:+.1f}%" if v is not None and v == v else "—",
                                 "Volume": lambda v: f"{v:,.0f}" if v is not None and v == v else "—"}))
            # width="stretch" + explicit per-column widths -- "content" left a big empty gap beside
            # the table instead of filling the page (same "sad open space" issue found on Home's
            # table); plain "stretch" with no column_config re-introduces the ORIGINAL bug (every
            # column, including short Price/1D %/Volume cells, padded out evenly).
            scr_col_cfg = {
                "Symbol": st.column_config.Column("Symbol", width="small", help=glossary.help_for("Symbol")),
                "Company": st.column_config.Column("Company", width="medium", help=glossary.help_for("Company")),
                "Sector": st.column_config.Column("Sector", width="medium", help=glossary.help_for("Sector")),
                "Chart": st.column_config.LineChartColumn("Chart", width="small", color="auto",
                                                           help=glossary.help_for("Chart")),
                "Price": st.column_config.Column("Price", width="small", help=glossary.help_for("Price")),
                "1D %": st.column_config.Column("1D %", width="small", help=glossary.help_for("1D %")),
                "Volume": st.column_config.Column("Volume", width="small", help=glossary.help_for("Volume")),
            }
            # Chart rendered ABOVE the table (user's ask -- previously it sat below the results
            # table, which at up to 50 rows tall meant scrolling past the whole table to see it).
            # Read the table's OWN prior selection from session_state, keyed by the same "scr_table"
            # key the widget below uses -- Streamlit mirrors a key-tracked widget's return value
            # into st.session_state[key] synchronously, ahead of the on_select="rerun" trigger, so
            # this reflects a just-clicked row immediately rather than lagging one run behind.
            # Before the widget has EVER been rendered under this key (very first scan), that
            # lookup is None -- default to row 0 there too so this matches the `selection_default`
            # passed to st.dataframe below instead of showing no chart on the very first scan.
            prev_sel_state = st.session_state.get("scr_table")
            if prev_sel_state is not None and getattr(prev_sel_state, "selection", None):
                prev_sel = list(prev_sel_state.selection.rows)
            else:
                prev_sel = [] if prev_sel_state is not None else [0]
            if prev_sel:
                _render_candlestick(rows[prev_sel[0]]["Symbol"], key_prefix="scr")
                st.divider()

            # Explicit height sized to every row, not Streamlit's default -- without it,
            # st.dataframe caps itself to a fixed ~400px box and scrolls internally even though
            # the page has plenty of room below, which is exactly the "small table you have to
            # scroll in" complaint. Same height formula already used for the Positions/Home tables.
            st.dataframe(styled_s, width="stretch", hide_index=True,
                        column_config=scr_col_cfg, height=(len(rows) + 1) * 35 + 3,
                        on_select="rerun", selection_mode="single-row", key="scr_table",
                        selection_default={"selection": {"rows": [0]}})
            st.caption(f"{len(rows)} of {len(results)} scanned tickers match your filters "
                       f"(out of {len(meta)} total S&P 500 + ETF tickers) — select a row for a "
                       "full price chart.")


if nav == "Screener":
    _render_screener()


# ─────────────────────────── Opportunity Feed ───────────────────────────
if nav == "Feed":
    _info_header("📰 Feed", (
        "Ranked by the options market's expected move (~30d) — the honest read of 'about to "
        "move big, in either direction.' News is context, not the ranking. You pick call or "
        "put; the Scorecard tracks how each decision plays out."), size="1.3rem")
    wsb = get_wsb()

    # Narrow columns + a trailing spacer, not st.columns(3) -- same fix as the Screener's filter
    # row: on this wide-layout page, 3 equal-width columns each stretch a small number_input to a
    # third of the full browser width.
    fc1, fc2, fc3, _sp = st.columns([1, 1, 1, 3])
    budget = fc1.number_input("Budget per trade ($)", value=500, step=50, min_value=1, key="feed_budget")
    topn = fc2.slider("Show top N", 3, 20, 12, key="feed_topn")
    max_price = fc3.number_input("Max share price ($, 0 = no cap)", value=0, step=1, min_value=0,
                                 key="feed_maxprice",
                                 help="Set e.g. 5 to see only low-priced / 'penny' names (all Robinhood-listed) "
                                      "— fun to play with small money. These dilute + have thin options, so lean "
                                      "on Deep Research's cash-runway and the ⚠️ liquidity flags.")
    if st.button("🔄 Scan stocks", type="primary"):
        with st.spinner("Scanning ~80 stocks (mega-caps + lower-priced + biotech + low-priced; ~2–3 min; cached 15 min)…"):
            scanned = run_scan()
        for _c in scanned:
            scanner.log_card_forecast(_c, DB)   # feed the Scorecard with the model's forecast (deduped)
        st.session_state["cards"] = scanned
    all_cards = st.session_state.get("cards", [])
    if not all_cards:
        st.info("Hit **Scan stocks** to build the ranked feed.")
    if all_cards:
        st.info("🎨 **The colored flags describe each stock's *condition* — they are NOT buy signals.** "
                "🟢 just means a reading is *present* (e.g. 'uptrend'); it does **not** mean 'buy,' and "
                "**all-green ≠ GO.** The direction (call vs put) is a **coin flip** — we backtested it. "
                "Use flags to decide what's worth *researching* → then the news, the specific contract, "
                "and your risk sizing make the trade. Click **Details ▸** for the full breakdown.")
    fcol1, fcol2, fcol3 = st.columns([1.6, 1, 1.1])
    ncol = fcol1.slider("Cards per row", 2, 5, 4, key="feed_ncol")
    hide_noise = fcol2.checkbox(
        "🔇 Hide unexplained-IV noise", value=False, key="feed_hidenoise",
        help="Drops chronic high-IV names with no known catalyst (e.g. RXRX) — the volatility the feed "
             "ranks on, with no dated reason behind it. Leaves cards where there IS a reason to look.")
    fits_budget = fcol3.checkbox(
        f"💵 Only cards I can afford (≤ \\${budget:,})", value=False, key="feed_fitsbudget",
        help="Hides cards whose cheapest *sensible* strike (within the expected move — not a lottery "
             "ticket) costs more than your budget above. With a small budget this mostly leaves cheaper "
             "stocks — the honest reality: pricier stocks' options need more capital.")
    if fits_budget and not any("min_entry_cost" in c for c in all_cards):
        st.warning("Re-run **🔄 Scan stocks** to compute budget fit for each card.")
        fits_budget = False
    # Tag each card (catalyst vs unexplained-IV noise) in expected-move order; optionally drop noise/over-budget.
    shown = []
    for c in all_cards:
        if max_price and c.get("spot", 0) > max_price:      # low-priced / penny filter
            continue
        c["_tag"] = assess.catalyst_tag(c, get_earnings(c["ticker"]))
        if hide_noise and c["_tag"]["key"] == "unexplained":
            continue
        mec = c.get("min_entry_cost")
        if fits_budget and (mec is None or mec > budget):
            continue
        shown.append(c)
        if len(shown) >= topn:
            break
    if max_price and all_cards and not shown:
        st.info(f"No scanned stocks are under ${max_price}. Try a higher cap, or re-scan.")
    for i in range(0, len(shown), ncol):
        cols = st.columns(ncol)
        for j in range(ncol):
            if i + j >= len(shown):
                continue
            c = shown[i + j]
            rank = i + j + 1
            qf = assess.quick_flags(c)
            vol = qf["Volatility"]
            conf = c.get("lean_conf") or 0
            tier = "🟩🟩🟩 strong" if conf >= 0.6 else ("🟩🟩⬜ moderate" if conf >= 0.4 else "🟩⬜⬜ weak")
            buzz = social.buzz_for(c["ticker"], wsb)
            tag = c.get("_tag") or {}
            tile = _tile_style(tag.get("color", "⚪"))
            me = c.get("min_entry")
            entry_line = ""
            if me:
                fit = "✅ fits" if me["cost"] <= budget else "🚫 over"
                entry_line = (f'<div style="font-size:0.78rem;color:#8b9a9d;margin-top:0.35rem">'
                              f'cheapest sensible entry ≈ ${me["cost"]:,} ({fit} ${budget:,}) — '
                              f'${me["strike"]:g} {me["type"]} @ {me["expiry"]}</div>')
            with cols[j]:
                st.markdown(
                    f'<div style="background:{tile["bg"]};border:1px solid {tile["border"]};'
                    'border-radius:8px;padding:0.75rem 0.9rem;margin-bottom:0.4rem">'
                    f'<div style="font-weight:700;font-size:0.92rem">#{rank} · {c["ticker"]} · ${c["spot"]:,.2f}</div>'
                    f'<div style="font-size:0.78rem;color:#8b9a9d;margin-top:0.1rem">{c.get("name", c["ticker"])}</div>'
                    f'<div style="font-size:0.8rem;margin-top:0.4rem">'
                    f'<b>±{c["expected_move_pct"]}%</b> move · IV {c["iv_pct"]}% ({vol["color"]} {vol["label"]})'
                    + (f' · 🔥 #{buzz["rank"]}' if buzz else '') + '</div>'
                    f'<div style="font-size:0.8rem;margin-top:0.3rem">'
                    + " · ".join(f'{qf[k]["color"]} {qf[k]["label"]}' for k in ("Trend", "Momentum", "Model lean"))
                    + '</div>'
                    f'<div style="font-size:0.78rem;color:#8b9a9d;margin-top:0.3rem">'
                    f'{tag.get("color", "")} <b>{tag.get("label", "")}</b> · Alignment: {tier}</div>'
                    f'{entry_line}'
                    '</div>', unsafe_allow_html=True)
                if st.button("Details ▸", key=f"details_{c['ticker']}", width="stretch"):
                    stock_detail_dialog(c["ticker"])
    _info_header("ℹ️ Card legend", (
        "#rank = by expected move (biggest mover first). The catalyst tag says whether the "
        "volatility has a reason: 🟢 dated event (earnings) ahead · 🟡 unexplained rich IV "
        "(expensive noise unless you know a catalyst) · ⚪ nothing scheduled. It flags what's "
        "worth researching — not who wins (direction is ~a coin flip). 'Alignment' = how "
        "strongly the mechanical flags agree, also NOT a win probability. 'Cheapest sensible "
        "entry' = least capital for a within-move (non-lottery) strike near ~30d; use the "
        "💵 toggle to hide cards your budget can't reach."), size="0.85rem")

# ─────────────────────────── Reddit Momentum ───────────────────────────
if nav == "Reddit Momentum":
    _info_header("🔥 Reddit Momentum — not a stock picker", (
        "Detects unusually fast Reddit-discussion increases across 6 subreddits (free, no "
        "key), scores momentum from real mention/rank data only, and points anything worth a "
        "look at the existing 🔭 Ticker Page / Deep Research tools for the 'is this real' "
        "step. No AI post-reading, no fabricated backtests — everything here is either live "
        "data or math on that data."), size="1.3rem")

    with st.expander("🔥 Trending across Reddit (r/wallstreetbets, r/stocks, r/investing, r/options, "
                     "r/pennystocks, r/StockMarket) — context, NOT a signal"):
        rmulti = get_reddit_multi()
        # Sectioned by subreddit (design chosen from the reddit_sectioned_options.html comparison,
        # "Option A": side-by-side columns) -- the old view was one flat list merged across all 6
        # subs, which used almost none of the page's width and buried WHICH subreddit each ticker
        # was actually trending in. This also honestly surfaces something the merged view hid: r/
        # wallstreetbets dominates real engagement (100s of mentions) while the other 5 subs are
        # much quieter (often single digits) -- real ApeWisdom data, not a rendering issue.
        if not any(rmulti.values()):
            st.write("Couldn't load Reddit data right now (aggregator down or rate-limited).")
        else:
            sub_cols = st.columns(len(social.SUBREDDITS))
            for col, sub in zip(sub_cols, social.SUBREDDITS):
                rows = rmulti.get(sub) or {}
                top = sorted(rows.items(), key=lambda kv: (kv[1]["rank"] or 1e9))[:6]
                with col:
                    # Green tile style (_tile_style("🟢")'s own colors), not the blue pill used
                    # elsewhere -- user's explicit preference after seeing the option comparison.
                    st.markdown(f'<div style="font-family:ui-monospace,Consolas,monospace;'
                               f'font-size:0.68rem;letter-spacing:0.04em;text-transform:uppercase;'
                               f'color:#79ed8e;font-weight:700;background:#152a1e;'
                               f'border:1px solid #2c5a3c;border-radius:4px;padding:0.3rem 0.5rem;'
                               f'margin-bottom:0.4rem">r/{sub}</div>', unsafe_allow_html=True)
                    if not top:
                        st.caption("No data")
                    for tk, d in top:
                        st.caption(f"#{d['rank']}  **{tk}**  ·  {d['mentions']:,}")
        st.caption("Combined across all 6 tracked subreddits (moved here from the Feed tab, and no "
                   "longer WSB-only) — real ApeWisdom mention/rank data, same source the momentum "
                   "scan below uses. Retail buzz is noisy and, at extremes, a **contrarian** tell "
                   "(a stampede often marks a top) — context, not a signal.")

    # Narrower first column + a trailing spacer, not [1, 3] -- same fix as the Screener/Feed filter
    # rows: 1/4 of this wide-layout page is still a huge box for a 2-3 digit number.
    rc1, rc2, _sp = st.columns([1, 1, 4])
    min_score = rc1.number_input("Min momentum score", 0, 100, 60, step=5, key="rm_min_score")
    if rc2.button("🔄 Scan Reddit", type="primary"):
        with st.spinner("Scanning r/wallstreetbets, r/stocks, r/investing, r/options, "
                        "r/pennystocks, r/StockMarket…"):
            cands = rm.scan(min_score=min_score)
            for c in cands[:20]:                        # log the top 20, not all — keeps the
                rm.log_prediction(c["ticker"], c)        # Scorecard meaningful, not spammed
        st.session_state["rm_candidates"] = cands
        st.session_state["rm_scanned_msg"] = (
            f"{len(cands)} ticker(s) crossed a momentum score of {min_score:.0f}. "
            f"Top 20 logged to the 🎯 Scorecard as a graded 30-day call.")

    if st.session_state.get("rm_scanned_msg"):
        st.success(st.session_state.pop("rm_scanned_msg"))

    cands = st.session_state.get("rm_candidates", [])
    if not cands:
        st.info("Hit **🔄 Scan Reddit** to build the candidate list.")
    else:
        rc3, rc4 = st.columns([1, 1])
        topn = rc3.slider("Show top N", 3, 30, 12, key="rm_topn")
        sort_by = rc4.radio("Order by", ["Momentum score", "Mentions count"], horizontal=True, key="rm_sort")
        sort_key = (lambda c: c["score"]) if sort_by == "Momentum score" else (lambda c: c["total_mentions"])
        shown = sorted(cands, key=sort_key, reverse=True)[:topn]

        rows = []
        for c in shown:
            cat = get_rm_catalyst(c["ticker"])
            risk = rm.risk_flag(c, cat)
            evidence = [risk["detail"]]
            if cat.get("next_catalyst"):
                evidence.append(f"📅 {cat['next_catalyst']} — {cat['next_catalyst_why']}")
            if c.get("is_new_appearance"):
                evidence.append("🆕 first time on the radar")
            evidence.append("Subreddits: " + ", ".join(f"r/{s}" for s in c["subs"]))
            rows.append({
                "label": c["ticker"],
                "context": f'{risk["label"]} · {c["total_mentions"]:,} mentions '
                           f'({c["mention_change_pct"]:+.0f}% vs 24h) · '
                           f'rank #{c["best_rank"]} in r/{c["best_rank_sub"]} · '
                           f'{c["n_subs"]} sub(s)',
                "value": f'{c["score"]:.0f}/100',
                "color": _score_color(c["score"]),
                "evidence": evidence,
                "ticker": c["ticker"],
            })
        _render_list_view([{"label": f"🔥 REDDIT MOMENTUM — sorted by {sort_by.lower()}", "rows": rows}],
                          container_key="rm_list_rows")
        _info_header("ℹ️ How Momentum Score works", (
            "Momentum Score is built only from real mention counts, rank, and cross-subreddit "
            "spread — it is NOT a win-rate signal. Context line shows Catalyst-linked (a known "
            "dated event lines up with the spike) vs Unexplained/Unconfirmed (high momentum, "
            "no known reason found yet — could be real, could be pure hype). Every scan is "
            "logged and graded later in the 🎯 Scorecard so you can see, honestly, whether "
            "Reddit momentum has actually meant anything. Tap a row for detail and a research "
            "shortcut."), size="0.85rem")


# ─────────────────────────── Catalyst Terminal ───────────────────────────
@st.cache_data(ttl=280)
def _ct_fetch_raw():
    return ct.fetch_raw()


@st.fragment(run_every=ct.REFRESH_SECONDS)
def _render_catalyst_terminal():
    from datetime import timezone
    st.caption(f"⏱️ As of {datetime.now().strftime('%H:%M:%S')} — refreshes every "
               f"{ct.REFRESH_SECONDS // 60} min (free-tier news API quota, not the 30s price feed)")
    if not os.getenv("FINNHUB_API_KEY"):
        st.info("📰 Catalyst Terminal needs a free Finnhub API key. Sign up at finnhub.io/register "
                "and add `FINNHUB_API_KEY=...` to your `.env` file (see `.env.example`).")
        return
    raw = _ct_fetch_raw()
    if not raw:
        st.warning("No headlines returned right now (API hiccup or empty feed) — try again shortly.")
        return
    scored = ct.score_and_dedupe(raw)
    st.caption(f"{len(raw)} raw headlines → {len(scored)} deduplicated stories")

    now = datetime.now(timezone.utc)
    with_calib = 0
    for item in scored[:15]:
        published_dt = datetime.fromisoformat(item["published"])
        age_min = max((now - published_dt).total_seconds() / 60, 0)
        age_str = f"{age_min:.0f}m ago" if age_min < 60 else f"{age_min / 60:.1f}h ago"
        context = catcal.summarize(item["calibration"])
        if context:
            with_calib += 1
        lean = item["lean"]
        tile = _tile_style(lean["emoji"])
        flags_line = (f'<div style="font-size:0.76rem;color:#8b9a9d;margin-top:0.2rem">'
                     f'Flags: {", ".join(item["importance_hits"])}</div>' if item["importance_hits"] else "")
        context_line = (f'<div style="font-size:0.76rem;color:#8b9a9d;margin-top:0.15rem">'
                        f'{context}</div>' if context else "")
        link_html = (f' · <a href="{item["url"]}" target="_blank" style="color:#87d1ff">open ↗</a>'
                    if item.get("url") else "")
        st.markdown(
            f'<div style="background:{tile["bg"]};border:1px solid {tile["border"]};'
            'border-radius:8px;padding:0.6rem 0.9rem;margin-bottom:0.3rem;display:flex;'
            'align-items:flex-start;justify-content:space-between;gap:0.75rem">'
            '<div style="flex:1 1 auto;min-width:0">'
            f'<div style="font-size:0.86rem;font-weight:600">{item["headline"]}'
            f'<span style="color:#8b9a9d;font-weight:400;font-size:0.78rem"> · {age_str}{link_html}</span></div>'
            + flags_line + context_line +
            '</div>'
            f'<div style="flex:0 0 auto;text-align:right;white-space:nowrap">'
            f'<div style="font-size:0.8rem;color:{_TEXT_BY_EMOJI[lean["emoji"]]};font-weight:700">'
            f'{lean["emoji"]} {lean["label"]}</div>'
            f'<div style="font-family:ui-monospace,Consolas,monospace;font-size:0.78rem;color:#8b9a9d;'
            f'margin-top:0.15rem">score {item["composite_score"]:.0f}</div>'
            '</div></div>', unsafe_allow_html=True)
    st.caption(f"{with_calib}/{min(len(scored), 15)} headlines matched a researched historical catalyst "
               "category. **Bullish/Bearish lean is a simple keyword heuristic (deterministic, no LLM) "
               "— NOT a backtested signal.** This session's own research found even real historical "
               "calibration rarely gives geopolitical/generic headlines a reliable directional edge; "
               "treat this the same as Feed's 'Model lean' flag — a coin-flip-level reading, not a call.")


# ─────────────────────────── 0DTE Intelligence ───────────────────────────
def _lean_color(text: str) -> str:
    """Direction/lean label -> palette color. Same palette as the rest of the app
    (success/danger/warning/info) so this reads consistently with Home/Paper."""
    t = (text or "").upper()
    if any(k in t for k in ("CALL", "BULL", "LONG")):
        return "#79ed8e"
    if any(k in t for k in ("PUT", "BEAR", "SHORT")):
        return "#ff8080"
    return "#87d1ff"



@st.cache_data
def _alert_beep_data_uri() -> str:
    """Short two-beep alert tone, synthesized locally (no external asset/network call) for the
    in-trade direction-flip alert. Best-effort: browser autoplay policies can block audio until
    the user has interacted with the page at least once -- the 'I'm in a trade' checkbox click
    (and the direction radio pick) satisfy that for the rest of the session in most browsers."""
    rate = 22050
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = bytearray()
        for freq, dur in [(880, 0.12), (0, 0.04), (880, 0.12)]:
            n = int(rate * dur)
            for i in range(n):
                val = int(32767 * 0.4 * math.sin(2 * math.pi * freq * i / rate)) if freq else 0
                frames += struct.pack("<h", val)
        w.writeframes(bytes(frames))
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


def _render_direction_alert(message: str):
    """Persistent red banner (renders every fragment refresh while the condition holds) plus a
    one-shot toast + audio beep on the refresh where the condition first becomes true -- edge-
    triggered so it doesn't re-toast/re-beep every 30s the flip stays flipped."""
    st.markdown(
        f'<div style="border:1px solid #ff8080;border-radius:8px;background:#2a1010;'
        f'padding:0.7rem 1rem;font-weight:700;color:#ff8080;margin:0.4rem 0">'
        f'\U0001f6a8 {message}</div>', unsafe_allow_html=True)


def _render_diverging_chart(rows: list[dict], footer_html: str | None = None):
    """Same metrics as the list-view table above, replotted on one shared 0-100/50-centered axis
    so 'does everything line up, or is this mixed' reads at a glance.

    Deliberately plain st.markdown, NOT st.iframe like the list-view above: an auto-height iframe
    rebuilt every 30s inside this page's st.fragment(run_every=30) was observed stacking up in the
    browser (the page kept growing, one extra copy of the table per refresh) instead of cleanly
    replacing itself. This chart has no click-to-expand interaction, so it doesn't need an iframe
    at all -- plain HTML injected via st.markdown is content Streamlit already knows how to diff
    and replace in place each rerun, sidestepping that failure mode entirely. Every row built on a
    single line, no leading indentation, so Markdown never mistakes any of this for a code block.
    """
    row_html = []
    for ri, r in enumerate(rows):
        left_w = (50 - r["value"]) / 50 * 100 if r["value"] < 50 else 0.0
        right_w = (r["value"] - 50) / 50 * 100 if r["value"] > 50 else 0.0
        border = "" if ri == 0 else ";border-top:1px solid #142020"
        row_html.append(
            f'<div style="display:grid;grid-template-columns:1.3fr 3.4fr 0.5fr;align-items:center;'
            f'gap:0.65rem;padding:0.45rem 0{border}"><div style="font-size:0.82rem;font-weight:600;'
            f'color:#8ecfe0">{r["label"]}</div><div style="display:flex;align-items:center;height:18px">'
            f'<div style="width:50%;height:8px;display:flex;justify-content:flex-end">'
            f'<div style="width:{left_w:.1f}%;height:8px;border-radius:2px 0 0 2px;background:{r["color"]}">'
            f'</div></div><div style="width:1px;height:20px;background:#2a3a3c;flex-shrink:0"></div>'
            f'<div style="width:50%;height:8px;display:flex;justify-content:flex-start">'
            f'<div style="width:{right_w:.1f}%;height:8px;border-radius:0 2px 2px 0;background:{r["color"]}">'
            f'</div></div></div><div style="font-family:ui-monospace,Consolas,monospace;font-weight:700;'
            f'font-size:0.85rem;text-align:right;color:{r["color"]}">{r["display"]}</div></div>')
    footer = (f'<div style="margin-top:0.7rem;padding-top:0.6rem;border-top:1px solid #142020;'
             f'font-size:0.78rem;color:#8b9a9d">{footer_html}</div>') if footer_html else ""
    axis = ('<div style="display:flex;justify-content:space-between;font-family:ui-monospace,Consolas,'
           'monospace;font-size:0.8rem;letter-spacing:0.06em;text-transform:uppercase;color:#87d1ff;'
           'font-weight:700;padding-bottom:0.6rem;border-bottom:1px solid #142020;margin-bottom:0.5rem">'
           '<span>&larr; bearish / weak</span><span>bullish / strong &rarr;</span></div>')
    html_out = (f'<div style="border:1px solid #1c2b2d;border-radius:8px;background:#0c1414;'
               f'padding:0.9rem 1rem 0.7rem">{axis}{"".join(row_html)}{footer}</div>')
    st.markdown(html_out, unsafe_allow_html=True)


# market_dna.classify()'s `metrics` dict is keyed by raw Python variable names (session_minutes,
# vwap_side_consistency, ...) meant for logging/backtesting, not display -- shown as-is in the DNA
# metrics tile grid below this used to read literally as e.g. "vwap_side_consistency  0.85" with no
# explanation anywhere. This maps each key to the human label used both for display AND as the
# glossary.help_for() lookup key (see the "Market DNA (day-type read)" glossary group), plus a
# per-key value formatter since the raw values are also unformatted (bare ratios, 1/0 booleans).
_DNA_METRIC_LABELS = {
    "session_minutes": "Session Length",
    "gap_pct": "Overnight Gap",
    "gap_held": "Gap Held",
    "range_vs_atr": "Range vs Normal (ATR)",
    "net_vs_range": "Net Move vs Range",
    "vwap_side_consistency": "VWAP Consistency",
    "day_change_pct": "Day Change",
}


def _dna_metric_display(key: str, val) -> str:
    if val is None:
        return "—"
    if key in ("gap_pct", "day_change_pct"):
        return f"{val:+.2f}%"
    if key in ("net_vs_range", "vwap_side_consistency"):
        return f"{val * 100:.0f}%"
    if key == "range_vs_atr":
        return f"{val:.2f}x"
    if key == "gap_held":
        return "Yes" if val else "No"
    if key == "session_minutes":
        return f"{val:.0f} min"
    return str(val)


def _render_lightweight_lines(series: dict, key_prefix: str, colors: dict | None = None,
                              height: int = 320, intraday: bool = False):
    """Generic multi-line Lightweight Charts renderer -- PIIP audit 2026-08, shared by the
    Research page's daily Close/SMA50/SMA200 chart and its intraday price/VWAP chart (both are
    just N line series over time, no candles/volume needed, so this avoids a 3rd/4th copy of the
    same chart-boilerplate JS). `series` is {label: pd.Series indexed by date/datetime}.
    `intraday=True` uses the ET-seconds time format (see _et_seconds()); False uses the plain
    {year, month, day} format the daily-bar charts use (no time-of-day component to get wrong)."""
    series = {label: s.dropna() for label, s in series.items() if s is not None and not s.dropna().empty}
    if not series:
        st.info("No data available.")
        return
    default_colors = ["#87d1ff", "#c792ea", "#82aaff", "#79ed8e", "#ff8080"]
    colors = colors or {}
    chart_id = f"{key_prefix}_lines".replace(" ", "_")
    payload_series = {}
    for i, (label, s) in enumerate(series.items()):
        if intraday:
            pts = [{"time": _et_seconds(ts), "value": round(float(v), 4)} for ts, v in s.items()]
        else:
            pts = [{"time": {"year": int(ts.year), "month": int(ts.month), "day": int(ts.day)},
                    "value": round(float(v), 4)} for ts, v in s.items()]
        payload_series[label] = {"points": pts, "color": colors.get(label, default_colors[i % len(default_colors)])}
    payload = json.dumps(payload_series)
    legend_html = "".join(
        f'<span style="margin-right:1rem"><span style="color:{d["color"]}">●</span> {label}</span>'
        for label, d in payload_series.items())
    html = f"""
<div style="font-size:0.78rem;color:#8b9a9d;margin-bottom:0.3rem">{legend_html}</div>
<div id="{chart_id}" style="width:100%;height:{height}px;"></div>
<script>{_load_lightweight_charts_js()}</script>
<script>
(function() {{
  const data = {payload};
  const container = document.getElementById("{chart_id}");
  const chart = LightweightCharts.createChart(container, {{
    autoSize: true,
    layout: {{ background: {{ type: "solid", color: "transparent" }}, textColor: "#8b9a9d" }},
    grid: {{ vertLines: {{ color: "#1c2426" }}, horzLines: {{ color: "#1c2426" }} }},
    rightPriceScale: {{ borderColor: "#232b2d" }},
    timeScale: {{ borderColor: "#232b2d", timeVisible: {str(intraday).lower()} }},
    crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
  }});
  for (const label in data) {{
    const s = data[label];
    const lineSeries = chart.addSeries(LightweightCharts.LineSeries, {{
      color: s.color, lineWidth: 2, title: label,
      lastValueVisible: true, priceLineVisible: false,
    }});
    lineSeries.setData(s.points);
  }}
  chart.timeScale().fitContent();
}})();
</script>
"""
    st.iframe(html, height=height + 30)


def _render_intraday_candlestick(ticker: str, intraday_df, key_prefix: str,
                                 data_quality: dict | None = None):
    """Live intraday candlestick chart for the 0DTE page — refreshes every 30s alongside the rest
    of the page fragment, using the SAME 1m bars already fetched above (no new network call).

    PIIP audit 2026-08: rebuilt on TradingView's Lightweight Charts (vendored locally, see
    LIGHTWEIGHT_CHARTS_JS) instead of Altair -- a real trading-terminal feel needs the price scale
    to auto-fit whatever time range is visible when you zoom/pan, which Altair/Vega-Lite (a
    statistical-charting grammar, not built for this) can't do well; Lightweight Charts does this
    natively, no extra work. Resamples via tf.resample_ohlc() (already built for the Multi-
    Timeframe Alignment group) instead of a new resampling implementation, and overlays the
    running VWAP (det.running_vwap(), already built for VWAP Crossings) rather than recomputing it.

    Trade-off to be upfront about: this is real JS running in the viewer's browser, so it can be
    verified to EMBED and RUN without a Python-side exception, but the actual zoom/pan/crosshair
    feel can't be confirmed the way everything else on this page has been -- that needs a human
    to look at it."""
    if intraday_df is None or intraday_df.empty:
        st.info("No intraday bars available right now (market closed, or a data gap).")
        return

    # Session freshness banner (PIIP audit 2026-08): confirmed live -- before market open (or
    # after a real data gap), yfinance's "today" fetch returns the LAST COMPLETE session instead
    # (period="1d" has no bars to give for a session that hasn't started yet). Expected behavior,
    # not a bug -- but a user caught this by hovering the chart and seeing an unexpected date, so
    # it needs to be obvious up front, not just discoverable via hover or the Data Quality
    # expander.
    #
    # PIIP audit 2026-08 (state-architecture review, Phase 7): this used to recompute the exact
    # same freshness math data_quality_snapshot() already does -- confirmed duplicate logic, one
    # genuine source of "could two freshness checks on this page ever disagree" risk. Now REUSES
    # the caller's already-computed data_quality dict when passed in (the 0DTE page always has one
    # by the time this chart renders); only falls back to a local computation when called without
    # one, so this function still works standalone.
    # `now_et` is needed below regardless of which freshness path is used (the chart's "now"
    # payload field extends the default view through the current moment) -- always computed
    # locally (cheap, no network call), only the STALENESS VERDICT itself prefers data_quality.
    last_bar_ts = intraday_df.index[-1]
    now_et = pd.Timestamp.now(tz=last_bar_ts.tz) if last_bar_ts.tzinfo else pd.Timestamp.now()
    if data_quality is not None:
        minutes_stale = data_quality.get("underlying_minutes_stale")
        is_stale = data_quality.get("underlying") == "Stale"
    else:
        minutes_stale = (now_et - last_bar_ts).total_seconds() / 60
        is_stale = minutes_stale > 15
    if is_stale:
        st.warning(f"⚠️ Showing the LAST COMPLETE session ({last_bar_ts.strftime('%A, %b %d')}) "
                  "— no live bars yet for today (market not open, or today's data hasn't printed "
                  "yet). This is NOT today's live chart.")

    c1, c2 = st.columns([2, 3])
    with c1:
        tf_choice = st.radio("Candle timeframe", ["1m", "5m", "15m", "30m"], index=1, horizontal=True,
                             key=f"{key_prefix}_candle_tf_{ticker}")
    with c2:
        # Toggleable overlays (PIIP audit 2026-08, per user request -- with all 3 lines always on,
        # overlapping made them hard to tell apart). VWAP on by default (the single most load-
        # bearing read on this page); EMA/SMA off by default so the chart starts decluttered.
        ov1, ov2, ov3 = st.columns(3)
        show_vwap = ov1.checkbox("VWAP", value=True, key=f"{key_prefix}_show_vwap_{ticker}")
        show_ema = ov2.checkbox("EMA(50)", value=False, key=f"{key_prefix}_show_ema_{ticker}")
        show_sma = ov3.checkbox("SMA(50)", value=False, key=f"{key_prefix}_show_sma_{ticker}")

    # Fixed chart height (per user request -- the adjustable size control was tried and removed;
    # this is the midpoint between its old "X-Large" (650) and "XX-Large" (800) presets).
    chart_px = 725

    minutes = {"1m": 1, "5m": 5, "15m": 15, "30m": 30}[tf_choice]
    vwap_1m = det.running_vwap(intraday_df)
    if minutes == 1:
        cdf = intraday_df.copy()
        cdf["VWAP"] = vwap_1m
    else:
        cdf = tf.resample_ohlc(intraday_df, minutes)
        if cdf.empty:
            st.caption(f"Not enough bars yet for {tf_choice} candles.")
            return
        cdf["VWAP"] = vwap_1m.resample(f"{minutes}min").last()
    cdf = cdf.reset_index()
    cdf.columns = ["Time"] + list(cdf.columns[1:])
    # EMA(50)/SMA(50) on THIS interval's own closes (PIIP audit 2026-08, per user request to match
    # a real trading terminal's chart) -- a different thing from deterministic.py's daily EMA20/
    # 50/200 (those run on daily bars for the Daily-timeframe alignment read, not this chart). With
    # under 50 bars (e.g. early in the session, or on 30m candles where 50 bars needs ~25 hours --
    # more than one session), these converge toward the visible data rather than being a "true"
    # 50-period average -- pandas' min_periods=1 default means they still render (no gap), just
    # less meaningful with a short history, same honesty caveat as every other lookback on this page.
    cdf["EMA50"] = cdf["Close"].ewm(span=50, adjust=False).mean()
    cdf["SMA50"] = cdf["Close"].rolling(window=50, min_periods=1).mean()

    candles = [{"time": _et_seconds(r.Time), "open": round(float(r.Open), 4),
               "high": round(float(r.High), 4), "low": round(float(r.Low), 4),
               "close": round(float(r.Close), 4)} for r in cdf.itertuples()]
    volumes = [{"time": _et_seconds(r.Time), "value": float(r.Volume),
               "color": "#79ed8ea0" if r.Close >= r.Open else "#ff8080a0"}
              for r in cdf.itertuples()]
    vwap_pts = [{"time": _et_seconds(r.Time), "value": round(float(r.VWAP), 4)}
               for r in cdf.itertuples() if r.VWAP == r.VWAP]
    ema_pts = [{"time": _et_seconds(r.Time), "value": round(float(r.EMA50), 4)}
              for r in cdf.itertuples() if r.EMA50 == r.EMA50]
    sma_pts = [{"time": _et_seconds(r.Time), "value": round(float(r.SMA50), 4)}
              for r in cdf.itertuples() if r.SMA50 == r.SMA50]

    # VWAP is a volume-weighted average, and confirmed live: yfinance's free tier reports 0
    # volume on every pre/post-market bar (prices are real, volume is not) -- so whenever the
    # visible session is entirely pre/post-market, running_vwap() is correctly all-NaN and there
    # is genuinely nothing to plot yet, not a rendering bug. Surface that honestly (same pattern
    # as the staleness banner) instead of a checkbox that silently does nothing.
    if show_vwap and not vwap_pts:
        st.caption("⚠️ VWAP has no value yet — pre/post-market bars report 0 trade volume on the "
                  "free data tier, so there's nothing to average until regular-hours volume starts "
                  "printing (9:30am ET).")

    chart_id = f"{key_prefix}_lwchart_{ticker}_{tf_choice}".replace(" ", "_")
    payload = json.dumps({"candles": candles, "volumes": volumes, "vwap": vwap_pts,
                          "ema": ema_pts, "sma": sma_pts, "showVwap": show_vwap,
                          "showEma": show_ema, "showSma": show_sma,
                          # PIIP audit 2026-08, per user request: "now" in the SAME faked-UTC
                          # scheme as every timestamp above (see _et_seconds()) -- lets the chart
                          # extend its default view through the actual current moment instead of
                          # fitting tightly around whatever data exists, which made stale data
                          # (e.g. Friday's session, pre-market Monday) look like a normal, current
                          # session at a glance instead of visibly stopping short of "now."
                          "now": _et_seconds(now_et)})
    html = f"""
<div id="{chart_id}_legend" style="font-family:ui-monospace,Consolas,monospace;font-size:0.78rem;
     color:#e8ecec;margin-bottom:0.3rem;min-height:1.2em;">Loading…</div>
<div id="{chart_id}" style="width:100%;height:{chart_px}px;"></div>
<script>{_load_lightweight_charts_js()}</script>
<script>
(function() {{
  const data = {payload};
  const ticker = {json.dumps(ticker)};
  const container = document.getElementById("{chart_id}");
  const legend = document.getElementById("{chart_id}_legend");
  const chart = LightweightCharts.createChart(container, {{
    autoSize: true,
    layout: {{ background: {{ type: "solid", color: "transparent" }}, textColor: "#8b9a9d" }},
    grid: {{ vertLines: {{ color: "#1c2426" }}, horzLines: {{ color: "#1c2426" }} }},
    rightPriceScale: {{ borderColor: "#232b2d" }},
    // shiftVisibleRangeOnNewBar defaults to true in Lightweight Charts -- documented, known
    // behavior (see github.com/tradingview/lightweight-charts issue #549): whenever the last bar
    // is visible, EVERY setData() call (this chart gets one every 30s from the page's own auto-
    // refresh) auto-shifts the view rightward to keep following the latest bar. That's exactly
    // what was fighting our own pan/zoom restore below -- confirmed via a live CDP-driven browser
    // session against the running app that our explicit setVisibleRange() call was being
    // silently overridden shortly after we made it. Disabling it here is the documented fix, not
    // a timing workaround: this chart already has its own explicit "extend to now" / restore-
    // saved-position logic, so the library's auto-follow is redundant and actively harmful here.
    timeScale: {{ borderColor: "#232b2d", timeVisible: true, secondsVisible: false,
                 shiftVisibleRangeOnNewBar: false }},
    crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
  }});

  const candleSeries = chart.addSeries(LightweightCharts.CandlestickSeries, {{
    upColor: "#79ed8e", downColor: "#ff8080", borderVisible: false,
    wickUpColor: "#79ed8e", wickDownColor: "#ff8080",
  }});
  candleSeries.setData(data.candles);

  let volumeSeries = null;
  try {{
    volumeSeries = chart.addSeries(LightweightCharts.HistogramSeries, {{
      priceFormat: {{ type: "volume" }}, priceScaleId: "",
    }}, 1);
    volumeSeries.setData(data.volumes);
    chart.panes()[1].setHeight(80);
  }} catch (e) {{ console.warn("PIIP: volume pane unavailable in this chart version", e); }}

  const vwapSeries = chart.addSeries(LightweightCharts.LineSeries, {{
    color: "#87d1ff", lineWidth: 1, lineStyle: 2, visible: data.showVwap,
    lastValueVisible: false, priceLineVisible: false, title: "VWAP",
  }});
  vwapSeries.setData(data.vwap);

  const emaSeries = chart.addSeries(LightweightCharts.LineSeries, {{
    color: "#c792ea", lineWidth: 1, visible: data.showEma,
    lastValueVisible: false, priceLineVisible: false, title: "EMA(50)",
  }});
  emaSeries.setData(data.ema);

  const smaSeries = chart.addSeries(LightweightCharts.LineSeries, {{
    color: "#82aaff", lineWidth: 1, visible: data.showSma,
    lastValueVisible: false, priceLineVisible: false, title: "SMA(50)",
  }});
  smaSeries.setData(data.sma);

  // Crosshair legend (PIIP audit 2026-08, per user request -- Robinhood-style OHLC + DATE
  // readout that follows the cursor, not just the chart library's small built-in axis label,
  // which is exactly how a user caught a real stale-data situation (Friday's session showing
  // pre-market Monday) that the axis label alone didn't make obvious enough).
  const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  function fmtDateTime(utcSeconds) {{
    // Deliberately uses the UTC getters, not local ones -- these timestamps are ET wall-clock
    // values reinterpreted as UTC (see _et_seconds() in app.py) specifically so they display
    // correctly regardless of the viewer's own browser timezone; local getters would apply the
    // browser's own offset on TOP of that and silently shift the displayed time.
    const d = new Date(utcSeconds * 1000);
    const hh = String(d.getUTCHours()).padStart(2, "0");
    const mi = String(d.getUTCMinutes()).padStart(2, "0");
    return `${{MONTHS[d.getUTCMonth()]}} ${{d.getUTCDate()}}, ${{d.getUTCFullYear()}} · ${{hh}}:${{mi}} ET`;
  }}
  function updateLegend(param) {{
    let bar = null, vol = null;
    if (param && param.time && param.seriesData) {{
      bar = param.seriesData.get(candleSeries);
      vol = volumeSeries ? param.seriesData.get(volumeSeries) : null;
    }}
    if (!bar && data.candles.length) {{
      bar = data.candles[data.candles.length - 1];
      vol = data.volumes.length ? data.volumes[data.volumes.length - 1] : null;
    }}
    if (!bar) {{ legend.textContent = "No data"; return; }}
    const upDown = bar.close >= bar.open ? "#79ed8e" : "#ff8080";
    legend.innerHTML = `<b>${{ticker}}</b> &nbsp; ${{fmtDateTime(bar.time)}} &nbsp; `
      + `O <span style="color:${{upDown}}">${{bar.open.toFixed(2)}}</span> `
      + `H <span style="color:${{upDown}}">${{bar.high.toFixed(2)}}</span> `
      + `L <span style="color:${{upDown}}">${{bar.low.toFixed(2)}}</span> `
      + `C <span style="color:${{upDown}}">${{bar.close.toFixed(2)}}</span>`
      + (vol ? ` &nbsp; Vol ${{Math.round(vol.value).toLocaleString()}}` : "");
  }}
  chart.subscribeCrosshairMove(updateLegend);
  updateLegend(null);

  // Restore whatever pan/zoom position the viewer left this chart at (PIIP audit 2026-08, per
  // user request -- the fragment reruns every 30s with fresh data, which was rebuilding the
  // iframe from scratch each time and silently snapping any manual zoom/pan back to the default
  // view). st.iframe's srcdoc has no sandbox attribute (confirmed via AppTest -- the rendered
  // proto carries only srcdoc/scrolling, no sandbox), so per the HTML spec it inherits the
  // parent page's origin and localStorage is real, persistent storage here, not a sandboxed
  // no-op -- verified directly against the running app via a real CDP-driven browser session,
  // not just AppTest. Keyed by chart_id (ticker+timeframe already baked in), with a 20h
  // freshness cap so a stale range from a prior session/day -- whose absolute time coordinates
  // no longer overlap today's data at all -- doesn't get silently restored into an empty-looking
  // view.
  const rangeKey = "piip_chart_range_{chart_id}";

  // Confirmed via a real CDP-driven browser session against the running app (not just AppTest):
  // applying our view synchronously right after setData(), or even a requestAnimationFrame tick
  // later, wasn't reliably late enough -- something in the chart's own init/layout sequence
  // (autoSize:true measures the container asynchronously) could still overwrite our explicit
  // setVisibleRange() call moments later. Deferring our own view-setting until OUR OWN
  // ResizeObserver reports the container's real, settled width -- and only subscribing to save
  // pan/zoom changes after that -- verified reliable across multiple real 30s fragment
  // auto-refresh cycles against the live server, restoring an exact realistic pan position every
  // time. A short setTimeout fallback covers the case where ResizeObserver never fires.
  let viewApplied = false;
  function applyInitialView() {{
    if (viewApplied) return;
    viewApplied = true;

    // Saving/restoring by LOGICAL range (bar index), not time range, per TradingView's own
    // documented pattern for preserving a chart's view across repeated setData() calls. Time-
    // based setVisibleRange({{from,to}}) is unreliable specifically for a view that extends into
    // "whitespace" past the last real bar -- which is exactly our situation, both the default
    // extend-to-now view and any user pan that leaves margin on the right -- because the library
    // re-derives how much of that whitespace to actually show each time setData() runs, so the
    // rendered width visibly changes refresh to refresh even with byte-identical {{from,to}}
    // values (confirmed: this is what a real screenshot comparison across a refresh showed,
    // AFTER shiftVisibleRangeOnNewBar was already ruled out as the cause). Logical range doesn't
    // have this problem -- a logical index of e.g. lastBarIndex+10 means "10 bars of whitespace
    // past the last bar" consistently, regardless of how much real data now exists there.
    let restoredLogicalRange = null;
    try {{
      const saved = JSON.parse(localStorage.getItem(rangeKey) || "null");
      if (saved && saved.logicalRange && (Date.now() - saved.savedAt) < 20 * 3600 * 1000) {{
        restoredLogicalRange = saved.logicalRange;
      }}
    }} catch (e) {{}}

    // Default view extends through the actual current moment (PIIP audit 2026-08, per user
    // request), not just a tight fit around whatever candles exist -- fitContent() alone made
    // stale data (e.g. Friday's last session, viewed pre-market Monday) look like a normal,
    // live session since the chart filled edge-to-edge with old bars. Now the right edge always
    // sits at "now," so a gap between the last real candle and the edge is immediately visible
    // whenever the data doesn't reach the present. Only applied when there's no restored
    // pan/zoom position to honor instead. This part stays time-based -- it's recomputed fresh
    // every refresh rather than remembered, so time-range's whitespace quirk doesn't apply here.
    if (restoredLogicalRange) {{
      chart.timeScale().setVisibleLogicalRange(restoredLogicalRange);
    }} else if (data.candles.length > 0) {{
      const firstTime = data.candles[0].time;
      const lastTime = data.candles[data.candles.length - 1].time;
      chart.timeScale().setVisibleRange({{ from: firstTime, to: Math.max(lastTime, data.now) }});
    }} else {{
      chart.timeScale().fitContent();
    }}

    chart.timeScale().subscribeVisibleLogicalRangeChange((logicalRange) => {{
      if (logicalRange) {{
        try {{ localStorage.setItem(rangeKey, JSON.stringify({{ logicalRange: logicalRange, savedAt: Date.now() }})); }} catch (e) {{}}
      }}
    }});
  }}

  const ro = new ResizeObserver(() => {{
    if (container.clientWidth > 0) {{ applyInitialView(); }}
  }});
  ro.observe(container);
  // Fallback in case ResizeObserver never reports a non-zero width for some reason.
  setTimeout(applyInitialView, 500);
}})();
</script>
"""
    st.iframe(html, height=chart_px + 30)
    st.caption(f"{tf_choice} candles · {len(cdf)} bars · scroll/pinch to zoom, drag to pan — "
              "price auto-fits the visible range · refreshes every 30s with the rest of this page.")


# Fixed display order (per user request): a bull-to-bear spectrum with the two transitional
# states placed where they conceptually sit, NOT the insertion order zdlog.regime_stats() happens
# to return (a python dict groupby, effectively arbitrary). INSUFFICIENT DATA is excluded --
# day_regime() only returns it before enough of today's OWN session has printed, so it's never a
# meaningful multi-day historical bucket to show outcome stats for.
_REGIME_OUTCOME_ORDER = ["BULL CONFIRMED", "BULL DEVELOPING", "TREND WEAKENING", "NEUTRAL / CHOP",
                        "REGIME TRANSITION", "BEAR DEVELOPING", "BEAR CONFIRMED"]
_REGIME_OUTCOME_HORIZONS = [("fwd_5m", "+5m"), ("fwd_15m", "+15m"), ("fwd_30m", "+30m"),
                           ("fwd_60m", "+60m"), ("fwd_eod", "+EOD")]


def _render_regime_outcome_table(stats: dict):
    """'What's happened before' headline table (per user request, modeled directly on a proposal
    the user brought from ChatGPT): for each Day Regime state, how often price was higher some
    time later, whenever there's a real sample to say so from -- zdlog.regime_stats() already
    computes exactly this (grouped by the day_regime value active AT SIGNAL TIME, forward-only,
    same-day-only horizons, see that function's own leakage discipline), this only renders it as a
    proper table instead of the flat text list it used to be.

    Explicitly does NOT show a percentage for any state/horizon below zdlog.regime_stats()'s own
    min_sample=30 threshold -- shows an insufficient-sample marker instead, per the user's own
    instruction not to present conclusions before the sample exists. Collection only started
    2026-08-15 -- expect most or all cells to read INSUFFICIENT SAMPLE for a while yet; that's
    correct honesty, not a bug."""
    # PIIP audit 2026-08 (per user request, same treatment as the Day Regime relationship text):
    # the methodology paragraph moved from a permanent multi-line caption into a hover tooltip on
    # the header itself -- only the genuinely at-a-glance fact (the live snapshot count) stays as
    # a visible caption below the table.
    _info_header("📊 What's Happened Before — Historical Outcome by State", (
        "% = positive-return rate at that horizon, only shown once a state/horizon has 30+ "
        "resolved samples (— otherwise, the sample so far is shown in parentheses). N column is "
        "the +5m sample size specifically — longer horizons need more elapsed time to resolve "
        "and always have an equal or smaller sample, shown per-cell when insufficient. Hover any "
        "percentage for avg/median/range. Forward-only, same-trading-day horizons — never uses "
        "data from before the observation. This is collection, not a validated edge — a 55%+ or "
        "45%- reading is colored only to be scannable, not to claim significance."))
    groups = stats.get("groups") or {}
    if not stats.get("total_snapshots"):
        st.caption("No logged snapshots with a recorded price yet — this table fills in as the "
                  "Signal Calibration Log collects real sessions. Not a bug: collection only "
                  "started 2026-08-15.")
        return

    header_cells = "".join(f'<th style="padding:0.4rem 0.6rem;text-align:center;font-size:0.72rem;'
                           f'color:#8b9a9d;border-bottom:1px solid #232b2d">{label}</th>'
                           for _, label in _REGIME_OUTCOME_HORIZONS)
    rows_html = []
    for state in _REGIME_OUTCOME_ORDER:
        horizons = groups.get(state)
        state_color = {"BULL CONFIRMED": "#79ed8e", "BULL DEVELOPING": "#a8e6a1",
                      "BEAR CONFIRMED": "#ff8080", "BEAR DEVELOPING": "#f5a3a3",
                      "NEUTRAL / CHOP": "#87d1ff", "TREND WEAKENING": "#fabf6b",
                      "REGIME TRANSITION": "#fabf6b"}.get(state, "#8b9a9d")
        # "N" column (per user's mockup) uses the +5m sample -- the largest/most-complete count,
        # since longer horizons need more elapsed time to resolve and so always have <= that many
        # observations. Footnote below makes this explicit rather than implying one N applies to
        # every column.
        n5 = horizons.get("fwd_5m", {}).get("n", 0) if horizons else 0
        cells = [f'<td style="padding:0.4rem 0.6rem;text-align:center;font-family:ui-monospace,'
                f'Consolas,monospace;color:#8b9a9d;font-size:0.8rem">{n5}</td>']
        for key, _ in _REGIME_OUTCOME_HORIZONS:
            h = (horizons or {}).get(key)
            if not h or h.get("status") == "INSUFFICIENT SAMPLE":
                needed = h.get("min_needed", 30) if h else 30
                have = h.get("n", 0) if h else 0
                cells.append(f'<td style="padding:0.4rem 0.6rem;text-align:center;color:#4a5254;'
                            f'font-size:0.78rem" title="Insufficient sample: {have}/{needed} needed">'
                            f'— <span style="font-size:0.65rem">({have}/{needed})</span></td>')
            else:
                pct = h["positive_rate_pct"]
                avg_r, med_r, worst_r, best_r, n_r = (h["avg_return_pct"], h["median_return_pct"],
                                                      h["worst_pct"], h["best_pct"], h["n"])
                pct_color = "#79ed8e" if pct >= 55 else "#ff8080" if pct <= 45 else "#e8ecec"
                cells.append(f'<td style="padding:0.4rem 0.6rem;text-align:center;font-weight:700;'
                            f'color:{pct_color}" title="avg {avg_r:+.3f}% · median '
                            f'{med_r:+.3f}% · range [{worst_r:+.2f}%, '
                            f'{best_r:+.2f}%] · N={n_r}">{pct:.0f}%</td>')
        rows_html.append(
            f'<tr><td style="padding:0.4rem 0.6rem;font-weight:700;color:{state_color};'
            f'font-size:0.82rem;white-space:nowrap">{state}</td>{"".join(cells)}</tr>')

    st.markdown(
        f'<table style="width:100%;border-collapse:collapse;margin:0.4rem 0 0.3rem 0;">'
        f'<thead><tr><th style="padding:0.4rem 0.6rem;text-align:left;font-size:0.72rem;'
        f'color:#8b9a9d;border-bottom:1px solid #232b2d">STATE</th>'
        f'<th style="padding:0.4rem 0.6rem;text-align:center;font-size:0.72rem;color:#8b9a9d;'
        f'border-bottom:1px solid #232b2d" title="Sample size for the +5m column">N</th>'
        f'{header_cells}</tr></thead><tbody>{"".join(rows_html)}</tbody></table>',
        unsafe_allow_html=True)
    st.caption(f"{stats['total_snapshots']:,} total logged snapshots with a recorded price. "
              "Hover the header above for methodology.")


def _compute_premarket_thesis(index_snaps: dict, vix_snap: dict | None, participation: dict,
                              intraday: dict, daily: dict, alignment: dict, now_et: datetime) -> dict | None:
    """SPY Premarket Thesis (deterministic phase) -- builds, persists, and checkpoints the
    thesis. Renders NOTHING itself; see _render_unified_morning_read() for the compact 'Morning
    Read' card this feeds (per the user's chosen 3-column layout, replacing what used to be 3
    separately-bordered boxes with a lot of dead space). Always about SPY specifically, regardless
    of which ticker is selected elsewhere on this page (index_snaps already has SPY/QQQ/IWM/DIA
    fetched from the same batch either way). See iip/premarket_thesis.py's module docstring for
    the full hierarchy/handoff/signal-family design.

    Returns None if SPY's snapshot isn't available yet; otherwise a dict of everything the
    rendering layer needs: spy_snap, vix_snap, thesis (today's live recompute), original (the
    immutable row logged once this morning, or None if logging hasn't succeeded yet), primary
    (whether this should still show as the PRIMARY read per pt.is_primary()), and direction."""
    spy_snap = index_snaps.get("SPY")
    if not spy_snap:
        return None
    qqq_snap, iwm_snap, dia_snap = index_snaps.get("QQQ"), index_snaps.get("IWM"), index_snaps.get("DIA")

    futures = pt_fetch_futures()
    mb = get_macro_batch()
    y10_df = mb.get(macro.YIELD_TICKERS["10Y"])
    y10_close = y10_df["Close"].dropna() if y10_df is not None else None
    y10_chg = (float(y10_close.iloc[-1] - y10_close.iloc[-2])
              if y10_close is not None and len(y10_close) >= 2 else None)
    wti_df = mb.get(macro.OIL_TICKERS["WTI"])
    wti_close = wti_df["Close"].dropna() if wti_df is not None else None
    wti_chg = (round((float(wti_close.iloc[-1]) / float(wti_close.iloc[-2]) - 1) * 100, 3)
              if wti_close is not None and len(wti_close) >= 2 else None)
    or15 = zd.opening_range(intraday.get("SPY"), minutes=15)
    prev_day = zd.previous_day_levels(daily.get("SPY"))
    qqq_rel = (round(qqq_snap["day_change_pct"] - spy_snap["day_change_pct"], 3)
              if qqq_snap and spy_snap else None)

    thesis = pt.build_thesis(spy_snap, qqq_snap, iwm_snap, dia_snap, vix_snap, futures, y10_chg,
                             wti_chg, participation, or15, prev_day, qqq_rel)
    direction = thesis["market_state"]["direction"]

    # Persistence -- idempotent (DB-enforced uniqueness), safe to call every 30s refresh.
    try:
        pt.log_thesis_once("SPY", thesis)
        if thesis["confirmation"]["status"] == "CONFIRMED":
            pt.log_confirmation_event_once("SPY", direction)
    except Exception:
        pass  # best-effort, same discipline as every other calibration log on this page

    original = None
    try:
        original = pt.get_todays_thesis("SPY")
    except Exception:
        pass

    # 10:00 AM checkpoint -- idempotent past the cutoff, only the first attempt after 10:00 ET
    # actually persists (schema-enforced), so calling this every refresh past that time is safe.
    if original and now_et.time() >= dtime(10, 0):
        try:
            confirmation_event = pt.get_confirmation_event("SPY")
            market_open_et = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
            checkpoint = pt.compute_checkpoint(original, spy_snap["last"], direction,
                                               confirmation_event, "10:00 AM", market_open_et)
            pt.log_checkpoint("SPY", "10:00 AM", checkpoint, thesis_id=original.get("_thesis_id"))
        except Exception:
            pass

    primary = pt.is_primary(now_et, alignment["total"])
    return {"spy_snap": spy_snap, "vix_snap": vix_snap, "thesis": thesis, "original": original,
           "primary": primary, "direction": direction}


def _render_thesis_column(pt_data: dict) -> None:
    """Compact Premarket Thesis column for the unified 'Morning Read' card -- full detail (Why?/
    Confirmation/Invalidation while PRIMARY, or logged checkpoints once archived) lives in the
    expander below, not inline."""
    thesis, original, primary = pt_data["thesis"], pt_data["original"], pt_data["primary"]
    ref = original or thesis
    ref_dir = ref["market_state"]["direction"]
    dir_color = {"Bullish": "#79ed8e", "Bearish": "#ff8080", "Neutral": "#87d1ff"}[ref_dir]
    dir_emoji = {"Bullish": "🟢", "Bearish": "🔴", "Neutral": "⚪"}[ref_dir]

    _info_header("🧭 Morning Thesis", (
        "Deterministic-only — direction/confidence/risk from 8 signal families (equity/growth/"
        "small-cap/defensive/volatility/rates/oil/breadth), each bounded so correlated "
        "instruments (SPY+ES, QQQ+NQ, IWM+RTY) don't get counted multiple times and inflate "
        "confidence artificially. Shown as PRIMARY only until Day Regime has real intraday data "
        "(or 10:00 ET, whichever comes first) — then this becomes a reference note, not the "
        "active read. Never a trade signal by itself; see Trade Permission in the detail below. "
        "Immutable once logged each morning — today's original read never gets silently "
        "rewritten, even by later refreshes."), size="0.95rem")
    st.markdown(f'<div style="font-size:1.25rem;font-weight:800;color:{dir_color}">'
               f'{dir_emoji} {ref_dir}</div>'
               f'<div style="font-size:0.78rem;color:#8b9a9d">{ref["market_state"]["confidence"]:.0f}/100'
               f' · {ref["risk_environment"]["level"]}</div>', unsafe_allow_html=True)

    if primary:
        perm = thesis["trade_permission"]
        perm_color = {"WAIT": "#fabf6b", "CALLS_FAVORED": "#79ed8e", "PUTS_FAVORED": "#ff8080",
                     "NO_TRADE": "#8b9a9d"}.get(perm["status"], "#8b9a9d")
        st.markdown(f'<div style="font-size:0.78rem;color:{perm_color};font-weight:700;'
                   f'margin:0.3rem 0 0.4rem">{perm["status"].replace("_", " ")}</div>',
                   unsafe_allow_html=True)
        with st.expander("Why?  ·  Confirmation  ·  Invalidation", expanded=False):
            st.markdown("**Why?** (each row is one bounded signal family, not one ticker)")
            for name, f in thesis["families"].items():
                unit = "pts" if "Rates" in name else "%"
                raw_str = f"{f['raw_pct']:+.2f}{unit}" if f["raw_pct"] is not None else "no data"
                st.write(f"{f['points']:+.1f}  {name}  ({raw_str})")

            conf = thesis["confirmation"]
            st.markdown(f"**Confirmation — {conf['status']}**")
            if conf["checks"]:
                for label, ok in conf["checks"]:
                    st.write(f"{'✓' if ok else '✗'} {label}")
            else:
                st.caption(conf.get("note", "No directional thesis to confirm."))

            inval = thesis["invalidation"]
            st.markdown(f"**Invalidation — {inval['status']}**")
            if inval["checks"]:
                for label, ok in inval["checks"]:
                    st.write(f"{'✓' if ok else '✗'} {label}")
            st.caption(f"Trade Permission: {perm['status'].replace('_', ' ')} — {_esc(perm['reason'])}")
    else:
        checkpoints = []
        try:
            checkpoints = pt.get_checkpoints("SPY")
        except Exception:
            pass
        with st.expander(f"{len(checkpoints)} checkpoint(s) logged" if checkpoints else "No checkpoint yet"):
            st.caption("Handed off to Day Regime once real intraday data existed — this is a "
                      "reference to the morning's original (immutable) read, not a live score.")
            for cp in checkpoints:
                st.write(f"**{cp['label']}** — direction: {cp['directional_accuracy']} · "
                        f"tradeability: {cp['tradeability']} ({cp['tradeability_reason']})")
            if not checkpoints:
                st.caption("First one fires at 10:00 ET.")


def _render_ai_column(pt_data: dict) -> None:
    """Compact AI Premarket Thesis column for the unified 'Morning Read' card -- the optional,
    cost-gated AI interpretation layer on top of the (already immutable) original thesis, 7 fixed
    questions per the ChatGPT-refined spec, ONE call, hard-schema-validated (see
    iip/premarket_ai.py's own docstring for all 3 guardrails). Always reads from `original` -- the
    row logged once this morning -- never a live re-fetch, so the frozen snapshot the AI actually
    sees matches exactly what gets persisted and can be reproduced later. Button/verdict badge
    stays compact; the full 7-section answer lives in the expander below once answered."""
    spy_snap, vix_snap = pt_data["spy_snap"], pt_data["vix_snap"]
    original = pt_data["original"] or pt_data["thesis"]

    _info_header("🧠 AI Premarket Thesis", (
        "ONE Claude call (optional, real spend, ~$0.006/run) answering 7 fixed questions over "
        "the deterministic numbers, plus real overnight news (Catalyst Terminal) and two "
        "SEPARATE historical backtests (never blended). The AI explains and reasons — it never "
        "recomputes a number, and has no say over Trade Permission, which stays entirely "
        "code-computed and is shown in the detail below regardless of what the AI concludes. "
        "Every response is schema-validated before display; a malformed response shows INVALID, "
        "never a silent best-effort guess."), size="0.95rem")

    if not os.getenv("ANTHROPIC_API_KEY"):
        st.caption("Add an Anthropic key to unlock — see the sidebar for other AI sections.")
        return

    ai_output = original.get("_ai_output")
    direction = original["market_state"]["direction"]
    risk_level = original["risk_environment"]["level"]
    # TODAY's own similarity dimensions -- computed from the SAME already-fetched spy_snap/
    # vix_snap the deterministic thesis itself used, the SAME functions the backtest applies to
    # every historical day, so "similar to today" actually compares like with like instead of
    # matching on direction/risk alone (gap_bkt/trend_align/vix_bkt left as None would silently
    # widen the similarity pool to almost everything, undermining the whole point of the
    # similarity-ingredients transparency requirement).
    gap_bkt = pbt.gap_bucket(spy_snap.get("day_change_pct")) if spy_snap else None
    trend_align = (pbt.trend_alignment(spy_snap.get("last"), (spy_snap.get("tech") or {}).get("sma50"))
                   if spy_snap else None)
    vix_bkt = pbt.vix_bucket(vix_snap.get("last")) if vix_snap else None

    if ai_output is None:
        if st.button("🧠 Ask AI", key="pt_ai_btn"):
            with st.spinner("Building snapshot + running historical backtests…"):
                try:
                    releases = get_econ_releases()
                    tier1 = pt_backtest_tier1(direction, risk_level, gap_bkt, trend_align)
                    tier2 = pt_backtest_tier2(direction, gap_bkt, trend_align, vix_bkt)
                    snapshot = pai.build_snapshot(spy_snap, original, releases, tier1, tier2,
                                                  datetime.now(ZoneInfo("America/New_York")))
                    client = agents.LLMClient(dry_run=False)
                    result = pai.ask_ai_thesis(snapshot, client)
                    methodology_version = tier1.get("methodology_version") or pbt.METHODOLOGY_VERSION
                    pt.log_thesis_ai_output("SPY", snapshot, result, methodology_version)
                    if result["_valid"]:
                        st.toast(f"AI thesis done — ${result.get('_cost', 0):.4f} spent.")
                    else:
                        st.toast("AI response failed validation — see below.")
                    st.rerun()
                except Exception as e:
                    st.error(f"AI call failed: {e}")
        return

    # Already answered this morning -- render the immutable stored result, never re-call.
    if not ai_output.get("_valid"):
        st.markdown('<div style="font-size:1.1rem;font-weight:800;color:#ff8080">INVALID</div>',
                   unsafe_allow_html=True)
        with st.expander("Why?"):
            st.caption(ai_output.get("_reason", "unknown"))
            if ai_output.get("_dry_run"):
                st.caption("(Dry-run mode — PIIP_LIVE_LLM isn't set to 1, so no real call was made.)")
        return

    synth = ai_output["synthesis"]
    verdict_color = {"SUPPORTS": "#79ed8e", "CONTRADICTS": "#ff8080",
                    "MIXED": "#fabf6b", "INSUFFICIENT": "#8b9a9d"}.get(synth["verdict"], "#8b9a9d")
    st.markdown(f'<div style="font-size:1.25rem;font-weight:800;color:{verdict_color}">'
               f'{_esc(synth["verdict"])}</div>'
               f'<div style="font-size:0.78rem;color:#8b9a9d">${ai_output.get("_cost", 0):.4f} spent</div>',
               unsafe_allow_html=True)

    with st.expander("Full AI read"):
        st.write(synth["analysis"])
        st.markdown(f"**Failure mode:** {_esc(synth['failure_mode'])}")

        for label, key in [("Trend Context", "trend_context"), ("Signal Conviction", "signal_conviction"),
                          ("Risk Environment (AI read)", "risk_environment"),
                          ("Catalyst Risk", "catalyst_risk"), ("Overnight News", "overnight_news"),
                          ("Historical Evidence", "historical_evidence")]:
            st.markdown(f"**{label}**")
            st.write(ai_output[key]["analysis"])

        # Historical similarity ingredients -- shown explicitly per the "audit why a day was
        # classified as similar" requirement, never just the HIGH/MEDIUM/LOW label alone.
        snap = original.get("_ai_snapshot") or {}
        st.markdown("**Historical similarity ingredients**")
        for tlabel, tier in (("Tier 1 (intraday, ~60d)", snap.get("historical_tier1") or {}),
                             ("Tier 2 (daily proxy, ~2y)", snap.get("historical_tier2") or {})):
            st.markdown(f"*{tlabel}* — {tier.get('similarity_label', 'N/A')}, "
                       f"N={tier.get('n', 0)}, methodology {tier.get('methodology_version', '?')}")
            for dim, val in (tier.get("similarity_ingredients") or {}).items():
                if dim in ("min_dims_matched", "of_dims"):
                    continue
                st.write(f"· {dim.replace('_', ' ').title()}: {val}")

        # Trade State -- ALWAYS from the deterministic engine, never restated by the AI (the
        # validator actively rejects a response that tries to include one of its own).
        perm = original["trade_permission"]
        perm_color = {"WAIT": "#fabf6b", "CALLS_FAVORED": "#79ed8e", "PUTS_FAVORED": "#ff8080",
                     "NO_TRADE": "#8b9a9d"}.get(perm["status"], "#8b9a9d")
        st.markdown(f'<div style="padding:0.4rem 0.7rem;background:{perm_color}1a;border:1px '
                   f'solid {perm_color};border-radius:6px;margin-top:0.5rem;font-size:0.85rem">'
                   f'<b style="color:{perm_color}">Trade State (deterministic, not the AI): '
                   f'{perm["status"].replace("_", " ")}</b></div>', unsafe_allow_html=True)


def _render_regime_column(regime: dict, ticker: str) -> None:
    """Compact Day Regime column for the unified 'Morning Read' card -- "what kind of trading
    environment is this right now," reasons live in the expander below."""
    regime_colors = {"BULL CONFIRMED": "#79ed8e", "BULL DEVELOPING": "#a8e6a1",
                     "BEAR CONFIRMED": "#ff8080", "BEAR DEVELOPING": "#f5a3a3",
                     "NEUTRAL / CHOP": "#87d1ff", "TREND WEAKENING": "#fabf6b",
                     "REGIME TRANSITION": "#fabf6b", "INSUFFICIENT DATA": "#8b9a9d"}
    regime_color = regime_colors.get(regime["state"], "#8b9a9d")
    _info_header(f"📍 Day Regime — {ticker}", (
        "The live directional state right now, synthesized from Trend State (iip/timeframe.py) "
        "+ Trend Integrity — a reuse of signals already computed elsewhere on this page, not a "
        "new independent read. Thresholds (10 min / 50 integrity / 40 integrity / 60 reversal "
        "pressure) are a first-pass guess, flagged for calibration once real sessions accumulate "
        "in the Signal Calibration Log below."), size="0.95rem")
    age = f'{regime["trend_age_minutes"]:.0f} min old' if regime.get("trend_age_minutes") else " "
    st.markdown(f'<div style="font-size:1.25rem;font-weight:800;color:{regime_color}">'
               f'{regime["state"]}</div>'
               f'<div style="font-size:0.78rem;color:#8b9a9d">{age}</div>', unsafe_allow_html=True)
    with st.expander("Why this regime?", expanded=False):
        for r in regime["reasons"]:
            st.write(f"• {r}")


def _render_unified_morning_read(pt_data: dict | None, regime: dict, ticker: str) -> None:
    """The 'Morning Read' card -- Premarket Thesis / AI Premarket Thesis / Day Regime side by
    side as one instrument panel, per the user's chosen layout (Option C from the reviewed
    mockups) -- replacing what used to be 3 separately-bordered boxes, each with its own
    header/border eating vertical space and the AI box mostly empty before it's ever asked. Each
    column stays compact; full detail is one click away in that column's own expander."""
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        if pt_data:
            with col1:
                _render_thesis_column(pt_data)
            with col2:
                _render_ai_column(pt_data)
        with col3:
            _render_regime_column(regime, ticker)


@st.fragment(run_every=30)
def _render_zero_dte(ticker: str):
    intraday = zd_fetch_intraday()
    daily = zd_fetch_daily()

    def snaps_for(tickers):
        return {tk: zd.ticker_snapshot(tk, intraday, daily) for tk in tickers}

    index_snaps = snaps_for(zd.INDEX_TICKERS)
    mega_snaps = snaps_for(zd.MEGA_CAPS)
    sector_snaps = snaps_for(list(zd.SECTOR_ETFS.values()))
    vix_snap = zd.ticker_snapshot("^VIX", intraday, daily)
    # NVDA is a selectable ticker on this page too (PIIP audit 2026-08, Batch 3) but isn't one of
    # the 4 INDEX_TICKERS -- it's already fetched as part of MEGA_CAPS, so no new network call,
    # just a second place to look it up.
    snap = index_snaps.get(ticker) or mega_snaps.get(ticker)

    # PIIP audit 2026-08 (state-architecture review, Phase 7): data-freshness policy, stated
    # explicitly rather than left implicit. This page's own quote is cached 25s (zd_fetch_intraday/
    # zd_fetch_daily); Ticker Page/Research/Paper/Screener fetch the SAME underlying tickers
    # through a SEPARATE, slower 900s cache (load_prices/load_chain) -- confirmed live these can
    # legitimately disagree by up to 15 minutes if viewed side by side at the same moment. That's
    # a deliberate tradeoff (unifying every page onto a 25s poll would multiply yfinance request
    # volume across pages users may have open simultaneously, risking the rate-limiting this
    # project has already hit before -- see fetch_intraday_batch()'s own docstring), not an
    # oversight, but it needs to be visible rather than silently assumed.
    st.caption(f"⏱️ As of {datetime.now().strftime('%H:%M:%S')} — refreshes every 30s. This page's "
              "own quote is cached ~25s; Ticker Page/Research/Paper use a separate, slower ~15min "
              "cache for the same tickers, so they may briefly show a different price.")
    if not snap:
        st.warning(f"No live data for {ticker} right now (market closed, or a data gap). "
                   "Try again during market hours.")
        return

    breadth = zd.breadth_score(index_snaps, mega_snaps, sector_snaps)
    # PIIP audit 2026-08 (state-architecture review, Phase 3): participation_state() replaces the
    # old double-counted vote (mega-caps counted individually AND via the sector ETFs that already
    # contain them) with a 3-bucket (Index/Mega-Cap/Sector) synthesis. breadth_score()'s own raw
    # green/red counts are unchanged and still shown directly below (sector/mega rows) -- this is
    # a new interpretation layer on top, not a replacement of the underlying data.
    participation = zd.participation_state(breadth)

    # Time-of-day-adjusted Relative Volume (PIIP audit 2026-08, Batch 2 / Phase 7; promoted to the
    # PRIMARY rvol read in the state-architecture review, Phase 4) -- moved earlier in this
    # function (was computed much later) so ticker_health()/confluence_score() can use it instead
    # of the cruder full-prior-day-average `rel_volume`, which reads artificially low before the
    # close. Needs a slower, separately-cached historical fetch (see zd_fetch_historical_5m, 4h
    # TTL) that can fail independently of the live 1m fetch -- every consumer below falls back to
    # the old `rel_volume` field automatically when this is unavailable.
    historical_5m = zd_fetch_historical_5m(ticker)
    tod_rel_vol = zd.time_of_day_relative_volume(historical_5m, intraday.get(ticker))

    health = zd.ticker_health(snap, tod_rel_vol)
    momentum = zd.momentum_engine(intraday.get(ticker))
    bias = zd.market_bias(snap, breadth, vix_snap)
    sector = zd.sector_health(sector_snaps)
    mega = zd.mega_cap_health(mega_snaps)

    chain = zd_fetch_chain(ticker)
    opt = zd.options_health(chain, snap["last"]) if chain else None
    dealer = zd.dealer_positioning(chain, snap["last"]) if chain else None
    put_call_ratio = None
    if chain:
        calls_oi = float(chain["calls"]["openInterest"].fillna(0).sum())
        puts_oi = float(chain["puts"]["openInterest"].fillna(0).sum())
        put_call_ratio = round(puts_oi / calls_oi, 2) if calls_oi else None

    reversal = zd.reversal_engine(momentum, snap["tech"], snap)
    entry = zd.entry_quality(bias, health, momentum, opt, dealer, breadth)
    confidence = zd.trade_confidence(bias, entry, opt)

    # Multi-timeframe alignment (PIIP audit 2026-08, Option B): resamples the SAME 1m bars already
    # fetched above into 5m/15m/30m, no new network call, plus the existing daily EMA alignment as
    # a "Daily" timeframe — see iip/timeframe.py's module docstring for why 60m isn't attempted.
    # Trend state persists across this fragment's 30s reruns via session_state, same edge-trigger
    # pattern the Exit Quality alert below already uses.
    tf_snapshot = tf.multi_timeframe_snapshot(intraday.get(ticker), snap["tech"])
    alignment = tf.timeframe_alignment(tf_snapshot)
    trend_state_key = f"zd_trend_state_{ticker}"
    trend_state = tf.update_trend_state(st.session_state.get(trend_state_key), alignment)
    st.session_state[trend_state_key] = trend_state

    # Trend age (PIIP audit 2026-08, Batch 2): how long the CONFIRMED state has held, timestamped
    # only when it actually changes -- same edge-trigger pattern as the Exit Quality alert below.
    # Feeds day_regime()'s Developing-vs-Confirmed tiering.
    trend_since_key = f"zd_trend_state_since_{ticker}"
    if trend_state["changed"] or trend_since_key not in st.session_state:
        st.session_state[trend_since_key] = datetime.now()
    trend_age_minutes = (datetime.now() - st.session_state[trend_since_key]).total_seconds() / 60

    confluence = zd.confluence_score(bias, participation, momentum, reversal, snap, alignment,
                                     tod_rel_vol)
    prev_day = zd.previous_day_levels(daily.get(ticker))
    or15 = zd.opening_range(intraday.get(ticker), minutes=15)

    # NVDA Relative Strength (PIIP audit 2026-08, Option C): reuses the mega-cap snapshot/intraday
    # bars already batched above -- NVDA is already fetched as part of MEGA_CAPS -- no new call.
    # Skipped entirely when NVDA itself is the selected ticker (Batch 3: NVDA is now selectable on
    # this page) -- comparing NVDA against NVDA is meaningless, not just an edge case to degrade.
    is_nvda_focus = (ticker == "NVDA")
    nvda_rs = None if is_nvda_focus else \
        zd.nvda_relative_strength(ticker, snap, mega_snaps.get("NVDA"), intraday.get("NVDA"))

    # NVDA vs all 4 comparison tickers + leadership acceleration (PIIP audit 2026-08, Batch 2 /
    # Phases 16-17). SMH/SOXX are new tickers in the batched fetch (see NVDA_COMPARISON_TICKERS);
    # SPY/QQQ reuse index_snaps. Acceleration is computed straight from today's already-fetched
    # intraday bars, no session_state needed. Both still make sense when NVDA IS the focus ticker
    # (NVDA vs the broader indices/semis ETFs is exactly the useful read there), so NOT gated on
    # is_nvda_focus the way the single-ticker nvda_rs row above is.
    nvda_compare_snaps = snaps_for(zd.NVDA_COMPARISON_TICKERS)
    nvda_rs_multi = zd.nvda_relative_strength_multi(nvda_compare_snaps, mega_snaps.get("NVDA"),
                                                     intraday.get("NVDA"))
    nvda_rs_accel = None if is_nvda_focus else \
        zd.nvda_relative_strength_acceleration(intraday.get(ticker), intraday.get("NVDA"))

    # Trend Integrity (PIIP audit 2026-08, Batch 1): VWAP crossings + path efficiency are new
    # primitives; Trend Integrity is a synthesis of those plus alignment/confluence already
    # computed above -- no new network calls, all derived from data already in hand.
    crossings = zd.vwap_crossings(intraday.get(ticker))
    efficiency = zd.trend_efficiency(intraday.get(ticker))
    integrity = zd.trend_integrity(alignment, crossings, efficiency, confluence)
    no_edge_reasons = (zd.no_clear_edge_reasons(bias, confluence, crossings, alignment, reversal)
                       if bias["direction_label"] == "No Clear Edge" else [])
    data_quality = zd.data_quality_snapshot(intraday.get(ticker), chain, tf_snapshot)

    # Day Regime (PIIP audit 2026-08, Batch 2 / Phase 1) and Timeframe Sequence (Phase 8) -- both
    # synthesize signals already computed above, no new data.
    regime = zd.day_regime(trend_state, integrity, alignment, reversal, trend_age_minutes)
    tf_sequence = tf.interpret_timeframe_sequence(tf_snapshot)

    # "What Changed?" detail (PIIP audit 2026-08, state-architecture review, Phase 5) and Market
    # DNA (day type) -- both moved EARLIER here (were computed further down, after the signal-log
    # write) per user request to also persist them on every routine 30s snapshot row, not just on
    # regime-transition rows. Pure reorder of two already-existing, already-cheap computations
    # (no new network calls, no new logic) -- everything they need (crossings/efficiency/
    # integrity/participation/alignment/trend_state/trend_age_minutes, and snap/intraday/daily)
    # was already available by this point in the function.
    regime_detail = zd.build_regime_detail(tf_snapshot, snap, crossings, integrity, efficiency,
                                           reversal, participation, alignment=alignment,
                                           trend_state=trend_state,
                                           trend_age_minutes=trend_age_minutes)
    dna = mdna.classify(snap, intraday.get(ticker), daily.get(ticker))

    # PIIP audit 2026-08 (accounting pass): gated on Fresh underlying data -- without this, leaving
    # the page open overnight/over a weekend would log the SAME stale closing price every 30s for
    # hours, since data_quality_snapshot() already correctly flags it Stale but nothing previously
    # stopped the log write itself. Those repeated identical rows would silently corrupt
    # compute_forward_outcomes()/regime_stats() later with fake "0% return over closed-market
    # minutes" data points once real historical depth builds up -- worth fixing now, before it
    # contaminates weeks of collection, not after.
    #
    # `day_type`/`state_age_minutes`/`detail` (PIIP audit 2026-08, state-architecture review,
    # follow-up per user request): now persisted on EVERY logged snapshot, not just regime-
    # transition rows -- these were already being computed every refresh (regime_detail/dna
    # above), just not written to the signal log before. Purely additive to the existing `extra`
    # JSON column (built for exactly this -- "room for fields added later without a schema
    # migration") -- no new table, no new column, no change to any existing field.
    if data_quality["underlying"] == "Fresh":
        try:
            zdlog.log_signal_snapshot(ticker, bias, confidence, entry, momentum, reversal, alignment,
                                      trend_state, spot_price=snap["last"], day_regime=regime["state"],
                                      day_type=dna["label"], state_age_minutes=trend_age_minutes,
                                      detail=regime_detail)
        except Exception:
            pass  # calibration logging is best-effort -- never break the page over a local DB write

    # Regime transition timeline (PIIP audit 2026-08, Batch 3 / Phase 9): edge-triggered, only
    # logs when the Day Regime state actually CHANGES since the last read -- same was_alert/
    # is_alert pattern the Exit Quality alert below already uses. Never one row per 30s refresh.
    # explain_transition() (called at render time in the Regime Timeline section below) diffs
    # each row's detail against the PRIOR logged row's detail -- i.e. "what was true when we
    # entered the state we're now leaving" vs. "what's true now" -- which is the decision-relevant
    # comparison ("why did BULL CONFIRMED become BULL WEAKENING" means vs. when it became
    # CONFIRMED, not vs. the last noisy 30s tick). Recomputed from stored facts every render,
    # never stored as a conclusion, so it can't drift from what was measured.
    regime_prev_key = f"zd_regime_prev_{ticker}"
    prev_regime_state = st.session_state.get(regime_prev_key)
    if prev_regime_state is not None and prev_regime_state != regime["state"]:
        try:
            zdlog.log_regime_transition(ticker, prev_regime_state, regime["state"],
                                        regime["reasons"], regime.get("trend_age_minutes"),
                                        detail=regime_detail)
        except Exception:
            pass
    st.session_state[regime_prev_key] = regime["state"]

    try:
        mdna.log_snapshot(ticker, dna)
    except Exception:
        pass  # logging is best-effort — never break the page over a local DB write


    sector_sorted = sorted(sector["sectors"].items(), key=lambda kv: kv[1]["score"], reverse=True)
    top_movers = sorted(mega["names"].items(), key=lambda kv: abs(kv[1]["day_change_pct"]), reverse=True)

    groups = [
        {"label": "🎯 Trade Readiness", "rows": [
            {"label": "Entry Quality", "value": f"{entry['score']:.0f}/100", "color": _score_color(entry["score"]),
             "context": entry["risk_label"],
             "evidence": [f"{pts:+.1f}  {label}" for label, pts in entry["reasons"].items()]},
        ]},
        {"label": "🧭 Trend Integrity", "rows": [
            {"label": "Trend Integrity", "value": f"{integrity['score']:.0f}/100 {integrity['label']}",
             "color": _score_color(integrity["score"]),
             "context": "How clean and persistent the current trend is — a synthesis of the "
                        "rows below, not a new independent read.",
             "evidence": [f"{pts:+.1f}  {label}" for label, pts in integrity["reasons"].items()]},
            {"label": "Trend Efficiency",
             "value": f"{efficiency['efficiency_pct']:.0f}%" if efficiency else "—",
             "color": _score_color(efficiency["efficiency_pct"]) if efficiency else "#8b9a9d",
             "context": "Net move ÷ total path walked — 100% is a straight line, low % is "
                        "back-and-forth chop covering the same net distance.",
             "evidence": ([f"Net move: {efficiency['net_move']:.3f}",
                          f"Path length: {efficiency['path_length']:.3f}",
                          "Different from market_dna's net_vs_range — this uses the actual "
                          "bar-to-bar path, more sensitive to whipsaw."] if efficiency
                         else ["Needs at least 5 intraday bars."])},
            {"label": "VWAP Crossings",
             "value": f"{crossings['count']}" if crossings else "—",
             "color": ("#79ed8e" if crossings and crossings["count"] <= 2 else
                      "#ff8080" if crossings and crossings["count"] > 4 else
                      "#fabf6b" if crossings else "#8b9a9d"),
             "context": (f"Currently {crossings['current_side']} · last crossing "
                        f"{crossings['minutes_since_last']:.0f} min ago" if crossings and
                        crossings["minutes_since_last"] is not None else
                        crossings["current_side"] if crossings else "Needs at least 5 intraday bars."),
             "evidence": ["Counted against the RUNNING (as-of-that-bar) VWAP, not the final one.",
                         "Not scored as bullish/bearish by count alone — that relationship is "
                         "being collected in the Signal Calibration Log for future validation, "
                         "not assumed here."]},
        ]},
        {"label": "🌐 Market Context", "rows": [
            {"label": "Market Bias", "value": bias["direction_label"], "color": _lean_color(bias["direction_label"]),
             "context": f"CALLS {bias['calls_pct']:.0f} / PUTS {bias['puts_pct']:.0f} · "
                        f"{bias['confidence_pct']:.0f}% conf · {bias['environment']}",
             "evidence": [f"{pts:+.1f}  {label}" for label, pts in bias["reasons"].items()]},
            # PIIP audit 2026-08 (state-architecture review, Phase 3): Participation is the new
            # top-level row -- the 3-bucket (Index/Mega-Cap/Sector) synthesis that fixed the
            # double-counting bug, shown FIRST since it's what confluence_score() actually votes
            # on now. Breadth/Sector/Mega Cap rows below it are unchanged, kept exactly as before
            # so the raw underlying green/red counts are still fully inspectable.
            {"label": "Participation", "value": participation["state"],
             "color": ("#79ed8e" if participation["state"] == "STRONG" else
                      "#a8e6a1" if participation["state"] == "MODERATE" else
                      "#fabf6b" if participation["state"] == "WEAK" else
                      "#ff8080" if participation["state"] == "MIXED" else "#8b9a9d"),
             "context": (f"Avg {participation['avg_signed']:+.0f} across {participation['n_buckets']} "
                        "bucket(s)" if participation["avg_signed"] is not None else
                        "No index/mega-cap/sector data available right now."),
             "evidence": [f"{name}: {v:+.1f}" if v is not None else f"{name}: —"
                         for name, v in participation["buckets"].items()] + [participation["note"]]},
            {"label": "Breadth (proxy)", "value": f"{breadth['score_signed']:+.0f} {breadth['label']}",
             "color": _lean_color(breadth["label"]),
             "context": f"Idx {breadth['indices_green']}\U0001f7e2/{breadth['indices_red']}\U0001f534 · "
                        f"Mega {breadth['mega_caps_green']}\U0001f7e2/{breadth['mega_caps_red']}\U0001f534 · "
                        f"Sec {breadth['sectors_green']}\U0001f7e2/{breadth['sectors_red']}\U0001f534",
             "evidence": [breadth["note"], "Raw underlying counts -- see the 'Participation' row "
                         "above for the de-duplicated 3-bucket interpretation used in Confluence."]},
            {"label": "Sector Health", "value": f"{sector['overall_confirmation_pct']:.0f}%",
             "color": _score_color(sector["overall_confirmation_pct"]),
             "context": f"{sector['confirming']}/{sector['total']} sectors confirming",
             "evidence": [f"{d['score']:.0f}  {name}  ({d['day_change_pct']:+.2f}%)" for name, d in sector_sorted]},
            {"label": "Mega Cap Health", "value": f"{mega['score']:.0f}/100", "color": _score_color(mega["score"]),
             "context": f"{mega['weighted_avg_change_pct']:+.2f}% wtd avg (approx. SPY weights)",
             "evidence": [f"{tk}: {d['day_change_pct']:+.2f}% (weight {d['weight']:.1f})" for tk, d in top_movers]},
            *([{"label": "NVDA Relative Strength",
                "value": nvda_rs["label"] if nvda_rs else "—",
                "color": ("#79ed8e" if nvda_rs and nvda_rs["label"] == "Outperforming" else
                         "#ff8080" if nvda_rs and nvda_rs["label"] == "Underperforming" else
                         "#87d1ff" if nvda_rs else "#8b9a9d"),
                "context": (f"NVDA {nvda_rs['nvda_day_change_pct']:+.2f}% vs {ticker} "
                           f"{nvda_rs['index_day_change_pct']:+.2f}% (spread {nvda_rs['spread_pct']:+.2f}pt)"
                           if nvda_rs else "NVDA data unavailable right now."),
                "evidence": ([nvda_rs["note"],
                             f"NVDA's own intraday momentum: {nvda_rs['nvda_momentum_label'] or '—'}",
                             "A single-stock read, not a market-wide signal — context alongside "
                             "Mega Cap Health, not its own trade signal."] if nvda_rs
                            else ["Needs a live NVDA snapshot — check back during market hours."])},
               {"label": "NVDA RS Acceleration",
                "value": nvda_rs_accel["trend"] if nvda_rs_accel else "—",
                "color": ("#79ed8e" if nvda_rs_accel and nvda_rs_accel["trend"] == "Accelerating" else
                         "#ff8080" if nvda_rs_accel and nvda_rs_accel["trend"] == "Fading" else
                         "#87d1ff" if nvda_rs_accel else "#8b9a9d"),
                "context": (f"{nvda_rs_accel['readings']['30m_ago']:+.2f} → "
                           f"{nvda_rs_accel['readings']['15m_ago']:+.2f} → "
                           f"{nvda_rs_accel['readings']['now']:+.2f} pt"
                           if nvda_rs_accel and nvda_rs_accel["trend"] != "Insufficient session history"
                           else "Needs at least 30 minutes of today's session for both NVDA and "
                                f"{ticker}."),
                "evidence": ["Is NVDA's lead over the selected index widening (Accelerating), holding "
                            "(Stable), or narrowing (Fading) — from 3 actual point-in-time reads, "
                            "not session-state accumulation.",
                            "Whether accelerating relative strength predicts anything about what "
                            "happens next hasn't been validated — this is a descriptive read only."]}]
              if not is_nvda_focus else []),
            {"label": "NVDA vs Indices/Semis ETFs",
             "value": f"{sum(1 for r in nvda_rs_multi.values() if r and r['label'] == 'Outperforming')}"
                      f"/{len(nvda_rs_multi)} Outperforming" if nvda_rs_multi else "—",
             "color": "#87d1ff",
             "context": "NVDA's day-change spread vs SPY, QQQ, SMH, and SOXX, each shown separately.",
             "evidence": [f"vs {tk}: {r['label']} ({r['spread_pct']:+.2f}pt) — {r['note']}"
                         for tk, r in nvda_rs_multi.items() if r]
                        or ["Needs a live NVDA snapshot — check back during market hours."]},
        ]},
        {"label": f"\U0001f4d0 Levels — {ticker}", "rows": [
            {"label": "VWAP", "value": f"${snap['vwap']:.2f}",
             "color": "#79ed8e" if snap["pct_from_vwap"] >= 0 else "#ff8080",
             "context": f"{snap['pct_from_vwap']:+.2f}% from VWAP — institutional fair value for the day",
             "evidence": [f"Last: ${snap['last']:.2f}", f"VWAP: ${snap['vwap']:.2f}",
                         f"Distance: {snap['pct_from_vwap']:+.2f}%"]},
            # PIIP audit 2026-08 (state-architecture review, Phase 4): the time-of-day-adjusted
            # RVOL is now the PRIMARY row (was second, below the cruder metric) -- it's what
            # ticker_health()/confluence_score() now actually use for scoring too (see app.py
            # above), so the displayed default matches what's driving the numbers elsewhere on
            # this page. Label states exactly what it measures ("vs expected volume by this time
            # of day"), never implying a plain daily-average comparison it isn't.
            {"label": "Relative Volume",
             "value": f"{tod_rel_vol['ratio']:.2f}x" if tod_rel_vol and tod_rel_vol["ratio"] else "—",
             "color": ("#79ed8e" if tod_rel_vol and tod_rel_vol["ratio"] and tod_rel_vol["ratio"] > 1.3 else
                      "#ff8080" if tod_rel_vol and tod_rel_vol["ratio"] and tod_rel_vol["ratio"] < 0.7 else
                      "#87d1ff" if tod_rel_vol else "#8b9a9d"),
             "context": (f"{tod_rel_vol['ratio']:.2f}x vs EXPECTED volume by this point in the "
                        f"session ({tod_rel_vol['sample_days']} historical trading days)" if tod_rel_vol
                        else "Needs a successful historical 5m-bar fetch — falling back to the "
                             "full-day-average row below."),
             "evidence": ([f"Actual volume so far: {tod_rel_vol['actual_volume']:,.0f}",
                          f"Expected by now (historical avg): {tod_rel_vol['expected_volume']:,.0f}",
                          f"Method: {tod_rel_vol['method']}",
                          "Time-of-day-adjusted -- compares against the historical average AT "
                          "THIS SAME POINT in the session, not a full-day figure, so it doesn't "
                          "read artificially low before the close."]
                         if tod_rel_vol else ["Historical 5m-bar fetch unavailable right now — "
                                             "the full-day-average row below still works."])},
            {"label": "Relative Volume (full-day avg)",
             "value": f"{snap['rel_volume']:.2f}x" if snap.get("rel_volume") is not None else "—",
             "color": ("#79ed8e" if not tod_rel_vol and health["volume_label"] == "Strong" else
                      "#ff8080" if not tod_rel_vol and health["volume_label"] == "Light" else "#8b9a9d"),
             "context": "Fallback metric — vs a full prior 20-day average, reads low before the "
                       "close. The row above (time-of-day-adjusted) is the primary read when available.",
             "evidence": [f"Today's volume: {snap['today_volume']:,.0f}",
                         f"20-day avg: {snap['avg20_volume']:,.0f}" if snap.get("avg20_volume") else "20-day avg: —",
                         "NOT time-of-day adjusted — a 2x reading here at 11am reads the same as "
                         "2x at 3:45pm even though far less of the day's volume has happened yet."]},
            {"label": "Previous Day",
             "value": f"C ${prev_day['close']:.2f}" if prev_day else "—",
             "color": (("#79ed8e" if snap["last"] >= prev_day["close"] else "#ff8080") if prev_day else "#8b9a9d"),
             "context": (f"H ${prev_day['high']:.2f} · L ${prev_day['low']:.2f} ({prev_day['date']})" if prev_day
                        else "Not enough daily history yet."),
             "evidence": ([f"High: ${prev_day['high']:.2f}", f"Low: ${prev_day['low']:.2f}",
                          f"Close: ${prev_day['close']:.2f}", f"Date: {prev_day['date']}"] if prev_day
                         else ["Needs at least one prior trading day of daily bars."])},
            {"label": "Opening Range (15m)",
             "value": or15["status"] if or15 else "—",
             "color": (("#79ed8e" if or15["status"] == "Above breakout" else
                       "#ff8080" if or15["status"] == "Below breakdown" else "#87d1ff") if or15 else "#8b9a9d"),
             "context": (f"H ${or15['high']:.2f} · L ${or15['low']:.2f} (first {or15['minutes']} min)" if or15
                        else "Available once the session's first 15 minutes have printed."),
             "evidence": ([f"Range high: ${or15['high']:.2f}", f"Range low: ${or15['low']:.2f}",
                          f"Current: ${or15['last']:.2f}",
                          "A breakout on strong relative volume carries more weight than one on light volume."]
                         if or15 else ["Needs today's opening bars — check back after the first 15 minutes."])},
        ]},
        {"label": f"\U0001f4c8 Price & Momentum — {ticker}", "rows": [
            {"label": f"{ticker} Health", "value": f"{health['score']:.0f}/100", "color": _score_color(health["score"]),
             "context": f"{health['trend_label']} · {health['momentum_label']} · {health['volume_label']}",
             "evidence": [f"{pts:+.1f}  {label}" for label, pts in health["reasons"].items()]},
            # PIIP audit 2026-08 (state-architecture review, Phase 6): momentum_engine() reads the
            # last 5 vs. prior 5 of whatever bars it's given -- 1m bars here, so specifically a
            # ~5-minute window -- and the label used to just say "Momentum" with the window buried
            # only in the expanded evidence. Now stated in both the label AND the main context
            # line, so it can't be mistaken for a longer-window read (see the "5m Momentum" /
            # "15m Momentum" / etc. rows in the Timeframes & Trend tab below for the LABELED
            # multi-timeframe versions of this same calculation shape).
            {"label": "Momentum (~5min)",
             "value": f"{momentum['continuation_score_pct']:.0f}" if momentum else "—",
             "color": _score_color(momentum["continuation_score_pct"]) if momentum else "#8b9a9d",
             "context": (f"{momentum['velocity_label']} · {momentum['acceleration_label']} — "
                        "last 5 one-minute bars vs. the prior 5" if momentum
                        else "Not enough intraday bars yet."),
             "evidence": ([f"Velocity: {momentum['velocity_pct']:+.3f}% (last 5 one-minute bars, ~5 min)",
                          f"Acceleration: {momentum['acceleration_pct']:+.3f}% (velocity vs. the "
                          "5 one-minute bars before that, ~5-10 min ago)",
                          "This is a heuristic score, not a calibrated probability — it has not "
                          "been validated against historical outcomes.",
                          "For longer windows, see 5m/15m/30m/Daily Momentum in the Timeframes & "
                          "Trend tab — those resample to real 5/15/30-minute bars first, this row "
                          "does not."]
                         if momentum else ["Needs at least 10 intraday bars — check back once the market's "
                                           "been open a little longer."])},
            {"label": "Reversal Pressure", "value": f"{reversal['reversal_pressure_score']:.0f}",
             "color": _score_color(100 - reversal["reversal_pressure_score"]),
             "context": f"Strength {reversal['strength']:.0f} · {reversal['exhaustion_label']} exhaustion",
             "evidence": ["Does NOT predict tops — estimates whether the current move is "
                         "strengthening or weakening.",
                         "A heuristic score, not a calibrated probability of reversal."]},
        ]},
        {"label": "🕰️ Multi-Timeframe Alignment", "rows": [
            *[{"label": f"{label} Momentum" if label != "Daily" else "Daily Trend",
               "value": r["direction"] if r["available"] else "—",
               "color": ("#79ed8e" if r.get("direction") == "Bullish" else
                        "#ff8080" if r.get("direction") == "Bearish" else
                        "#87d1ff" if r.get("available") else "#8b9a9d"),
               "context": (f"Score {r['continuation_score_pct']:.0f} · {r['bars_used']} {label} bars"
                          if r["available"] and "continuation_score_pct" in r else
                          r.get("note", "") if r["available"] else r["note"]),
               "evidence": ([f"Velocity: {r['velocity_pct']:+.3f}%", f"Acceleration: {r['acceleration_pct']:+.3f}%",
                            "Heuristic score, not a calibrated probability."] if "velocity_pct" in r
                           else [r.get("note", "Existing daily EMA20/50/200 alignment, reused as the "
                                              "session's broader trend context.")])}
              for label, r in tf_snapshot.items()],
            {"label": "Timeframe Alignment",
             "value": f"{alignment['agree']}/{alignment['total']} {alignment['aligned_direction']}"
                      if alignment["total"] else "—",
             "color": _lean_color(alignment["aligned_direction"]) if alignment["total"] else "#8b9a9d",
             "context": alignment["note"],
             "evidence": [f"{lbl}: {d} {'✓' if ok else '✗'}" for lbl, d, ok in alignment["checks"]]
                        or ["No timeframe has enough of today's session printed yet."]},
            {"label": "Timeframe Sequence", "value": tf_sequence["interpretation"],
             "color": ("#79ed8e" if "BULL" in tf_sequence["interpretation"] else
                      "#ff8080" if "BEAR" in tf_sequence["interpretation"] else
                      "#fabf6b" if "TRANSITION" in tf_sequence["interpretation"] else "#87d1ff"),
             "context": tf_sequence["note"],
             "evidence": [f"{lbl}: {d}" for lbl, d in tf_sequence["sequence"]]
                        or ["Not enough available timeframes yet."]},
            {"label": "Trend State", "value": trend_state["state"],
             "color": ("#79ed8e" if trend_state["state"] == "Uptrend" else
                      "#ff8080" if trend_state["state"] == "Downtrend" else
                      "#87d1ff" if trend_state["state"] == "Range / Chop" else "#8b9a9d"),
             "context": (f"Pending: {trend_state['pending']} ({trend_state['pending_count']}/"
                        f"{trend_state['confirm_reads']} confirms)" if trend_state["pending"] != trend_state["state"]
                        else "Confirmed"),
             "evidence": [f"Requires {trend_state['confirm_reads']} consecutive 30s reads agreeing "
                         "before the state changes — avoids flipping on single-bar noise.",
                         f"Latest raw candidate this read: {trend_state['candidate']}"]},
        ]},
        {"label": "⚙️ Options & Dealer", "rows": [
            {"label": "Options Health",
             "value": f"{opt['execution_quality']:.0f}/100" if opt else "—",
             "color": _score_color(opt["execution_quality"]) if opt else "#8b9a9d",
             "context": (f"Liquidity {opt['liquidity_score']:.0f} · {opt['spread_label']} spread · "
                        f"IV {opt['atm_iv_pct']}%") if opt else "No listed options chain available right now.",
             "evidence": ([f"ATM strike: {opt['atm_strike']}",
                          f"Spread: {opt['spread_pct']}%" if opt["spread_pct"] is not None else "Spread: —",
                          f"Open interest: {opt['oi']:,.0f}  ·  Volume: {opt['volume']:,.0f}",
                          f"Delta: {opt['delta']:.2f}" if opt["delta"] is not None else "Delta: —",
                          f"Theta/day: {opt['theta_per_day']:.3f}" if opt["theta_per_day"] is not None else "Theta/day: —",
                          f"Expected move (straddle): ±{opt['expected_move_pct']}%"] if opt
                         else ["This ticker has no listed options chain, or the chain fetch failed."])},
            {"label": "Dealer Positioning",
             "value": dealer["regime"].split(" (")[0] if dealer else "—",
             "color": _lean_color(dealer["regime"]) if dealer else "#8b9a9d",
             "context": f"Net GEX ${dealer['net_gex_millions']:+,.0f}M (est.)" if dealer
                       else "No listed options chain available right now.",
             "evidence": ([dealer["note"], "Gamma wall strikes (context only): " +
                          ", ".join(f"{w:.0f}" for w in dealer["gamma_wall_strikes"])] if dealer
                         else ["This ticker has no listed options chain, or the chain fetch failed."])},
        ]},
    ]

    # PIIP audit 2026-08 (readability pass): one helper so each tab below can pull just the
    # groups it needs from the SAME `groups` list built above -- no duplicate list-building,
    # just a different slice rendered in each tab.
    def _pick(*label_prefixes):
        return [g for g in groups if any(g["label"].startswith(p) for p in label_prefixes)]

    # Macro data (PIIP audit 2026-08, readability pass v2): hoisted here, BEFORE the tabs,
    # from where it used to sit inside the Macro tab's own section -- the AI Synthesis section
    # (now at the bottom of Overview, which renders before the Macro tab) reads yield_10y/dxy/
    # wti/rsp_chg/spy_chg directly. Only the DATA fetch moves; macro_groups/fed_groups'
    # construction and rendering stay in the Macro tab, unchanged.
    mb = get_macro_batch()
    rsp_spy = get_rsp_vs_spy()

    def _last_close(df):
        return data.last_valid_close(df)

    def _day_chg(df):
        if df is None:
            return None
        c = df["Close"].dropna()
        if len(c) < 2:
            return None
        return (float(c.iloc[-1]) / float(c.iloc[-2]) - 1) * 100

    yield_10y = _last_close(mb.get(macro.YIELD_TICKERS["10Y"]))
    dxy = _last_close(mb.get(macro.DXY_TICKER))
    wti = _last_close(mb.get(macro.OIL_TICKERS["WTI"]))
    rsp_chg = _day_chg(rsp_spy.get("RSP"))
    spy_chg = _day_chg(rsp_spy.get("SPY"))
    pcr_label = ("Slightly call-heavy" if put_call_ratio is not None and put_call_ratio < 0.9 else
                "Slightly put-heavy" if put_call_ratio is not None and put_call_ratio > 1.1 else
                "Balanced" if put_call_ratio is not None else None)


    tab_overview, tab_timeframes, tab_context, tab_options, tab_macro = st.tabs(
        ["Overview", "Timeframes & Trend", "Market Context", "Options & Contract",
         "Macro & Diagnostics"])

    with tab_overview:
        # Morning Read (PIIP audit 2026-08): Premarket Thesis / AI Premarket Thesis / Day Regime
        # side by side as one unified card, per the user's chosen layout after reviewing 3 mockup
        # options -- these 3 answer sequential "what's the read right now" questions (before Day
        # Regime has data / AI's take on the same data / the live intraday regime once it does),
        # so they read as one instrument panel rather than 3 separately-bordered boxes stacked
        # with a lot of dead space (the AI box in particular was mostly empty before ever asked).
        pt_data = _compute_premarket_thesis(index_snaps, vix_snap, participation, intraday, daily,
                                            alignment, datetime.now(ZoneInfo("America/New_York")))
        _render_unified_morning_read(pt_data, regime, ticker)

        # PIIP audit 2026-08 (per user request): was a permanent 13-line paragraph -- collapsed to
        # a single-line header + bright hover-info icon (_info_header -- the native st.caption
        # `help=` icon was tried first and confirmed too pale/easy to miss) so the explanation is
        # still one hover away without permanently eating vertical space on every page load.
        _info_header("How Day Regime, Market DNA & Trade Confidence relate", (
            "Market DNA (below) and Day Regime (above) answer two DIFFERENT questions, not the "
            "same one twice — Market DNA is the day's overall SHAPE (Gap & Go / Trend Day / Chop "
            "/ Grind / Range Bound), set once it has enough evidence and stable for the session; "
            "Day Regime is the live DIRECTIONAL STATE right now, which can pass through several "
            "stages (Developing → Confirmed → Weakening → Confirmed again) within a single Market "
            "DNA day-type as the intraday trend develops and pulls back. Day Regime is the "
            "headline synthesis. Trade Confidence (below) blends Market Bias + Entry Quality + "
            "liquidity into one directional-confidence number. Trend State, Timeframe Sequence, "
            "and Market Bias (Timeframes & Trend / Market Context tabs) are the more granular "
            "reads Day Regime is built from — they usually agree; when they don't, that "
            "disagreement is itself the signal (see REGIME TRANSITION / TREND WEAKENING / NO "
            "CLEAR EDGE above)."), size="0.85rem")

        # "What's happened before" headline table (per user request): the same
        # zero_dte_log.regime_stats() data that used to sit behind a manual button in the
        # diagnostics tab, promoted to a headline table right next to "what this looks like"
        # (Day Regime above) -- pairs the live read with its own historical outcome record in one
        # place. 5-min cache (zd_regime_stats), not recomputed every 30s refresh.
        _render_regime_outcome_table(zd_regime_stats(ticker))

        with st.container(border=True):
            st.markdown(f"**📊 Intraday Chart — {ticker}**")
            _render_intraday_candlestick(ticker, intraday.get(ticker), key_prefix="zd",
                                         data_quality=data_quality)

        # ── Hero: the one number worth seeing without scrolling ──
        conf_color = _score_color(confidence["score"])
        with st.container(border=True):
            hc1, hc2 = st.columns([3, 2])
            with hc1:
                st.caption(f"{ticker} · TRADE CONFIDENCE")
                st.markdown(f'<div style="font-size:2.6rem;font-weight:800;color:{conf_color};'
                            f'line-height:1.1">{confidence["score"]:.0f}/100 &nbsp; '
                            f'<span style="font-size:1.6rem">{confidence["bias_direction"]}</span></div>',
                            unsafe_allow_html=True)
                dna_line = "Insufficient evidence" if dna["label"] == "Insufficient Evidence" else dna["label"]
                st.caption(f"Market DNA: **{dna_line}** — {dna['note']}")
                _info_header("⚠️ Not proven to beat the market — hover for the backtest", (
                    "Backtested Jul 2023–2026 (546 days, reusing this exact scoring code): even "
                    "the highest-confidence bucket hit ~57% next-day directional accuracy — "
                    "statistically indistinguishable from SPY's own 57.3% base rate over the same "
                    "period. This score has not been shown to beat the market's baseline. Treat it "
                    "as an internal-agreement heuristic, not a track record."),
                    size="0.8rem", color="#fabf6b")
                # NO CLEAR EDGE (PIIP audit 2026-08, Batch 1 / Phase 21): a first-class, EXPLAINED
                # outcome -- only renders when Market Bias itself has no real lean, and only shows
                # reasons already computed elsewhere on the page, never a new judgment.
                if no_edge_reasons:
                    reasons_html = "".join(f"<li>{_esc(r)}</li>" for r in no_edge_reasons)
                    st.markdown(f'<div style="margin-top:0.6rem;padding:0.6rem 0.8rem;background:'
                               f'#2a2415;border:1px solid #5a4c2c;border-radius:6px;">'
                               f'<b style="color:#fabf6b">⚠️ NO CLEAR EDGE</b>'
                               f'<ul style="margin:0.3rem 0 0 1.1rem;padding:0;font-size:0.85rem;'
                               f'color:#e8ecec">{reasons_html}</ul></div>', unsafe_allow_html=True)
            with hc2:
                # Confluence: how many independent signals actually agree with the bias's own lean --
                # market_bias() already counts this internally to build its own confidence number, but
                # never showed the count itself. One glanceable line here (the "quick look" the user
                # asked for); the full itemized ✓/✗ breakdown lives in the expander below, not here,
                # so the hero stays a fast read instead of turning into an 8-line checklist.
                conf_ratio = confluence["agree"] / confluence["total"] if confluence["total"] else 0.0
                confluence_color = "#79ed8e" if conf_ratio >= 0.7 else "#fabf6b" if conf_ratio >= 0.4 else "#ff8080"
                st.markdown(f'<div style="font-weight:700;margin-bottom:0.3rem">'
                            f'Confluence: <span style="color:{confluence_color}">'
                            f'{confluence["agree"]}/{confluence["total"]} signals agree</span></div>',
                            unsafe_allow_html=True)
                for mark, label in confidence["checks"]:
                    st.write(f"{mark} {label}")
            with st.expander("Confluence breakdown"):
                st.markdown(f"**Confluence detail — {confluence['lean_label']} lean**")
                # Colored 2-column tile grid, not a flat list of plain-text ✓/✗ lines -- the flat list
                # (confirmed from a live screenshot) was genuinely hard to scan: 8 same-weight lines in
                # a column, no color on the check marks themselves. Same green/red tile-tint palette
                # (#152a1e/#2a1515 bg) already used app-wide for Feed/Lottery/Catalysts cards, so this
                # reads consistently instead of inventing a new look just for this one panel.
                conf_tiles = "".join(
                    f'<div style="display:flex;align-items:center;gap:0.5rem;padding:0.4rem 0.6rem;'
                    f'background:{"#152a1e" if ok else "#2a1515"};border:1px solid '
                    f'{"#2c5a3c" if ok else "#5a2c2c"};border-radius:6px;">'
                    f'<span style="color:{"#79ed8e" if ok else "#ff8080"};font-weight:800;'
                    f'font-size:0.9rem;flex-shrink:0">{"✓" if ok else "✗"}</span>'
                    f'<span style="font-size:0.8rem;color:#e8ecec">{label}</span></div>'
                    for label, ok in confluence["checks"])
                st.markdown(f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;'
                           f'margin-bottom:0.9rem">{conf_tiles}</div>', unsafe_allow_html=True)
                st.markdown(f"**Entry Quality breakdown — {entry['score']:.0f}/100 "
                           f"({entry['risk_label']} risk)**")
                for label, pts in entry["reasons"].items():
                    st.write(f"{pts:+.1f}  {label}")

        st.caption("Tap any row to see the evidence behind its number.")

        with st.container(border=True):
            _info_header("🤖 5-agent AI synthesis (optional, real spend)", (
                "5 weighted Claude Haiku calls (News & Catalyst 30% · Technical & Market "
                "Structure 25% · Options & Positioning 20% · Macro & Cross-Asset 15% · "
                "Skeptic/Risk 10%), each scoring the signals above from its own focus area. "
                "The final blend below is CODE-computed from those scores, never invented by "
                "any single call. Same backtest caveat as Trade Confidence: this is a "
                "descriptive lean, not a trade recommendation. Real spend, ~$0.03–0.05/run "
                "(5 calls), governed by a $3/day · $0.15/run · 5-call cap."), size="1.3rem")
            if not os.getenv("ANTHROPIC_API_KEY"):
                _ai_key_missing_notice()
            else:
                ai_0dte_key = f"ai_0dte_{ticker}"
                zd_question = st.text_input(
                    "Anything specific you want the agents to consider? (optional)",
                    key=f"ai_0dte_q_{ticker}",
                    placeholder="e.g. Does this change if I'm only looking at the next hour?")
                if st.button("🤖 Run 5-agent synthesis", key=f"ai_0dte_btn_{ticker}"):
                    with st.spinner("Running 5 Claude calls…"):
                        try:
                            # Explicit timing context -- without this, an agent scoring "Bullish 82"
                            # can't tell 9:35am (6.5 hours left) from 3:50pm (10 minutes left), even
                            # though time-to-expiration is arguably THE defining variable for a 0DTE
                            # decision (theta/gamma both accelerate hard into the close). Computed here,
                            # never invented by the model.
                            now_et = datetime.now(ZoneInfo("America/New_York"))
                            mkt_open_et = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
                            mkt_close_et = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
                            in_session = mkt_open_et <= now_et <= mkt_close_et
                            # Expiry note is now conditional on the ACTUAL fetched chain (PIIP audit
                            # 2026-08, Batch 3: NVDA is selectable here too and doesn't list same-day
                            # expiries the way the index ETFs do) -- never assume 0DTE just because
                            # this is the "0DTE Intelligence" page.
                            chain_expires_today = bool(chain) and chain.get("expiry") == date.today().isoformat()
                            if in_session and chain_expires_today:
                                expiry_note = "These options expire at TODAY's market close (~16:00 ET)."
                            elif in_session and chain:
                                expiry_note = (f"Nearest listed expiry is {chain['expiry']} — NOT 0DTE "
                                              f"for {ticker} today.")
                            elif in_session:
                                expiry_note = "No listed options chain available right now."
                            else:
                                expiry_note = ("Outside regular market hours (9:30-16:00 ET) -- signals "
                                              "may reflect the last completed session, not a live one.")
                            session_timing = {
                                "current_time_et": now_et.strftime("%H:%M:%S"),
                                "session_minutes_elapsed": dna.get("metrics", {}).get("session_minutes"),
                                "minutes_until_close": (round((mkt_close_et - now_et).total_seconds() / 60)
                                                        if in_session else None),
                                "note": expiry_note,
                            }
                            signals = {
                                "ticker": ticker,
                                "session_timing": session_timing,
                                "trade_confidence": confidence["score"],
                                "bias_direction": confidence["bias_direction"],
                                "market_dna": dna["label"],
                                "market_bias": {"direction_label": bias["direction_label"],
                                               "confidence_pct": bias["confidence_pct"],
                                               "environment": bias["environment"]},
                                "breadth": {"score_signed": breadth["score_signed"], "label": breadth["label"]},
                                "sector_health_pct": sector["overall_confirmation_pct"],
                                "mega_cap_health": mega["score"],
                                "nvda_relative_strength": ({"label": nvda_rs["label"],
                                                            "spread_pct": nvda_rs["spread_pct"],
                                                            "acceleration": nvda_rs_accel["trend"]
                                                            if nvda_rs_accel else None}
                                                           if nvda_rs else None),
                                "momentum": ({"continuation_score_pct":
                                             momentum["continuation_score_pct"]} if momentum else None),
                                "reversal_pressure_score": reversal["reversal_pressure_score"],
                                "trend_integrity": {"score": integrity["score"], "label": integrity["label"]},
                                "vwap_crossings": crossings["count"] if crossings else None,
                                "day_regime": regime["state"],
                                "timeframe_sequence": tf_sequence["interpretation"],
                                "timeframe_alignment": ({"aligned_direction": alignment["aligned_direction"],
                                                         "agree": alignment["agree"], "total": alignment["total"]}
                                                        if alignment["total"] else None),
                                "trend_state": trend_state["state"],
                                "entry_quality": entry["score"],
                                "confluence": f"{confluence['agree']}/{confluence['total']}",
                                "options_health": opt["execution_quality"] if opt else None,
                                "dealer_positioning": dealer["regime"] if dealer else None,
                                "macro": {"10y_yield": yield_10y, "dxy": dxy, "wti": wti,
                                         "put_call_oi_ratio": put_call_ratio,
                                         "rsp_vs_spy": {"rsp_chg": rsp_chg, "spy_chg": spy_chg}},
                            }
                            client = agents.LLMClient(budget=agents.Budget(max_calls_per_run=5), dry_run=False)
                            out, cost = agents.zero_dte_agent_synthesis(
                                signals, client, user_question=zd_question.strip() or None)
                            st.session_state[ai_0dte_key] = {"out": out, "cost": cost,
                                                             "question": zd_question.strip() or None}
                            st.toast(f"5-agent synthesis done — ${cost:.4f} spent.")
                        except Exception as e:
                            st.error(f"AI call failed: {e}")
                zd_cached = st.session_state.get(ai_0dte_key)
                if zd_cached:
                    o = zd_cached["out"]
                    fscore, fconf = o["final_score"], o["final_confidence"]
                    # Lean/strategy/watch-level are CODE-computed from the blended score + already-
                    # fetched real levels (VWAP) -- never invented by the LLM, same "AI interprets,
                    # code computes" rule as everywhere else. Descriptive framing (Lean / "if
                    # considering" / Watch level), not "Suggested Trade" -- this page says elsewhere it
                    # never recommends a trade, and the backtest shows no proven edge to act on.
                    if fscore > 15:
                        lean, lean_color = "Bullish", "#79ed8e"
                        strategy = "If considering a directional structure: ATM/near-the-money calls (illustrative only, not a recommendation)."
                        watch = f"VWAP reclaim/hold above ${snap['vwap']:.2f}"
                    elif fscore < -15:
                        lean, lean_color = "Bearish", "#ff8080"
                        strategy = "If considering a directional structure: ATM/near-the-money puts (illustrative only, not a recommendation)."
                        watch = f"VWAP breakdown/hold below ${snap['vwap']:.2f}"
                    else:
                        lean, lean_color = "Neutral", "#87d1ff"
                        strategy = "No clear directional edge — no illustrative structure shown."
                        watch = f"Watching VWAP (${snap['vwap']:.2f}) for a decisive break either way."

                    st.caption(f"Last run cost **${zd_cached['cost']:.4f}** · based on the signals at "
                              "the moment you clicked — click again for a fresh read.")
                    fc1, fc2 = st.columns([2, 3])
                    with fc1:
                        st.markdown(f'<div style="font-size:2.2rem;font-weight:800;color:{lean_color};'
                                   f'line-height:1.1">{fscore:+.0f} &nbsp; <span style="font-size:1.3rem">'
                                   f'{lean}</span></div>', unsafe_allow_html=True)
                        st.caption(f"Final {ticker} Score · Confidence {fconf:.0f}%")
                    with fc2:
                        st.markdown(f"**Lean:** {lean}  \n**If considering a strategy:** {strategy}  \n"
                                  f"**Watch level:** {watch}")

                    # Debate framing: every agent's key evidence pooled into "case FOR"/"case AGAINST"
                    # by its own recommended_bias, each line tagged with its source agent -- not a
                    # second scoreboard, since the hero block above already shows the final score/
                    # lean/watch level and repeating those at the bottom read as unexplained duplicate
                    # info (confirmed live against a mockup). Neutral-biased agents sit out of the
                    # debate entirely -- a "no side taken" call has no evidence for either column.
                    for_items = [(name, e) for name, a in o["agents"].items()
                                if a["recommended_bias"] == "Bullish" for e in a.get("key_evidence", [])]
                    against_items = [(name, e) for name, a in o["agents"].items()
                                     if a["recommended_bias"] == "Bearish" for e in a.get("key_evidence", [])]

                    def _debate_col_html(title, dot, items, bg, border):
                        rows = "".join(
                            f'<div style="font-size:0.83rem;margin-bottom:0.5rem;padding-left:0.9rem;'
                            f'border-left:2px solid #2a3a3c">{_esc(e)}'
                            f'<span style="font-family:ui-monospace,Consolas,monospace;font-size:0.68rem;'
                            f'color:#8b9a9d;display:block;margin-top:0.1rem">— {_esc(name)}</span></div>'
                            for name, e in items) or (
                            '<div style="font-size:0.82rem;color:#8b9a9d">No agent landed here this run.</div>')
                        return (f'<div style="background:{bg};border:1px solid {border};border-radius:8px;'
                               f'padding:0.8rem 1rem">'
                               f'<div style="font-weight:700;margin-bottom:0.5rem;display:flex;'
                               f'justify-content:space-between">'
                               f'<span>{dot} {title}</span><span>{len(items)} point{"s" if len(items) != 1 else ""}</span>'
                               f'</div>{rows}</div>')

                    dcol1, dcol2 = st.columns(2)
                    with dcol1:
                        st.markdown(_debate_col_html("The case FOR", "🟢", for_items,
                                                     "#101d15", "#2c5a3c"), unsafe_allow_html=True)
                    with dcol2:
                        st.markdown(_debate_col_html("The case AGAINST", "🔴", against_items,
                                                     "#1d1212", "#5a2c2c"), unsafe_allow_html=True)

                    # Neutral agents contribute nothing to either column above (correctly -- a "no
                    # side taken" call has no evidence for a debate), but silently dropping them made
                    # a run where most agents declined to guess (stale/pre-market data, etc.) look like
                    # only 1 of 5 agents had responded at all (confirmed live -- a real "why does the
                    # FOR column say 0" confusion, not an actual computation bug). Naming them here
                    # makes clear all 5 calls ran.
                    neutral_names = [name for name, a in o["agents"].items()
                                     if a["recommended_bias"] == "Neutral"]
                    if neutral_names:
                        st.caption(f"⚪ No strong lean this run: {', '.join(neutral_names)} "
                                  f"({len(neutral_names)} of 5 agents didn't take a side).")

                    # Direct answers to the user's own question, one per agent that actually returned
                    # one -- shown LAST, after all the other agent context above, not first. Showing
                    # each agent's own answer (not a single blended one) is deliberate -- they can
                    # reasonably disagree, and that disagreement is itself useful, not noise to average away.
                    if zd_cached.get("question"):
                        answers = [(name, a["user_question_answer"]) for name, a in o["agents"].items()
                                  if a.get("user_question_answer")]
                        if answers:
                            q_tile = _tile_style("🔵")
                            rows = "".join(
                                f'<div style="margin-bottom:0.5rem"><span style="font-family:ui-monospace,'
                                f'Consolas,monospace;font-size:0.68rem;color:#8b9a9d">{_esc(name)}</span><br>'
                                f'{_esc(ans)}</div>' for name, ans in answers)
                            st.markdown(
                                f'<div style="background:{q_tile["bg"]};border:1px solid {q_tile["border"]};'
                                f'border-radius:10px;padding:0.9rem 1.1rem;margin-top:0.9rem">'
                                f'<div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.05em;'
                                f'color:{_TEXT_BY_EMOJI["🔵"]};font-weight:700;margin-bottom:0.5rem">'
                                f'❓ Your question: "{_esc(zd_cached["question"])}"</div>{rows}</div>',
                                unsafe_allow_html=True)
                        else:
                            st.caption("⚠️ None of the 5 agents returned a direct answer to your "
                                      "question this run — try rephrasing it or click again.")

    with tab_timeframes:
        _render_list_view(_pick("🧭 Trend Integrity", "🕰️ Multi-Timeframe Alignment",
                                "📈 Price & Momentum"),
                          container_key="zd_tf_rows")

        st.caption("Diverging view — every metric on one shared axis, sorted so the biggest leans surface first.")
        momentum_val = momentum["continuation_score_pct"] if momentum else 50.0
        opt_val = opt["execution_quality"] if opt else 50.0
        breadth_val = (breadth["score_signed"] + 100) / 2
        diverging_raw = [
            ("Reversal Pressure", reversal["strength"]),
            ("Entry Quality", entry["score"]),
            ("Market Bias", bias["calls_pct"]),
            ("Momentum", momentum_val),
            (f"{ticker} Health", health["score"]),
            ("Breadth (proxy)", breadth_val),
            ("Sector Health", sector["overall_confirmation_pct"]),
            ("Mega Cap Health", mega["score"]),
            ("Options Health", opt_val),
        ]
        diverging_raw.sort(key=lambda kv: abs(kv[1] - 50), reverse=True)
        diverging_rows = [{"label": label, "value": val, "display": f"{val:.0f}", "color": _score_color(val)}
                          for label, val in diverging_raw]
        footer_html = None
        if dealer:
            footer_html = (f"◆ Dealer Positioning: <b>{dealer['regime'].split(' (')[0]}</b> "
                           f"— Net GEX ${dealer['net_gex_millions']:+,.0f}M (est.) — "
                           f"volatility-regime flag, not a directional lean")
        _render_diverging_chart(diverging_rows, footer_html)


    with tab_context:
        _render_list_view(_pick("🌐 Market Context", "📐 Levels"),
                          container_key="zd_context_rows")

    with tab_options:
        _render_list_view(_pick("⚙️ Options & Dealer"), container_key="zd_options_rows")

        # Exit Quality needs real Python interactivity (checkbox + radio), which a static embedded
        # component can't provide -- rendered as native widgets, right above the embedded list so it
        # still reads as part of "Trade Readiness" even though it's a different building block.
        with st.container(border=True):
            ec1, ec2 = st.columns([1.6, 5])
            ec1.markdown('<div style="font-size:0.85rem;font-weight:700;padding-top:0.3rem">🎯 Exit Quality</div>',
                        unsafe_allow_html=True)
            with ec2:
                in_trade = st.checkbox("I'm in a trade", key=f"zd_intrade_{ticker}")
                if in_trade:
                    direction = st.radio("Direction", ["CALL", "PUT"], horizontal=True, key=f"zd_dir_{ticker}")
                    eq = zd.exit_quality(direction, bias, momentum, breadth, snap)
                    for note in eq["notes"]:
                        st.write(f"• {note}")

                    # Contract-specific quote (PIIP audit 2026-08, Option C): everything ABOVE this
                    # (Options Health, Dealer Positioning) is auto-picked ATM -- often not the strike
                    # someone actually holds. Snap the strike list to whichever side (calls/puts) the
                    # chosen Direction implies, default the picker to the ATM strike so it's a one-
                    # click confirm for the common case, but let it be overridden.
                    strikes = zd.available_strikes(chain, direction) if chain else []
                    if strikes:
                        default_strike = (opt["atm_strike"] if opt and opt["atm_strike"] in strikes
                                          else min(strikes, key=lambda s: abs(s - snap["last"])))
                        picked_strike = st.selectbox(
                            "Your contract's strike", strikes,
                            index=strikes.index(default_strike),
                            key=f"zd_strike_{ticker}_{direction}")
                        cq = zd.contract_quote(chain, snap["last"], picked_strike, direction)
                        if cq:
                            cq_color = "#79ed8e" if cq["execution_quality"] > 60 else \
                                       "#fabf6b" if cq["execution_quality"] > 40 else "#ff8080"
                            delta_bit = f" · Δ {cq['delta']:.2f}" if cq["delta"] is not None else ""
                            theta_bit = f" · θ/day {cq['theta_per_day']:.3f}" if cq["theta_per_day"] is not None else ""
                            spread_bit = cq["spread_pct"] if cq["spread_pct"] is not None else "—"
                            dte_bit = f" · {cq['dte_days']}DTE" if cq["dte_days"] is not None else ""
                            money_bit = f" · {cq['moneyness_label']}" if cq["moneyness_label"] else ""
                            if cq["last_trade_minutes"] is None:
                                trade_bit = "no trades reported"
                            elif cq["last_trade_minutes"] < 60:
                                trade_bit = f"last traded {cq['last_trade_minutes']:.0f} min ago"
                            else:
                                trade_bit = f"last traded {cq['last_trade_minutes'] / 60:.1f}h ago"
                            st.markdown(
                                f'<div style="padding:0.5rem 0.7rem;background:#15191a;border:1px solid '
                                f'#232b2d;border-radius:6px;margin-top:0.4rem">'
                                f'<b>{ticker} {cq["matched_strike"]:.0f}{direction[0]}</b>{dte_bit}{money_bit} · '
                                f'Bid ${cq["bid"]:.2f} / Ask ${cq["ask"]:.2f} · '
                                f'<span style="color:{cq_color}">{cq["spread_label"]} spread '
                                f'({spread_bit}%)</span> · '
                                f'OI {cq["oi"]:,.0f} · Vol {cq["volume"]:,.0f}{delta_bit}{theta_bit}'
                                f'<br><span style="font-size:0.78rem;color:#8b9a9d">{trade_bit}</span>'
                                f'</div>', unsafe_allow_html=True)
                            if cq["strike_snapped"]:
                                st.caption(f"No exact {picked_strike:.0f} strike listed — showing the "
                                          f"nearest one, {cq['matched_strike']:.0f}.")

                            # Bid Simulator (PIIP audit 2026-08, Batch 1 / Phase 19): objective diff
                            # math only, against the SAME quote already fetched above -- no new call,
                            # no fill-probability claim (that needs historical fill data this project
                            # doesn't have yet, see zero_dte_log.py's own collection-only stance).
                            if cq["mid"]:
                                hyp_bid = st.number_input(
                                    "Bid Simulator — hypothetical bid", min_value=0.0,
                                    value=round(cq["bid"], 2), step=0.01, format="%.2f",
                                    key=f"zd_bidsim_{ticker}_{direction}_{cq['matched_strike']}")
                                vs_bid = hyp_bid - cq["bid"]
                                vs_mid = hyp_bid - cq["mid"]
                                vs_ask = hyp_bid - cq["ask"]
                                pct_below_mid = (1 - hyp_bid / cq["mid"]) * 100 if cq["mid"] else None
                                st.caption(
                                    f"vs current bid: {vs_bid:+.2f} · vs mid: {vs_mid:+.2f} "
                                    f"({pct_below_mid:+.1f}% below mid) · vs ask: {vs_ask:+.2f} · "
                                    f"**Fill probability: UNKNOWN** — not tracked yet, needs "
                                    "historical quote/fill data this project doesn't collect yet.")
                    else:
                        st.caption("No listed options chain available for a contract-specific quote right now.")

                    # Edge-triggered alert: bias flipping against your direction is the "major change"
                    # signal -- toast + beep fire once on the refresh it first flips, the red banner
                    # stays up every refresh for as long as it's still flipped (never auto-exits you).
                    alert_key = f"zd_alert_{ticker}_{direction}"
                    was_alert = st.session_state.get(alert_key, False)
                    is_alert = not eq["trend_aligned"]
                    st.session_state[alert_key] = is_alert
                    if is_alert:
                        _render_direction_alert(
                            f"Bias has flipped against your {direction} — now {bias['environment']} "
                            f"({bias['direction_label']}). Not an auto-exit signal — your call.")
                        if not was_alert:
                            st.toast(f"⚠️ {ticker} bias flipped against your {direction}", icon="🚨")
                            st.markdown(f'<audio autoplay src="{_alert_beep_data_uri()}"></audio>',
                                       unsafe_allow_html=True)
                else:
                    st.caption("Flip on once in a trade — never an auto-exit signal.")


    with tab_macro:
        with st.expander("Market DNA metrics, Catalyst Terminal, Signal Calibration Log, "
                         "Regime Timeline, Historical Stats, and Data Quality"):
                if dna["metrics"]:
                    st.markdown("**Market DNA metrics**")
                    # Same 2-column tile grid as Confluence detail just above -- neutral (⚪) tint since
                    # these are raw metric readouts, not pass/fail checks, so green/red would falsely
                    # imply a judgment call that isn't there. Was a flat `label: value` st.write loop,
                    # the same hard-to-scan format the user just flagged for Confluence.
                    dna_tiles = "".join(
                        f'<div style="display:flex;justify-content:space-between;align-items:center;'
                        f'gap:0.5rem;padding:0.4rem 0.6rem;background:#15191a;border:1px solid #232b2d;'
                        f'border-radius:6px;" title="{glossary.help_for(_DNA_METRIC_LABELS.get(key, key)).replace(chr(34), chr(39))}">'
                        f'<span style="font-size:0.78rem;color:#8b9a9d">{_DNA_METRIC_LABELS.get(key, key)}</span>'
                        f'<span style="font-family:ui-monospace,Consolas,monospace;font-weight:700;'
                        f'font-size:0.82rem;color:#e8ecec">{_dna_metric_display(key, val)}</span></div>'
                        for key, val in dna["metrics"].items())
                    st.markdown(f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;'
                               f'margin-bottom:0.9rem">{dna_tiles}</div>', unsafe_allow_html=True)
                st.markdown("**📰 Catalyst Terminal**")
                st.caption("General market news, deduplicated and scored — same for every index, "
                          "not per-ticker.")
                _render_catalyst_terminal()

                # Signal Calibration Log status (PIIP audit 2026-08, Option C): every refresh's
                # scores are now being logged locally (zdlog.log_signal_snapshot() above) -- this is
                # collection status ONLY, not a backtest or win-rate claim. Same explicit deferral
                # already used for Reddit Momentum's own log_snapshot(): a real validation/calibration
                # view is honest once weeks/months of logged sessions exist, not before.
                st.markdown("**📊 Signal Calibration Log**")
                log_status = zdlog.collection_status(ticker)
                if log_status["rows"]:
                    st.caption(f"Collecting since {log_status['first_date']} — {log_status['rows']:,} "
                              f"{ticker} snapshots logged across {log_status['days']} day(s). This is "
                              "collection only, NOT a backtest — there isn't enough history yet to "
                              "validate any of this page's scores against real outcomes.")
                else:
                    st.caption("No snapshots logged yet for this ticker — starts collecting on the "
                              "next refresh.")

                # State Timeline (PIIP audit 2026-08, Batch 3 / Phase 9): today's actual Day Regime
                # transitions, edge-triggered-logged above -- reads back what really happened, never
                # reconstructs history after the fact.
                st.markdown("**🕰️ Today's Regime Timeline — What Changed?**")
                timeline = zdlog.regime_timeline(ticker)
                if timeline:
                    # PIIP audit 2026-08 (state-architecture review, Phase 5): each row's "why" is
                    # RECOMPUTED here from the stored detail facts (this row's vs. the prior row's),
                    # never stored as a conclusion -- explain_transition() only reports measured
                    # deltas, matching the example format: Primary driver / Supporting changes /
                    # Higher-timeframe context / a templated (never LLM-invented) interpretation.
                    for i, t in enumerate(timeline):
                        ts_short = t["ts"].split("T")[-1] if "T" in t["ts"] else t["ts"].split(" ")[-1]
                        reasons_bit = "; ".join(t["reasons"]) if t["reasons"] else ""
                        st.write(f"**{ts_short}** {t['from_state'] or '—'} → {t['to_state']}"
                                + (f"  \n_{reasons_bit}_" if reasons_bit else ""))
                        prev_detail = timeline[i - 1]["detail"] if i > 0 else None
                        if t["detail"] is not None:
                            explanation = zd.explain_transition(prev_detail, t["detail"],
                                                                 t["from_state"], t["to_state"],
                                                                 to_reasons=t["reasons"])
                            with st.expander("Why?", expanded=False):
                                if explanation["primary"]:
                                    st.write(f"**Primary:** {explanation['primary']}")
                                if explanation["supporting"]:
                                    st.write("**Supporting:**")
                                    for s in explanation["supporting"]:
                                        st.write(f"　• {s}")
                                if explanation["higher_timeframes"]:
                                    tf_bits = " · ".join(f"{tf} {d}" for tf, d in
                                                         explanation["higher_timeframes"].items())
                                    st.caption(f"Timeframes at this moment: {tf_bits}")
                                st.write(f"**Interpretation:** {explanation['interpretation']}")
                                st.caption("Every line above traces back to an actual measured "
                                          "before/after comparison, stored at the time of each "
                                          "transition -- never invented after the fact, never "
                                          "LLM-generated.")
                        elif i > 0:
                            st.caption("　(Logged before the detailed What-Changed tracking was "
                                      "added — only the state change and its own reasons are "
                                      "available for this row.)")
                else:
                    st.caption("No regime changes logged yet today — the state has held steady since "
                              "this page started tracking it, or this is the first read of the day.")

                # Historical Regime Stats (PIIP audit 2026-08, Batch 3 / Phases 12-13; promoted to
                # a headline table on the Overview tab per user request, state-architecture review
                # follow-up) -- the manual on-demand button this used to be is gone; the SAME
                # zd_regime_stats()-cached data (5-min TTL) now renders as "📊 What's Happened
                # Before" right under the Day Regime banner on Overview, avoiding two different-
                # looking displays of the same numbers. This pointer replaces the old duplicate.
                st.caption("**📈 Historical Regime Stats** moved to the Overview tab — see "
                          "'📊 What's Happened Before' right under the Day Regime banner.")

                # Data Quality panel (PIIP audit 2026-08, Batch 1 / Phase 22): consolidates the
                # freshness/proxy caveats already scattered across this page's captions into one
                # place, plus which timeframes actually have enough of today's session to compute yet.
                st.markdown("**🔍 Data Quality**")
                dq_color = "#79ed8e" if data_quality["underlying"] == "Fresh" else "#ff8080"
                st.write(f":{'green' if data_quality['underlying']=='Fresh' else 'red'}[●] "
                        f"Underlying: **{data_quality['underlying']}**" +
                        (f" (last bar {data_quality['underlying_minutes_stale']:.0f} min old)"
                         if data_quality["underlying_minutes_stale"] is not None else ""))
                st.write(f":{'green' if data_quality['options_available'] else 'red'}[●] "
                        f"Options: {data_quality['options_note']}")
                tf_bits = " · ".join(f"{lbl} {'✓' if avail else '✗'}"
                                     for lbl, avail in data_quality["timeframe_availability"].items())
                st.caption(f"Timeframe availability: {tf_bits}")

        # Macro Context, deliberately last on the page -- real (not approximated/estimated) yields/
        # dollar/oil data from iip/macro.py (already existed, was never wired into any page before
        # this) plus put/call OI ratio and an equal-weight-vs-cap-weight breadth check. Cached for 5min
        # (not the page's usual 30s) because this is slower-moving day-regime context, not a 0DTE
        # timing signal -- the same reason it's deprioritized to the bottom of the page.
        #
        # Same _render_list_view() rows as the Levels/Market Context groups above, NOT a plain
        # st.metric row -- a first pass used st.metric and the user compared it directly against the
        # zero_dte_new_sections.html mockup they'd approved: metrics alone lost the per-item
        # description text and read far sparser/harder to scan than the labeled rows with context.
        # container_key must differ from the main list-view's default ("zd_list_rows") -- it becomes
        # both the CSS scope and every row's session_state key, so reusing it would collide.
        macro_groups = [{"label": "🌐 Macro Context", "rows": [
            {"label": "10-Year Treasury Yield", "value": f"{yield_10y:.2f}%" if yield_10y is not None else "—",
             "color": "#87d1ff",
             "context": "Often moves SPY more than people realize, especially on CPI/Fed days",
             "evidence": ["Real Treasury yield, not an estimate (^TNX)",
                         "Slower-moving than everything above — day-regime context, not a timing signal."]},
            {"label": "Dollar Index (DXY)", "value": f"{dxy:.2f}" if dxy is not None else "—",
             "color": "#87d1ff",
             "context": "Strong dollar historically a headwind for large-cap earnings / SPY",
             "evidence": ["ICE Dollar Index (DX-Y.NYB) — real data, not an estimate."]},
            {"label": "WTI Crude Oil", "value": f"${wti:.2f}" if wti is not None else "—",
             "color": "#87d1ff",
             "context": "Context for Energy-sector-driven moves in the sector rotation panel above",
             "evidence": ["WTI front-month futures (CL=F) — real data, not an estimate."]},
            {"label": "Put/Call OI Ratio",
             "value": f"{put_call_ratio:.2f}" if put_call_ratio is not None else "—",
             "color": ("#79ed8e" if pcr_label == "Slightly call-heavy" else
                      "#ff8080" if pcr_label == "Slightly put-heavy" else "#87d1ff"),
             "context": "From the same options chain already fetched for GEX above — no new network call",
             "evidence": ([f"{pcr_label} ({put_call_ratio:.2f})",
                          "Sentiment context, noisier than Dealer Positioning above — not a strong "
                          "signal on its own."] if put_call_ratio is not None
                         else ["No listed options chain available right now."])},
            {"label": "Equal-Weight vs Cap-Weight",
             "value": (f"RSP {rsp_chg:+.2f}%<br>SPY {spy_chg:+.2f}%"
                      if rsp_chg is not None and spy_chg is not None else "—"),
             "color": ("#79ed8e" if (rsp_chg is not None and rsp_chg >= (spy_chg or 0)) else "#ff8080")
                      if rsp_chg is not None and spy_chg is not None else "#8b9a9d",
             "context": "RSP vs SPY today — is the whole market moving, or just mega-caps?",
             "evidence": (["RSP = equal-weight S&P 500 ETF, SPY = cap-weight.",
                          "RSP lagging SPY means the move is concentrated in a handful of mega-caps, "
                          "not broad participation."] if rsp_chg is not None else ["Not enough data right now."])},
        ]}]

        # Fed liquidity + real economic releases -- both fetched via macro.py's free, keyless FRED CSV
        # export (same pattern already used above for nothing until now: liquidity_snapshot() and
        # yield_curve_snapshot() were fully built in iip/macro.py but never actually called from any
        # page). Its own group, not folded into "Macro Context" above -- that group is real-time-ish
        # market proxies (yields/DXY/oil), this is monthly/quarterly-cadence Fed & economic data, a
        # genuinely different update rhythm that deserves its own clearly-labeled section.
        yc = macro.yield_curve_snapshot(mb)
        liq = get_liquidity_snapshot()
        econ = get_econ_releases()

        def _liq_row(label, key, unit=""):
            d = liq.get(key)
            if not d:
                return {"label": label, "value": "—", "color": "#8b9a9d",
                       "context": "FRED data unavailable right now.", "evidence": ["No data returned."]}
            chg = d.get("chg_1w")
            return {"label": label, "value": f"{d['latest']:,.0f}{unit}",
                   "color": "#79ed8e" if (chg or 0) >= 0 else "#ff8080",
                   "context": f"{chg:+,.0f}{unit} vs a week ago" if chg is not None else f"As of {d['as_of']}",
                   "evidence": [f"As of {d['as_of']}", "Source: FRED (Federal Reserve Bank of St. Louis), "
                               "free public CSV export, no API key."]}

        econ_rows = []
        cpi = econ.get("CPI")
        econ_rows.append({
            "label": "CPI (YoY)", "value": f"{cpi['yoy_pct']:+.2f}%" if cpi and cpi.get("yoy_pct") is not None else "—",
            "color": "#87d1ff", "context": f"As of {cpi['as_of']}" if cpi else "FRED data unavailable right now.",
            "evidence": ["Headline CPI, seasonally adjusted (FRED series CPIAUCSL).",
                        "Real monthly release, not an estimate — updates once a month."] if cpi else ["No data returned."]})
        unemp = econ.get("Unemployment Rate")
        econ_rows.append({
            "label": "Unemployment Rate", "value": f"{unemp['latest']:.1f}%" if unemp else "—",
            "color": "#87d1ff", "context": f"As of {unemp['as_of']}" if unemp else "FRED data unavailable right now.",
            "evidence": ["FRED series UNRATE — real monthly release, not an estimate."] if unemp else ["No data returned."]})
        payrolls = econ.get("Nonfarm Payrolls")
        pr_chg = payrolls.get("mom_change_thousands") if payrolls else None
        econ_rows.append({
            "label": "Nonfarm Payrolls (MoM)", "value": f"{pr_chg:+,.0f}K jobs" if pr_chg is not None else "—",
            "color": "#79ed8e" if (pr_chg or 0) >= 0 else "#ff8080",
            "context": f"As of {payrolls['as_of']}" if payrolls else "FRED data unavailable right now.",
            "evidence": ["Change in total nonfarm employment vs the prior month (FRED series PAYEMS) — "
                        "the actual 'jobs added' headline number."] if payrolls else ["No data returned."]})

        fed_groups = [{"label": "🏦 Fed Liquidity & Economic Data", "rows": [
            {"label": "Yield Curve (13W–10Y)",
             "value": f"{yc['spread_13w_10y']:+.2f}pp" if yc.get("spread_13w_10y") is not None else "—",
             "color": "#ff8080" if (yc.get("spread_13w_10y") or 0) < 0 else "#87d1ff",
             "context": "Negative (inverted) has historically preceded recessions, with a long and variable lag",
             "evidence": ["10Y minus 13-week yield — the classic 2s10s spread isn't buildable here "
                         "(no free 2Y yield series exists), this is the closest free equivalent.",
                         "A single day's reading is noisy — the trend over weeks matters more than any one print."]},
            _liq_row("Treasury General Account", "TGA", "M"),
            _liq_row("Reverse Repo Usage", "RRP", "B"),
            _liq_row("Bank Reserves", "bank_reserves", "M"),
            *econ_rows,
        ]}]
        _render_list_view(macro_groups, container_key="zd_macro_rows")
        _render_list_view(fed_groups, container_key="zd_fed_rows")




if nav == "0DTE Intelligence":
    st.error(
        "⚠️ **RESEARCH TOOL, NOT A LIVE-TRADING FEED.** This page refreshes every ~30 seconds on "
        "free, near-real-time data (yfinance) — it is NOT tick-by-tick real time. It never "
        "recommends a trade. Making 0DTE scalping decisions that need to react in real time would "
        "require a **paid real-time feed (~0.5s latency)**, which this app does not currently have. "
        "Treat every number on this page as directional research context, not a timing signal.")

    zd_ticker = st.radio("Ticker", zd.ANALYSIS_TICKERS, horizontal=True, key="zd_ticker")
    _render_zero_dte(zd_ticker)

# ─────────────────────────── Lottery (gambling) ───────────────────────────
if nav == "Lottery":
    st.error("⚠️ **LOTTERY BUYS — GAMBLING.** Treat every one of these as a **near-certain total "
             "loss.** They're far-out-of-the-money bets (~2× the expected move) that only pay on a "
             "big, fast move. Buy ONLY money you're 100% willing to lose. Research / what-if — NOT advice.")
    lotto_cards = st.session_state.get("cards", [])
    if not lotto_cards:
        if st.button("🔄 Scan stocks", type="primary", key="lotto_scan_btn"):
            with st.spinner("Scanning ~80 stocks (mega-caps + lower-priced + biotech + low-priced; "
                            "~2–3 min; cached 15 min)…"):
                scanned = run_scan()
            for _c in scanned:
                scanner.log_card_forecast(_c, DB)   # same dedup as the Feed tab's own scan button
            st.session_state["cards"] = scanned
            lotto_cards = scanned
        else:
            st.info("Hit **Scan stocks** above (same scan as the 📰 Feed tab — this list is built "
                    "from the biggest expected-move names).")
    lotto_days = st.selectbox("Timeline (short = fast & cheap but brutal theta · long = more time, pricier)",
                              [7, 14, 30, 60, 90, 180], index=4, format_func=lambda d: f"~{d} days",
                              key="lotto_days")
    _info_header(f"🎰 Lottery — ~{lotto_days}d contracts", (
        "Biggest expected-move names first. Each is a ~2×-expected-move OTM contract "
        "expiring in the selected window. P(profit) is deliberately tiny — that's the whole "
        "point. Left to chance; sized as a burned ticket."), size="1.1rem", color="#fabf6b")
    wsb_lotto = get_wsb()
    for c in lotto_cards[:8]:
        try:
            lp = get_lottery(c["ticker"], c["spot"], lotto_days)
        except Exception:
            continue
        earn = get_earnings(c["ticker"])
        buzz = social.buzz_for(c["ticker"], wsb_lotto)
        reasons, side_txt, lean = lottery_why(c, lp, earn, buzz)
        has_catalyst = bool(earn and earn.get("days") is not None and 0 <= earn["days"] <= 45)
        tile = _tile_style("🟢" if has_catalyst else "⚪")
        st.markdown(
            f'<div style="background:{tile["bg"]};border:1px solid {tile["border"]};'
            'border-radius:8px;padding:0.85rem 1rem;margin-bottom:0.4rem">'
            f'<div style="font-weight:700;font-size:1.05rem">{c["ticker"]} — {c.get("name", c["ticker"])}</div>'
            f'<div style="font-size:0.8rem;color:#8b9a9d;margin-top:0.1rem">'
            f'${c["spot"]:,.2f} · {lp["dte"]}d move ±{lp["em_pct"]:.0f}%</div>'
            '<div style="font-size:0.85rem;margin-top:0.5rem"><b>Why a big move might be coming:</b></div>'
            + "".join(f'<div style="font-size:0.8rem;color:#8b9a9d;margin-top:0.15rem">- {r}</div>' for r in reasons)
            + f'<div style="font-size:0.85rem;margin-top:0.5rem"><b>Data-suggested direction:</b> {side_txt}</div>'
            '<div style="font-size:0.76rem;color:#8b9a9d;margin-top:0.1rem">'
            '⚠️ the lean is ~a coin flip — the data\'s best guess, NOT a prediction.</div>'
            '</div>', unsafe_allow_html=True)
        with st.container(border=True):
            with st.expander("📰 Articles — read these to find the real catalyst"):
                for nsi in c.get("news", [])[:4]:
                    if nsi.get("link"):
                        st.markdown(f"• [{nsi['title']}]({nsi['link']})  \n*{nsi.get('publisher', '')}*")
                    else:
                        st.markdown(f"• {nsi['title']}")
                if st.button("Load sector-specific articles (FDA/trial, chip/AI, etc.)",
                             key=f"lnews_{c['ticker']}"):
                    st.session_state[f"lfullnews_{c['ticker']}"] = get_full_news(c["ticker"])
                for a in st.session_state.get(f"lfullnews_{c['ticker']}", []):
                    meta = " · ".join(x for x in (a.get("publisher", ""), a.get("date", "")) if x)
                    st.markdown(f"• [{a['title']}]({a['link']})  \n*{meta}*")
            lcols = st.columns(2)
            for lcol, key, base_label in [(lcols[0], "call", "🟢 CALL — moonshot up"),
                                          (lcols[1], "put", "🔴 PUT — crash down")]:
                label = base_label + ("  👈 **suggested**"
                                      if (lean == "bullish" and key == "call")
                                      or (lean == "bearish" and key == "put") else "")
                o = lp.get(key)
                with lcol:
                    if not o:
                        st.write(f"{label}: n/a")
                        continue
                    pop_txt = f"{o['pop'] * 100:.0f}%" if o["pop"] is not None else "—"
                    if o.get("stale"):
                        warn = ("  \n⚠️ **stale price — no live quote right now** (market closed / this "
                                "strike barely trades). This is the *last trade*, not a live market: the "
                                "real cost to buy (the ask) is usually higher, and with no bid you may not "
                                "be able to sell. Check a live broker before trusting the number.")
                    elif not o.get("has_bid"):
                        warn = "  \n⚠️ **no bid** — nothing to sell into; you may be unable to exit this contract."
                    else:
                        warn = ""
                    st.markdown(f"**{label}**  \n"
                                f"\\${o['option_strike']:g} · exp {o['option_expiry']}  \n"
                                f"\\${o['option_entry_premium']}/sh = **\\${o['cost']:,.0f}**/contract  \n"
                                f"needs a {o['pct_needed']:.0f}% move · breakeven \\${o['be']}  \n"
                                f"**P(profit) {pop_txt}**" + warn)
                    lbud = int(st.session_state.get("feed_budget", 500))
                    n = int(min(lbud, portfolio.get_cash(DB)) // o["cost"]) if o["cost"] > 0 else 0
                    if st.button(f"🎰 Gamble {n} (${n * o['cost']:,.0f})", key=f"lotto_{c['ticker']}_{key}"):
                        if n < 1:
                            st.warning(f"1 contract is \\${o['cost']:,.0f} — over your budget/cash.")
                        else:
                            portfolio.buy(DB, c["ticker"], o["option_type"], o["option_strike"],
                                          o["option_expiry"], n, o["option_entry_premium"], c["spot"])
                            st.session_state["last_buy_msg"] = (
                                f"🎰 Gambled {n} {c['ticker']} {key.upper()} for ${n * o['cost']:,.0f}. See 💰 Paper.")
                            st.rerun()
    st.caption("If one hits it can pay many multiples — that's the trade. But expect to lose the whole "
               "premium almost every time. This is the 🔴 moonshot tier by design.")

# ─────────────────────────── Paper account ───────────────────────────
if nav == "Paper":
    # PIIP audit 2026-08, per user request: the starting balance was already fully configurable
    # in the data layer (portfolio.get_or_create()/reset() both take a `start` param, $1,000 was
    # only ever the app.py CALL SITE's hardcoded default) -- just needed a UI to actually offer it.
    # Caption reads the account's REAL start_cash, not a hardcoded "$1,000", so it stays accurate
    # after someone resets with a different amount.
    _paper_start = _fetch_paper_summary()["start"]
    _info_header(f"💰 Paper — ${_paper_start:,.0f} account", (
        "Every trade is saved to disk (iip.db) and survives a crash or restart. Fills at "
        "mid-price, no commission — real trading pays the bid-ask spread + fees, so treat "
        "results as optimistic. Open positions marked live."), size="1.3rem")
    btns = st.columns([1, 1, 4])
    if btns[0].button("🔄 Refresh prices"):
        st.session_state["paper_stale"] = True
        st.rerun()
    if btns[1].button("↻ Reset account"):
        st.session_state["confirm_paper_reset"] = True
    if st.session_state.get("confirm_paper_reset"):
        with st.container(border=True):
            st.warning("⚠️ This permanently deletes every open/closed position and the equity "
                      "history for this account. There's no undo.")
            new_start = st.number_input(
                "Starting balance", min_value=1.0, max_value=10_000_000.0,
                value=float(_paper_start), step=100.0, format="%.2f",
                key="paper_reset_start_input",
                help="Pick whatever you want to paper-trade with — $1,000 is just the default, "
                     "not a limit.")
            cc1, cc2 = st.columns(2)
            if cc1.button("Yes, reset everything", type="primary", key="confirm_reset_yes"):
                portfolio.reset(DB, float(new_start))
                st.session_state["paper_stale"] = True
                st.session_state["confirm_paper_reset"] = False
                st.success(f"Account reset to ${new_start:,.2f}.")
                st.rerun()
            if cc2.button("Cancel", key="confirm_reset_no"):
                st.session_state["confirm_paper_reset"] = False
                st.rerun()
    st.caption("Prices are marked to market when you **open the tab, add/close a position, or hit "
               "Refresh** — not on every click, so selecting rows is instant. No timer (that would "
               "hammer the data feed).")
    try:
        s = _fetch_paper_summary()
        _render_positions_block(s, "paper")

        if s["closed"]:
            st.subheader("Closed trades")
            _c = pd.DataFrame(s["closed"])
            cdf = pd.DataFrame({
                "Contract": _c["ticker"] + " " + _c["opt_type"].str.upper() + " $" + _c["strike"].map(lambda x: f"{x:g}"),
                "Expiry": _c["expiry"],
                "Qty": _c["contracts"],
                "Entry": _c["entry_premium"],
                "Exit": _c["exit_premium"],
                "Realized P&L": _c["realized_pnl"],
            })
            # _pnl_text_color/_signed_money, not a local lambda -- this table was hand-rolling its
            # own green/red pair (#22c55e/#ef4444), a different color than every other P&L display
            # in the app (#79ed8e/#ff8080), visibly mismatched right below the Open positions table
            # on this exact same page.
            styled_c = (cdf.style
                        .map(_pnl_text_color, subset=["Realized P&L"])
                        .format({"Entry": "${:.2f}", "Exit": "${:.2f}", "Realized P&L": _signed_money}))
            # width="stretch" + explicit widths -- same "sad open space" fix as Home/Positions above.
            closed_col_cfg = {c: st.column_config.Column(c, help=glossary.help_for(c),
                              width="medium" if c in ("Contract", "Expiry") else "small")
                              for c in cdf.columns}
            st.dataframe(styled_c, width="stretch", hide_index=True, column_config=closed_col_cfg)
            st.metric("Realized P&L (all closed trades)", f"${s['realized_pnl']:+,.0f}")

        with st.container(border=True):
            _info_header("🤖 AI opinion on how you're doing (optional, real spend)", (
                "One Claude Haiku call that reads your actual equity, P&L, open positions, "
                "and recent closed trades — plain-English coaching commentary, not a trade "
                "recommendation. Never invents numbers; reasons only from the account state "
                "above. Real spend, ~$0.01/run, same cost caps as the Research page's AI "
                "section."), size="1.3rem")
            if not os.getenv("ANTHROPIC_API_KEY"):
                _ai_key_missing_notice()
            else:
                pf_question = st.text_input(
                    "Anything specific you want the agent to consider? (optional)",
                    key="ai_portfolio_q",
                    placeholder="e.g. Am I too concentrated in one ticker right now?")
                if st.button("🤖 Get an AI read on my account", key="ai_portfolio_btn"):
                    with st.spinner("Running 1 Claude call…"):
                        try:
                            client = agents.LLMClient(dry_run=False)
                            out, cost = agents.interpret_portfolio(
                                s, client, user_question=pf_question.strip() or None)
                            st.session_state["ai_portfolio"] = {"out": out, "cost": cost,
                                                                "question": pf_question.strip() or None}
                            st.toast(f"AI read done — ${cost:.4f} spent.")
                        except Exception as e:
                            st.error(f"AI call failed: {e}")
                pf_cached = st.session_state.get("ai_portfolio")
                if pf_cached:
                    o = pf_cached["out"]
                    st.caption(f"Last run cost **${pf_cached['cost']:.4f}**")

                    read_tile = _tile_style("🔵")
                    st.markdown(
                        f'<div style="background:{read_tile["bg"]};border:1px solid {read_tile["border"]};'
                        f'border-radius:10px;padding:0.9rem 1.1rem;margin-bottom:0.6rem">'
                        f'<div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.05em;'
                        f'color:{_TEXT_BY_EMOJI["🔵"]};font-weight:700;margin-bottom:0.4rem">🤖 Portfolio Agent</div>'
                        f'<div style="font-size:0.92rem;line-height:1.6;color:#e8ecec">'
                        f'{_esc(o.get("read") or o.get("summary") or "—")}</div></div>',
                        unsafe_allow_html=True)

                    sr_cols = st.columns(2)
                    if o.get("strengths"):
                        g_tile = _tile_style("🟢")
                        items = "".join(f'<div style="margin-bottom:0.3rem">✓ {_esc(x)}</div>'
                                       for x in o["strengths"])
                        with sr_cols[0]:
                            st.markdown(
                                f'<div style="background:{g_tile["bg"]};border:1px solid {g_tile["border"]};'
                                f'border-radius:10px;padding:0.8rem 1rem;height:100%">'
                                f'<div style="font-size:0.72rem;text-transform:uppercase;'
                                f'letter-spacing:0.05em;color:{_TEXT_BY_EMOJI["🟢"]};font-weight:700;'
                                f'margin-bottom:0.4rem">Strengths</div>'
                                f'<div style="font-size:0.85rem;color:#e8ecec">{items}</div></div>',
                                unsafe_allow_html=True)
                    if o.get("risks"):
                        r_tile = _tile_style("🔴")
                        items = "".join(f'<div style="margin-bottom:0.3rem">✗ {_esc(x)}</div>'
                                       for x in o["risks"])
                        with sr_cols[1]:
                            st.markdown(
                                f'<div style="background:{r_tile["bg"]};border:1px solid {r_tile["border"]};'
                                f'border-radius:10px;padding:0.8rem 1rem;height:100%">'
                                f'<div style="font-size:0.72rem;text-transform:uppercase;'
                                f'letter-spacing:0.05em;color:{_TEXT_BY_EMOJI["🔴"]};font-weight:700;'
                                f'margin-bottom:0.4rem">Risks</div>'
                                f'<div style="font-size:0.85rem;color:#e8ecec">{items}</div></div>',
                                unsafe_allow_html=True)
                    if o.get("biggest_uncertainty"):
                        a_tile = _tile_style("🟡")
                        st.markdown(
                            f'<div style="background:{a_tile["bg"]};border:1px solid {a_tile["border"]};'
                            f'border-radius:10px;padding:0.8rem 1rem;margin-top:0.6rem">'
                            f'<span style="color:{_TEXT_BY_EMOJI["🟡"]};font-weight:700">'
                            f'⚠️ Biggest uncertainty:</span> <span style="color:#e8ecec">'
                            f'{_esc(o["biggest_uncertainty"])}</span></div>',
                            unsafe_allow_html=True)

                    # Direct answer to the user's own question, shown LAST -- after all the other
                    # agent context above, not first.
                    if pf_cached.get("question"):
                        if o.get("user_question_answer"):
                            aq_tile = _tile_style("🟡")
                            st.markdown(
                                f'<div style="background:{aq_tile["bg"]};border:1px solid '
                                f'{aq_tile["border"]};border-radius:10px;padding:0.9rem 1.1rem;'
                                f'margin-top:0.6rem"><div style="font-size:0.72rem;'
                                f'text-transform:uppercase;letter-spacing:0.05em;'
                                f'color:{_TEXT_BY_EMOJI["🟡"]};font-weight:700;margin-bottom:0.4rem">'
                                f'❓ Your question: "{_esc(pf_cached["question"])}"</div>'
                                f'<div style="color:#e8ecec">{_esc(o["user_question_answer"])}'
                                f'</div></div>', unsafe_allow_html=True)
                        else:
                            st.caption("⚠️ The AI didn't return a direct answer to your question "
                                      "this run — try rephrasing it or click again.")
    except Exception as e:
        st.error(f"Error: {e}")


def _ai_key_missing_notice():
    """Shared 'no Anthropic key yet' gate for every optional AI-interpretation section (Research,
    Deep Research, Paper) -- one consistent message + a direct link to get a key, instead of each
    section re-explaining it slightly differently."""
    st.info("Add an Anthropic API key to unlock this. Everything else on this page works fully "
            "without it.")
    st.link_button("Get an Anthropic API key →", "https://console.anthropic.com/settings/keys")


# ─────────────────────────── Research ───────────────────────────
if nav == "Research":
    ticker = st.text_input("Ticker", "AAPL").strip().upper()
    if ticker:
        try:
            prices = load_prices(ticker)
            spot = data.last_valid_close(prices)
            if spot is None:
                raise ValueError(f"No valid close price for {ticker!r}")
            st.metric(f"{ticker} — {get_company_info(ticker).get('name') or ticker}", f"${spot:,.2f}")

            # PIIP audit 2026-08: rolling SMAs computed on the FULL price history first, THEN
            # windowed to the last 400 for display -- same as the original st.line_chart version,
            # so SMA200 near the start of the visible window still reflects 200 real days of
            # history, not a truncated average starting from nothing.
            close_full = prices["Close"]
            sma50_full = close_full.rolling(50).mean()
            sma200_full = close_full.rolling(200).mean()
            _render_lightweight_lines(
                {"Close": close_full.tail(400), "SMA50": sma50_full.tail(400),
                 "SMA200": sma200_full.tail(400)},
                key_prefix=f"research_daily_{ticker}",
                colors={"Close": "#87d1ff", "SMA50": "#c792ea", "SMA200": "#79ed8e"})

            with st.container(border=True):
                st.subheader("⚡ Intraday snapshot (for active watching)")
                if st.button("Load / refresh intraday", key=f"intrabtn_{ticker}"):
                    st.session_state[f"intra_{ticker}"] = get_intraday_snapshot(ticker)
                snap = st.session_state.get(f"intra_{ticker}")
                if snap:
                    # Trailing spacer, not a bare st.columns(4) -- same fix applied everywhere else
                    # this session: 4 equal columns across this wide-layout page scatters small
                    # metric values with a lot of dead space between them.
                    ic = st.columns([1, 1, 1, 1, 3])
                    ic[0].metric("Last", f"${snap['last']:.2f}", f"{snap['day_change_pct']:+.1f}% vs open")
                    ic[1].metric("vs VWAP", f"{snap['pct_from_vwap']:+.1f}%")
                    ic[2].metric("Range position", f"{snap['range_pos']:.0f}%")
                    ic[3].metric("Today range", f"${snap['low']:.2f}–${snap['high']:.2f}")
                    _render_lightweight_lines(
                        {"Price": snap["chart"]["price"], "VWAP": snap["chart"]["VWAP"]},
                        key_prefix=f"research_intraday_{ticker}", height=220, intraday=True,
                        colors={"Price": "#87d1ff", "VWAP": "#c792ea"})
                    _info_header("ℹ️ How to read this", (
                        "VWAP = today's volume-weighted avg price. Above VWAP = buyers in "
                        "control intraday; below = sellers. Range position: 0% = at the day's "
                        "low, 100% = at the high. Descriptive only — this does NOT predict the "
                        "next few hours. For same-day / 0–2 DTE options, theta + gamma are "
                        "brutal; use this to time your exit."), size="0.85rem")
                else:
                    st.caption("Click **Load intraday** for today's VWAP, range, and move (market-hours only).")

            with st.container(border=True):
                st.subheader("Deterministic forecasts (free)")
                st.caption("The mechanical baseline's read at three horizons. Direction is **~a coin flip** — "
                           "treat each as one input, not a call.")
                cols = st.columns(3)
                det_fcs, chains = {}, {}   # reused below by the AI interpretation section, not recomputed
                for i, H in enumerate((7, 30, 90)):
                    fc = baseline.deterministic_forecast(ticker, prices, H)
                    chain = load_chain(ticker, H)
                    opt = baseline.simulated_option(spot, chain, fc["stock_direction"])
                    pred.log_if_new({**fc, "spot": spot, **(opt or {}),
                                     "reasoning": {"mode": "research_view"}}, DB)   # feed the Scorecard
                    det_fcs[H], chains[H] = fc, chain
                    with cols[i]:
                        with st.container(border=True):
                            arrow = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}[fc["stock_direction"]]
                            st.markdown(f"#### {H}-day  {arrow} {fc['stock_direction'].title()}")
                            st.metric("Expected move", f"±{fc['stock_expected_move_pct']}%")
                            st.caption(f"P(up) {fc['stock_prob_up']} · conf {fc['confidence']}")
                            if opt:
                                g = opt.get("option_greeks", {})
                                st.markdown(f"**{opt['option_type'].title()} \\${opt['option_strike']}**  \n"
                                            f"exp {opt['option_expiry']}")
                                st.caption(f"Premium \\${opt['option_entry_premium']} · IV {opt['option_iv_pct']}%  \n"
                                           f"Δ {g.get('delta')} · θ/day {g.get('theta_per_day')} · vega {g.get('vega')}")

            with st.container(border=True):
                _info_header("🤖 AI interpretation (optional, real spend)", (
                    "Runs 4 short Claude Haiku calls (technical / options / skeptic / "
                    "executive) that interpret the deterministic numbers above — they never "
                    "invent their own. Real spend, ~$0.02–0.03/run, governed by the same "
                    "$3/day · $0.15/run cap as the CLI. Every result is logged to the 🎯 "
                    "Scorecard and graded against the deterministic baseline above — the AI "
                    "is on trial, not trusted by default."), size="1.3rem")
                if not os.getenv("ANTHROPIC_API_KEY"):
                    _ai_key_missing_notice()
                else:
                    ai_key = f"ai_interp_{ticker}"
                    ai_question = st.text_input(
                        "Anything specific you want the agents to consider? (optional)",
                        key=f"ai_q_{ticker}",
                        placeholder="e.g. How does this look for a 2-week hold specifically?")
                    if st.button("🤖 Run AI interpretation (~$0.02–0.03)", key=f"ai_btn_{ticker}"):
                        with st.spinner("Running 4 Claude calls…"):
                            try:
                                client = agents.LLMClient(dry_run=False)
                                llm_fcs, cost, outs = agents.run_ticker(
                                    ticker, prices, spot, chains, det_fcs, client,
                                    user_question=ai_question.strip() or None)
                                for H, lfc in llm_fcs.items():
                                    lopt = baseline.simulated_option(spot, chains[H], lfc["stock_direction"])
                                    pred.log_if_new({**lfc, "spot": spot, **(lopt or {})}, DB)
                                st.session_state[ai_key] = {"fcs": llm_fcs, "cost": cost, "outs": outs,
                                                            "question": ai_question.strip() or None}
                                st.toast(f"AI interpretation done — ${cost:.4f} spent.")
                            except Exception as e:
                                st.error(f"AI call failed: {e}")
                    cached = st.session_state.get(ai_key)
                    if cached:
                        st.caption(f"Last run cost **${cached['cost']:.4f}** · logged to the 🎯 Scorecard "
                                  "alongside the deterministic baseline above.")
                        # NOT another 7/30/90-day card grid -- the deterministic section above
                        # already shows direction/expected-move/P(up) for each horizon, and the AI's
                        # own per-horizon numbers next to it in the same card shape read as an
                        # unexplained duplicate (confirmed live). The genuinely NEW thing the AI
                        # adds is the qualitative reasoning below, not a repeat of numbers already free.
                        exec_reason = (cached["fcs"].get(30) or next(iter(cached["fcs"].values()), {})) \
                            .get("reasoning", {})
                        st.markdown("**Executive Agent**")
                        if exec_reason.get("bull_factors"):
                            st.markdown("🟢 " + "  \n🟢 ".join(_md_safe(x) for x in exec_reason["bull_factors"]))
                        if exec_reason.get("bear_factors"):
                            st.markdown("🔴 " + "  \n🔴 ".join(_md_safe(x) for x in exec_reason["bear_factors"]))
                        st.divider()
                        st.markdown("**Specialist reads**")
                        for name in ("technical", "options", "skeptic"):
                            summ = cached["outs"].get(name, {}).get("summary")
                            if summ:
                                st.markdown(f"**{name.title()} Agent:** {_md_safe(summ)}")
                        if exec_reason.get("biggest_uncertainty"):
                            st.warning(f"⚠️ Executive Agent — biggest uncertainty: "
                                      f"{_md_safe(exec_reason['biggest_uncertainty'])}")

                        # Direct answer to the user's own question, shown LAST -- after all the
                        # other agent context above, not first.
                        if cached.get("question"):
                            if exec_reason.get("user_question_answer"):
                                q_tile = _tile_style("🔵")
                                st.markdown(
                                    f'<div style="background:{q_tile["bg"]};border:1px solid '
                                    f'{q_tile["border"]};border-radius:10px;padding:0.9rem 1.1rem;'
                                    f'margin-top:0.9rem"><div style="font-size:0.72rem;'
                                    f'text-transform:uppercase;letter-spacing:0.05em;'
                                    f'color:{_TEXT_BY_EMOJI["🔵"]};font-weight:700;margin-bottom:0.4rem">'
                                    f'❓ Your question: "{_esc(cached["question"])}"</div>'
                                    f'<div style="color:#e8ecec">{_esc(exec_reason["user_question_answer"])}'
                                    f'</div></div>', unsafe_allow_html=True)
                            else:
                                st.caption("⚠️ The AI didn't return a direct answer to your question "
                                          "this run — try rephrasing it or click again.")

            with st.container(border=True):
                st.subheader("Options — nearest ~30-day expiry")
                om = det.option_metrics(spot, load_chain(ticker, 30))
                oc = st.columns([1, 1, 1, 1, 3])
                oc[0].metric("ATM IV", f"{om['atm_iv_pct']}%", help=glossary.help_for("Implied volatility"))
                oc[1].metric("Exp move (straddle)", f"{om['expected_move_straddle_pct']}%",
                             help=glossary.help_for("Expected move"))
                oc[2].metric("Exp move (IV)", f"{om['expected_move_iv_pct']}%")
                oc[3].metric("Put/Call OI", om["put_call_oi_ratio"],
                             help=glossary.help_for("Put/Call OI ratio"))
                st.caption("Straddle ≈ realized-move basis; IV-move usually ≥ it (the vol-risk premium). "
                           "Theta is the daily bleed you fight when long options.")

            with st.container(border=True):
                st.subheader("💵 Option price example — how many contracts your budget buys")
                st.caption("1 contract controls **100 shares**, so it costs the premium × 100. "
                           "This is why options on high-priced stocks need real capital.")
                ope_hz = st.selectbox("Expiry (~horizon days)", [7, 30, 90], index=1, key="ope_hz")
                bcol1, bcol2 = st.columns(2)
                budget_a = bcol1.number_input("Budget A ($)", value=50, step=10, min_value=1, key="ope_a")
                budget_b = bcol2.number_input("Budget B ($)", value=100, step=10, min_value=1, key="ope_b")
                ope_chain = load_chain(ticker, ope_hz)

                def _atm(side_df):
                    d = side_df.copy()
                    d["_x"] = (d["strike"] - spot).abs()
                    return d.loc[d["_x"].idxmin()]

                ope_rows = []
                for label, row in [("Call", _atm(ope_chain["calls"])), ("Put", _atm(ope_chain["puts"]))]:
                    prem = det._mid(row)
                    per_contract = prem * 100 if prem == prem else float("nan")
                    ope_rows.append({
                        "Type": label,
                        "ATM strike": f"${float(row['strike']):,.0f}",
                        "Premium/share": f"${prem:,.2f}" if prem == prem else "n/a",
                        "Cost / 1 contract": f"${per_contract:,.0f}" if per_contract == per_contract else "n/a",
                        f"Contracts for ${int(budget_a)}": int(budget_a // per_contract) if per_contract == per_contract and per_contract > 0 else 0,
                        f"Contracts for ${int(budget_b)}": int(budget_b // per_contract) if per_contract == per_contract and per_contract > 0 else 0,
                    })
                st.table(pd.DataFrame(ope_rows).set_index("Type"))
                st.caption("**0 contracts** means your budget is below one contract's cost — the "
                           "'Cost / 1 contract' column is the real entry ticket. (Premium = bid/ask mid; "
                           "real fills are worse. This is the *cost to buy*, not a P&L projection.)")

            with st.container(border=True):
                st.subheader("🧮 Options chain — pick a strike & paper-buy")
                render_option_chain(ticker, spot)

            with st.container(border=True):
                st.subheader("Historical 30-day move distribution")
                hm = det.historical_move_distribution(prices, 30)
                if "std_pct" in hm:
                    st.bar_chart(pd.DataFrame({"% move": [hm["p10_pct"], hm["p50_pct"], hm["p90_pct"]]},
                                              index=["p10 (down)", "p50 (median)", "p90 (up)"]))
                    st.caption(f"Over {hm['n']} historical windows · prob_up {hm['prob_up']} · 1σ ≈ {hm['std_pct']}%")

            with st.container(border=True):
                st.subheader("📰 Recent articles")
                fn = get_full_news(ticker)
                if fn:
                    for a in fn:
                        meta = " · ".join(x for x in (a["publisher"], a["date"]) if x)
                        st.markdown(f"• [{a['title']}]({a['link']})  \n*{meta}*")
                else:
                    st.write("No articles found right now.")
            st.caption("These forecasts are now auto-logged to the 🎯 Scorecard so it fills as you "
                       "use the app. (The same AI layer is also available from the terminal: "
                       "`python -m iip.cli research " + ticker + " --llm --live-llm`.)")
        except Exception as e:
            st.error(f"Error: {e}")

# ─────────────────────────── Scorecard ───────────────────────────
if nav == "Scorecard":
    st.caption("Your forward track record. Empty until logged predictions resolve (7-day soonest).")
    if st.button("Load scorecard"):
        try:
            scorer.resolve_due(DB)
            rep = scorer.report(DB)["overall"]
            if rep.get("n", 0) == 0:
                st.warning("No resolved predictions yet — run some `research` forecasts and come back after the horizons elapse.")
            else:
                # Trailing spacer, not a bare st.columns(4) -- same fix applied to every other
                # metrics row this session (Paper, 0DTE, Research): 4 equal columns across this
                # wide-layout page scatters small values with a lot of dead space between them.
                m = st.columns([1, 1, 1, 1, 3])
                m[0].metric("Resolved", rep["n"])
                m[1].metric("Hit rate", f"{rep['hit_rate']:.0%}", help=f"95% CI {rep['hit_rate_ci']}")
                m[2].metric("Brier", rep["brier"], help="0.25 = coin flip; lower is better")
                m[3].metric("Alpha vs SPY", f"{rep.get('alpha_vs_spy_pct')}%")
            rows = pred.all_rows(DB)
            if rows:
                df = pd.DataFrame([dict(r) for r in rows])
                display_cols = ["ticker", "source", "horizon_days", "stock_direction", "confidence",
                                "resolved", "hit", "realized_return_pct", "spy_return_pct",
                                "ts", "resolve_due"]
                display_cols = [c for c in display_cols if c in df.columns]
                view = df[display_cols].copy()
                if "hit" in view:
                    view["hit"] = view["hit"].map({1: "✅ hit", 0: "❌ miss"}).fillna("⏳ pending")
                if "resolved" in view:
                    view["resolved"] = view["resolved"].map({1: "Yes", 0: "No"})
                # Explicit width on every column, not just the 2 with NumberColumn formatting --
                # "stretch" with only some columns configured still pads the REST out evenly (same
                # "sad open space" pattern fixed everywhere else this session), and ts/resolve_due
                # (real timestamps) are the only columns that actually need "medium"+.
                sc_col_cfg = {
                    "ticker": st.column_config.Column("Ticker", width="small"),
                    "source": st.column_config.Column("Source", width="small"),
                    "horizon_days": st.column_config.NumberColumn("Horizon (d)", width="small"),
                    "stock_direction": st.column_config.Column("Direction", width="small"),
                    "confidence": st.column_config.NumberColumn("Confidence", format="%.0f", width="small"),
                    "resolved": st.column_config.Column("Resolved", width="small"),
                    "hit": st.column_config.Column("Result", width="small"),
                    "realized_return_pct": st.column_config.NumberColumn("Realized %", format="%.1f%%", width="small"),
                    "spy_return_pct": st.column_config.NumberColumn("SPY %", format="%.1f%%", width="small"),
                    "ts": st.column_config.Column("Logged", width="medium"),
                    "resolve_due": st.column_config.Column("Due", width="medium"),
                }
                # Explicit height sized to every row -- same fix as Screener/Watchlist: without it
                # Streamlit caps the box to a fixed default height and scrolls internally.
                st.dataframe(view, width="stretch", hide_index=True, column_config=sc_col_cfg,
                            height=(len(view) + 1) * 35 + 3)
                with st.expander("Full raw table (all columns)"):
                    st.dataframe(df, width="stretch")
        except Exception as e:
            st.error(f"Error: {e}")

# ─────────────────────────── Backtest ───────────────────────────
if nav == "Backtest":
    st.caption("Point-in-time deterministic replay — real results now, no waiting. Stock thesis only "
               "(options are forward-only). LLM is not backtested (look-ahead).")
    tks = st.text_input("Tickers (comma-separated)", "AAPL,MSFT,NVDA,AMZN,GOOGL,META,JPM,XOM")
    if st.button("Run backtest", type="primary"):
        try:
            with st.spinner("Replaying years of point-in-time history…"):
                db, nlog = bt.run_backtest([t.strip().upper() for t in tks.split(",")], db="iip_bt_ui.db")
            rep = scorer.report(db)
            o = rep["overall"]
            m = st.columns([1, 1, 1, 1, 3])
            m[0].metric("Predictions", o["n"])
            m[1].metric("Hit rate", f"{o['hit_rate']:.0%}", help=f"95% CI {o['hit_rate_ci']}  ·  beats-coinflip={o['beats_coinflip']}")
            m[2].metric("Brier", o["brier"], help="0.25 = coin flip")
            m[3].metric("Alpha vs SPY", f"{o.get('alpha_vs_spy_pct')}%")
            bh = rep["by_horizon"]
            st.bar_chart(pd.DataFrame(
                {"hit rate": {f"{h}d": bh[h].get("hit_rate", 0) for h in (7, 30, 90)}}))
            st.caption("Reading it honestly: a hit-rate CI that includes 50% and a Brier near 0.25 "
                       "mean **no demonstrated edge** — the correct, valuable answer.")
        except Exception as e:
            st.error(f"Error: {e}")

# ─────────────────────────── Deep Research ───────────────────────────
if nav == "Deep Research":
    st.caption("A **data-first research dossier** that *reduces uncertainty* — every field carries a "
               "**confidence + source**, and anything with no free source is shown as **UNKNOWN**. "
               "It does **not** recommend buying or selling. Every field here is real, verifiable "
               "data — an optional AI read below can interpret it, but never invents its own numbers.")
    dtk = st.text_input("Ticker", "NTLA", key="deep_tk").strip().upper()
    if st.button("🔬 Build dossier", type="primary") and dtk:
        st.session_state["dossier_tk"] = dtk
    tk_show = st.session_state.get("dossier_tk")
    if tk_show:
        try:
            with st.spinner(f"Researching {tk_show} (free sources: Yahoo, ClinicalTrials.gov)…"):
                D = get_dossier(tk_show)
        except Exception as e:
            st.error(f"Couldn't build dossier for {tk_show}: {e}")
            D = None
        if D:
            st.header(f"{D['company']}  ·  {tk_show}")
            st.caption(f"{D['sector']} / {D['industry']} · ${D['spot']:,.2f} · generated {D['generated'][:16]}Z")
            render_dossier(D, use_expanders=True)

            with st.container(border=True):
                _info_header("🤖 AI interpretation (optional, real spend)", (
                    "One Claude Haiku call that interprets the dossier above — bull case, "
                    "bear case, and the biggest open uncertainty. Never invents numbers; "
                    "reasons only from the fields already shown, each with its own confidence/"
                    "source. Real spend, ~$0.01/run, same cost caps as the Research page's AI "
                    "section."), size="1.3rem")
                if not os.getenv("ANTHROPIC_API_KEY"):
                    _ai_key_missing_notice()
                else:
                    dr_ai_key = f"ai_dossier_{tk_show}"
                    dr_question = st.text_input(
                        "Anything specific you want the agent to consider? (optional)",
                        key=f"ai_dossier_q_{tk_show}",
                        placeholder="e.g. How much of the bull case depends on the pipeline data specifically?")
                    if st.button("🤖 Get AI read on this dossier", key=f"ai_dossier_btn_{tk_show}"):
                        with st.spinner("Running 1 Claude call…"):
                            try:
                                client = agents.LLMClient(dry_run=False)
                                out, cost = agents.interpret_dossier(
                                    D, client, user_question=dr_question.strip() or None)
                                st.session_state[dr_ai_key] = {"out": out, "cost": cost,
                                                               "question": dr_question.strip() or None}
                                st.toast(f"AI read done — ${cost:.4f} spent.")
                            except Exception as e:
                                st.error(f"AI call failed: {e}")
                    dr_cached = st.session_state.get(dr_ai_key)
                    if dr_cached:
                        o = dr_cached["out"]
                        st.caption(f"Last run cost **${dr_cached['cost']:.4f}**")
                        st.markdown("**Research Agent**")
                        st.markdown(f"**Bull case:** {_md_safe(o.get('bull_case', '—'))}")
                        st.markdown(f"**Bear case:** {_md_safe(o.get('bear_case', '—'))}")
                        if o.get("biggest_uncertainty"):
                            st.warning(f"⚠️ Biggest uncertainty: {_md_safe(o['biggest_uncertainty'])}")
                        st.caption(f"Confidence: {o.get('confidence', '—')}")

                        # Direct answer to the user's own question, shown LAST -- after all the
                        # other agent context above, not first.
                        if dr_cached.get("question"):
                            if o.get("user_question_answer"):
                                q_tile = _tile_style("🔵")
                                st.markdown(
                                    f'<div style="background:{q_tile["bg"]};border:1px solid '
                                    f'{q_tile["border"]};border-radius:10px;padding:0.9rem 1.1rem;'
                                    f'margin-top:0.9rem"><div style="font-size:0.72rem;'
                                    f'text-transform:uppercase;letter-spacing:0.05em;'
                                    f'color:{_TEXT_BY_EMOJI["🔵"]};font-weight:700;margin-bottom:0.4rem">'
                                    f'❓ Your question: "{_esc(dr_cached["question"])}"</div>'
                                    f'<div style="color:#e8ecec">{_esc(o["user_question_answer"])}'
                                    f'</div></div>', unsafe_allow_html=True)
                            else:
                                st.caption("⚠️ The AI didn't return a direct answer to your question "
                                          "this run — try rephrasing it or click again.")

# ─────────────────────────── Ticker Page ───────────────────────────
if nav == "Ticker Page":
    st.caption("One ticker's full picture: the Deep Research snapshot + every bull/bear thesis you've "
               "logged on it over time, dated and scored — so you can look back and see if the call played "
               "out. Log a new thesis without leaving the page.")
    known_tickers = sorted({e["ticker"] for e in journal.all_entries(DB)})
    tp_pick = st.selectbox("Ticker", ["— new ticker —"] + known_tickers, key="tp_pick")
    tp_free = st.text_input("...or type a new ticker", "", key="tp_free").strip().upper()
    tp_tk = tp_free or (tp_pick if tp_pick != "— new ticker —" else "")

    if not tp_tk:
        st.info("Pick a ticker you've already logged a thesis on, or type a new one, to open its page.")
    else:
        try:
            with st.spinner(f"Researching {tp_tk}…"):
                D = get_dossier(tp_tk)
            st.header(f"{D['company']}  ·  {tp_tk}")
            st.caption(f"{D['sector']} / {D['industry']} · ${D['spot']:,.2f} · generated {D['generated'][:16]}Z")
            with st.expander("📊 Snapshot (Deep Research dossier)"):
                render_dossier(D, use_expanders=False)
        except Exception as e:
            st.warning(f"Couldn't build the research snapshot for {tp_tk}: {e}")

        _render_decision_form(f"➕ Log a thesis on {tp_tk}", default_ticker=tp_tk, key_suffix="_tp")
        if st.session_state.get("journal_msg"):
            st.success(st.session_state.pop("journal_msg"))

        st.subheader(f"📜 Your theses on {tp_tk}")
        tp_entries = journal.entries_for_ticker(DB, tp_tk)
        if not tp_entries:
            st.write("No theses logged yet for this ticker — use the form above.")
        due_ids_tp = {d["id"] for d in journal.due_for_review(DB)}
        for e in tp_entries:
            if e["status"] == "OPEN":
                _render_open_entry(e, due_ids_tp, key_prefix="tp_")
        reviewed_tp = [e for e in tp_entries if e["status"] == "REVIEWED"]
        if reviewed_tp:
            st.markdown("**Reviewed:**")
            for e in reviewed_tp:
                _render_reviewed_entry(e, key_prefix="tp_")

# ─────────────────────────── Catalyst Radar ───────────────────────────
if nav == "Catalysts":
    _info_header("📡 Catalyst Radar", (
        "Public but low-visibility catalysts — earnings dates, SEC 8-K filings, insider Form "
        "4 trades, and (for biotech) clinical trial readouts. All technically disclosed, none "
        "of it requires a paid feed — the point is most of it doesn't get headline coverage "
        "before the market reacts. NOT insider information: everything here is public SEC/"
        "ClinicalTrials.gov data."), size="1.3rem")
    default_watch = sorted({e["ticker"] for e in journal.all_entries(DB)})
    cat_extra = st.text_input("Add tickers (comma-separated)", "", key="cat_extra")
    cat_universe = st.checkbox("Also include the full scan universe (~90 names, slower)",
                               value=False, key="cat_universe")

    # ETFs (SPY/QQQ/IWM/DIA) don't themselves file 8-Ks or report earnings -- confirmed live via
    # SEC EDGAR (SPY/QQQ/DIA only file fund-administrative paperwork: NPORT-P, 497, N-30D, etc.;
    # IWM doesn't even resolve to a ticker->CIK match). Per explicit user decision: scan each
    # ETF's real, LIVE top-10 holdings (yfinance funds_data, never a hardcoded list that could
    # drift stale) instead -- an ETF's actual catalysts are really its constituents' catalysts.
    # Kept in its OWN session_state key (not folded straight into cat_picks) so it stays part of
    # `candidates` across reruns -- otherwise a previously-added ETF holding would vanish from the
    # multiselect's own options list (Streamlit requires every selected value to also be a valid
    # option) as soon as this button isn't the thing that just ran.
    if "cat_etf_universe" not in st.session_state:
        st.session_state["cat_etf_universe"] = set()

    candidates = list(default_watch)
    extra_tickers = [t.strip().upper() for t in cat_extra.split(",") if t.strip()]
    if extra_tickers:
        candidates += extra_tickers
    if cat_universe:
        candidates += list(scanner.NAMES.keys())
    candidates += list(st.session_state["cat_etf_universe"])
    candidates = sorted(set(candidates))

    # Streamlit forbids passing both `default=` and writing to st.session_state[key] for the same
    # widget in one run -- default= only ever matters before the key exists at all, so the correct
    # fix is to seed session_state directly on first run and drop default= from the widget call,
    # not pass both and hope Streamlit reconciles them.
    if "cat_picks" not in st.session_state:
        st.session_state["cat_picks"] = default_watch[:15]

    # Auto-select anything just typed into "Add tickers" — otherwise typing a ticker here and
    # hitting Scan silently does nothing until the user separately re-picks it in the multiselect
    # below, which isn't an obvious two-step flow.
    if extra_tickers:
        already_picked = set(st.session_state.get("cat_picks", default_watch[:15]))
        st.session_state["cat_picks"] = sorted(already_picked | set(extra_tickers))

    if st.button(f"🔄 Scan {'/'.join(zd.INDEX_TICKERS)} top holdings", key="cat_etf_scan_btn"):
        with st.spinner(f"Fetching top holdings for {', '.join(zd.INDEX_TICKERS)}…"):
            etf_universe: set[str] = set()
            for etf in zd.INDEX_TICKERS:
                etf_universe.update(get_etf_holdings(etf))
        if not etf_universe:
            st.warning("Couldn't fetch ETF holdings right now — try again in a moment.")
        else:
            st.session_state["cat_etf_universe"] |= etf_universe
            already_picked = set(st.session_state.get("cat_picks", default_watch[:15]))
            new_picks = sorted(already_picked | etf_universe)
            st.session_state["cat_picks"] = new_picks
            candidates = sorted(set(candidates) | etf_universe)   # keep this run's options in sync too
            with st.spinner(f"Checking earnings, SEC filings, and trial dates for "
                            f"{len(new_picks)} ticker(s)…"):
                st.session_state["cat_rows"] = get_catalyst_rows(tuple(new_picks))
            st.toast(f"Added {len(etf_universe)} tickers from {', '.join(zd.INDEX_TICKERS)}'s "
                    f"top holdings and scanned.")

    cat_picks = st.multiselect("Scan these tickers", candidates, key="cat_picks")

    if st.button("🔄 Scan for catalysts", type="primary") and cat_picks:
        with st.spinner(f"Checking earnings, SEC filings, and trial dates for {len(cat_picks)} ticker(s)…"):
            st.session_state["cat_rows"] = get_catalyst_rows(tuple(cat_picks))

    rows = st.session_state.get("cat_rows")
    if not rows:
        st.info("Pick tickers above and hit **Scan for catalysts**.")
    else:
        near = [r for r in rows if r["days"] is not None and abs(r["days"]) <= 14]
        if near:
            st.warning("⚡ **Within 14 days (past or upcoming):** " + ", ".join(
                f"{r['ticker']} ({r['kind']}, {r['days']:+d}d)" for r in near))
        rows_sorted = sorted(rows, key=lambda r: (r["days"] is None, abs(r["days"]) if r["days"] is not None else 0))
        for r in rows_sorted:
            days = r["days"]
            emoji = "🟡" if (days is not None and abs(days) <= 3) else \
                    "🟢" if (days is not None and abs(days) <= 14) else "⚪"
            tile = _tile_style(emoji)
            days_txt = f"{days:+d}d" if days is not None else "—"
            link_html = (f' · <a href="{r["url"]}" target="_blank" style="color:#87d1ff">open ↗</a>'
                        if r.get("url") else "")
            st.markdown(
                f'<div style="background:{tile["bg"]};border:1px solid {tile["border"]};'
                'border-radius:8px;padding:0.55rem 0.9rem;margin-bottom:0.3rem;display:flex;'
                'align-items:center;justify-content:space-between;gap:0.75rem">'
                f'<div><b>{r["ticker"]}</b> · {r["kind"]}'
                f'<span style="color:#8b9a9d;font-size:0.82rem"> · {r["date"]}{link_html}</span></div>'
                f'<div style="font-family:ui-monospace,Consolas,monospace;font-weight:700;'
                f'color:{_TEXT_BY_EMOJI[emoji]};white-space:nowrap">{days_txt}</div>'
                '</div>', unsafe_allow_html=True)
        st.caption("**FDA PDUFA / decision dates have no free calendar** and are NOT included — that gap "
                   "is real; those still have to be found manually in filings/press releases.")

# ─────────────────────────── Glossary ───────────────────────────
if nav == "Glossary":
    st.caption("Every term on the platform, in plain English with examples. Hover the ⓘ icons and column "
               "headers anywhere in the app for the short version — this tab is the full reference.")
    q = st.text_input("🔎 Search terms", "", key="gloss_search").strip().lower()

    shown_any = False
    for category, terms in glossary.GLOSSARY_GROUPS:
        matches = [(t, d) for t, d in terms if not q or q in t.lower() or q in d.lower()]
        if not matches:
            continue
        shown_any = True
        st.markdown(f"### {category}")
        cards = ["<div style='display:flex;flex-direction:column;gap:0.55rem;margin-bottom:0.4rem'>"]
        for term, desc in matches:
            cards.append(
                "<div style='background:rgba(255,255,255,0.035);border:1px solid rgba(255,255,255,0.09);"
                "border-left:3px solid #38bdf8;border-radius:10px;padding:0.65rem 0.9rem'>"
                f"<div style='font-size:1.04rem;font-weight:700;color:#87d1ff;margin-bottom:0.2rem'>{_esc(term)}</div>"
                f"<div style='color:#d7dee7;line-height:1.55;font-size:0.94rem'>{_esc(desc)}</div></div>")
        cards.append("</div>")
        st.markdown("\n".join(cards), unsafe_allow_html=True)
    if not shown_any:
        st.info(f"No terms match “{q}”. Try a shorter word (e.g. 'call', 'IV', 'breakeven').")

# ─────────────────────────── Feedback / bug reports ───────────────────────────
if nav == "Feedback":
    st.caption("Found a bug, or have an idea? This opens a **pre-filled GitHub issue** in your "
               "browser — nothing is sent automatically, and nothing about your account, "
               "positions, or API keys is included unless you paste it in yourself. You'll need "
               "a free GitHub account to actually submit it (PIIP itself never touches GitHub).")

    kind = st.radio("Type", ["🐞 Bug report", "💡 Feature idea", "💬 General feedback"],
                    horizontal=True, key="fb_kind")
    title = st.text_input("Short summary", key="fb_title",
                          placeholder="e.g. Screener chart doesn't update after selecting a row")

    if kind == "🐞 Bug report":
        steps = st.text_area("What did you do, and what happened? (steps to reproduce help a lot)",
                             key="fb_steps", height=120)
        expected = st.text_input("What did you expect to happen instead?", key="fb_expected")
        body = (f"**Type:** Bug report\n\n"
               f"**Steps / what happened:**\n{steps or '_(not filled in)_'}\n\n"
               f"**Expected:**\n{expected or '_(not filled in)_'}")
        label = "bug"
    else:
        details = st.text_area("Details", key="fb_details", height=140,
                               placeholder="What would you like to see, or what's on your mind?")
        body = f"**Type:** {kind}\n\n{details or '_(not filled in)_'}"
        label = "enhancement" if kind == "💡 Feature idea" else "feedback"
    body += "\n\n---\n*Filed from PIIP's in-app Feedback page.*"

    st.markdown("**Preview**")
    with st.container(border=True):
        st.markdown(body)

    issue_url = (f"https://github.com/{GITHUB_REPO}/issues/new?" +
                urlencode({"title": title or f"{kind}: (no summary provided)",
                          "body": body, "labels": label}))
    st.link_button("📤 Open in GitHub to submit →", issue_url, type="primary")
    st.caption(f"Reports go to github.com/{GITHUB_REPO}/issues — you can review/edit everything "
              "on GitHub before it's actually posted.")

# ─────────────────────────── Decision Journal ───────────────────────────
if nav == "Journal":
    st.caption("Log **decisions, not just trades** — your thesis, what would prove you WRONG, and an exit "
               "plan — then review the outcome honestly. Directional decisions auto-log to the 🎯 Scorecard. "
               "This grades your **process**, not the stock.")

    ps = journal.process_score(DB)
    if ps.get("n"):
        st.markdown("#### 🧠 Process quality (your decisions so far)")
        pc = st.columns(5)
        pc[0].metric("Decisions", ps["n"])
        pc[1].metric("Wrote a thesis", f"{ps['thesis_pct']}%")
        pc[2].metric("Pre-set falsifiers", f"{ps['falsifiers_pct']}%")
        pc[3].metric("Set an exit plan", f"{ps['exitplan_pct']}%")
        pc[4].metric("Reviewed outcome", f"{ps['reviewed_pct']}%")
        st.caption("A **process** score (did you follow good practice?), NOT a win rate — better process "
                   "beats better luck. Your win/loss track record lives in the 🎯 Scorecard.")

    due = journal.due_for_review(DB)
    due_ids = {d["id"] for d in due}
    if due:
        st.warning("⏰ **" + str(len(due)) + " decision(s) due for review:** "
                   + ", ".join(f"{d['ticker']} ({d['review_date']})" for d in due)
                   + " — open them below and close the loop honestly.")

    _render_decision_form("➕ New decision")
    if st.session_state.get("journal_msg"):
        st.success(st.session_state.pop("journal_msg"))

    st.subheader("Open decisions")
    opens = journal.all_entries(DB, "OPEN")
    if not opens:
        st.write("None yet — log one above.")
    for e in opens:
        _render_open_entry(e, due_ids)

    reviewed = journal.all_entries(DB, "REVIEWED")
    if reviewed:
        st.subheader("Reviewed decisions")
        for e in reviewed:
            _render_reviewed_entry(e)