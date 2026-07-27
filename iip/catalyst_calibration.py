"""Catalyst calibration data -- historical magnitude/direction/reliability for the catalyst
categories catalyst_terminal.py's keyword scanner can flag in a live headline.

Every number here was pulled from real yfinance price history and cross-checked against news
coverage for the largest outliers (confounder-checking), not guessed or LLM-generated. See project
chat history for the full research session that produced this. Nothing here is exhaustive --
absence of a category means it hasn't been researched yet, not that it has no effect.

Reliability tiers (assigned by sample size and whether independent confounder-checking was done):
  "high"        -- n>=50 independent events (e.g. VIX spikes)
  "medium"      -- n=15-35 real calendar-scheduled events, outliers confounder-checked (FOMC/CPI/NFP)
  "low"         -- n<15 individually-verified events, too few to generalize a hit rate
  "single-case" -- n=1-3, real confirmed examples, illustrative only -- never treat as a base rate

THE ONE FINDING THAT SURVIVED EVERY CATEGORY TESTED: raw direction (cut/hike, beat/miss, hot/cold)
is a weak predictor on its own. What actually moves price is the outcome relative to what was
already expected/feared -- a "hawkish cut," a relief rally after a feared-worse regulatory remedy,
a delivery beat that still disappoints on margin commentary. A scoring system that only classifies
headline valence (bullish keyword vs bearish keyword) will get this wrong regularly. Wherever
possible, prefer signals that capture surprise-vs-expectation (e.g. Fed-funds-futures-implied-
probability shift, analyst-estimate vs actual) over a raw keyword's assumed polarity.
"""
from __future__ import annotations

MACRO_CATALYSTS = {
    "fomc_cut": {
        "mean_spy_day_pct": None, "reliability": "medium", "n": 6,
        "note": "Raw direction unreliable -- mean/std both swamp any signal. What matters is "
                "surprise-vs-expectation: the Dec 2024 'hawkish cut' (dot plot cut fewer 2025 "
                "reductions than priced) caused the S&P's worst Fed day since 2001 (~-3-4%), "
                "genuinely continuing through the full next hour, not fading. A cut in line with "
                "expectations barely moves anything. Do not score by direction alone.",
    },
    "fomc_hike": {
        "mean_spy_day_pct": None, "reliability": "medium", "n": 11,
        "note": "Same caveat as cuts, opposite direction confirmed: July 2022's +75bp hike caused "
                "a broad RALLY (MSFT +6.4%, GOOGL +8%, AMZN +4%) because Powell's guidance read "
                "dovish despite the large hike itself. Guidance tone > headline bps number.",
    },
    "fomc_hold": {
        "mean_spy_day_pct": None, "reliability": "medium", "n": 19,
        "note": "Near-zero average effect when genuinely expected -- holds are mostly non-events. "
                "Post-announcement continuation-vs-fade is close to a coin flip (6/15 continued "
                "into the close in the intraday sample) except when the hold carries a real "
                "surprise in the dot plot/tone.",
    },
    "fomc_vix_regime_note": "High-vs-low VIX median-split re-slice across all 36 FOMC meetings "
            "2022-2026 (not split by cut/hike/hold, n=18/18): SPY next-day mean_abs 1.13% high-VIX "
            "vs 0.82% low-VIX -- a modest ~1.4x amplification, weaker than NFP's.",
    "fomc_sector_leadership_note": "Same sector lead-lag test as CPI/NFP (n=36): NULL RESULT, no "
            "sector's day-of move predicts SPY's next-day move on FOMC days either. Same-day "
            "correlations are very high (0.64-0.97, mechanically -- everything reacts together to "
            "the same headline), but that's coincidence, not leadership.",
    "cpi": {
        "mean_spy_day_pct": 0.58, "reliability": "medium", "n": 23,
        "note": "n=24 raw, one excluded (Apr 10 2025 -- confounded by the concurrent Liberation "
                "Day tariff shock, not a clean CPI reaction). Gap-vs-continuation is close to a "
                "coin flip (43%). One confirmed-clean example: Jan 15 2025 (+1.82%, cooler-than- "
                "expected print + dovish Fed commentary).",
        "vix_regime_note": "High-vs-low VIX median-split re-slice (n=12/12): mean_abs 0.79% "
                "high-VIX vs 0.68% low-VIX -- essentially flat, std devs overlap heavily. No "
                "reliable VIX-regime amplification for CPI specifically.",
        "sector_leadership_note": "Tested corr(sector day-of move, SPY next-day move) across all "
                "10 SPDR sectors, n=24: no sector cleared |corr|>0.5, all near zero or mildly "
                "negative. NULL RESULT -- no sector leads SPY into the next day on CPI releases; "
                "same-day corr is high (0.3-0.9) simply because everything moves together.",
    },
    "nfp": {
        "mean_spy_day_pct": 1.0, "reliability": "medium", "n": 21,
        "note": "n=23 raw, 2 excluded/reclassified: Apr 4 2025 confounded by tariff shock "
                "(confirmed not jobs-driven despite landing on NFP day). IMPORTANT ASYMMETRY: both "
                "a too-weak report (Aug 2024, Sahm Rule recession fear, -1.86%) and a too-strong "
                "report (Jun 2026, rate-hike fear hit growth/semis, -2.58%) were bearish, for "
                "opposite reasons. Do not score 'strong jobs = bullish' -- it depends on the "
                "prevailing regime (recession fear vs. overheating fear).",
        "vix_regime_note": "High-vs-low VIX median-split re-slice (n=12/11): mean_abs 1.74% "
                "high-VIX vs 0.60% low-VIX -- a real ~2.9x amplification, the strongest of the "
                "three macro catalysts tested. Small-n caveat applies (11-12 per bucket); "
                "suggestive, not something to size positions on alone.",
        "sector_leadership_note": "Same null result as CPI (n=23): no sector's day-of move "
                "predicts SPY's next-day move. No cross-sector leadership signal on NFP days.",
    },
    "vix_spike": {
        "mean_spy_nextday_pct": 0.13, "reliability": "high", "n": 102,
        "note": "Well-powered NULL RESULT: a single-day VIX spike >=10% has no reliable next-day "
                "mean-reversion or continuation edge (53% positive, std 1.52 dwarfs the mean). The "
                "one huge positive outlier (Apr 8->9 2025, +10.5%) is the already-confirmed tariff- "
                "pause relief rally; excluding it the mean is ~zero. Score this as 'no standalone "
                "directional edge,' not as a contrarian buy signal.",
    },
    "oil_spike": {
        "mean_spy_day_pct": -0.05, "reliability": "low", "n": 27,
        "note": "The 'oil spike = bearish for SPY' intuition does NOT hold up (59% negative, "
                "barely above coin flip). Sample is really only 2-3 distinct geopolitical episodes "
                "(Mar 2022 Ukraine invasion, 2026 Middle East events), not 27 independent trials -- "
                "oil and SPY are usually both symptoms of the same event, not oil driving SPY.",
    },
    "oil_drop": {
        "mean_spy_day_pct": 0.01, "reliability": "low", "n": 34,
        "note": "Same clustering caveat as oil_spike. Essentially zero average relationship.",
    },
    "retail_sales": {
        "mean_spy_day_pct": None, "reliability": "medium", "n": 26,
        "note": "Census Bureau Advance Retail Sales report. SPY itself moves modestly (~0.65% "
                "mean_abs), but AMZN moves ~2.2x harder (1.41% mean_abs) on the exact same dates -- "
                "a real magnitude effect, not a directional bias, given AMZN's retail-heavy "
                "business. If you're tracking AMZN specifically, this release matters more than "
                "its broad-market importance alone suggests. Known gap: Mar-Aug 2025 dates missing.",
    },
}

# Per-company calibration. "earnings_mean_abs_pct" is each company's own earnings-day baseline
# (mean absolute move, all from real EPS-estimate-vs-actual data via yfinance get_earnings_dates,
# n=18 quarters per company, 2022-2026, unless noted). Use this as the reference scale for judging
# whether any other catalyst for that company is big or small relative to its normal earnings move.
COMPANY_CATALYSTS = {
    "NVDA": {
        "earnings_mean_abs_pct": 6.37, "earnings_n": 18,
        "hyperscaler_earnings_reaction_pct": 3.17,  # mean abs move on MSFT/GOOGL/AMZN/META earnings days, n=51 deduped dates
        "hyperscaler_note": "NVDA's own earnings day moves it ~2x harder than any hyperscaler's "
                            "earnings day. Cannot attribute reaction to one specific hyperscaler -- "
                            "MSFT/GOOGL/META/AMZN report within days of each other most quarters.",
        "china_restriction_severity_tiers": {
            "outright_ban_no_exceptions": {"cumulative_pct": -8.0, "reliability": "single-case",
                "note": "2022-10-07 sweeping export controls: -8.03% same day. 2022-08-31 initial "
                        "halt: -2.42%/-7.67% (2-day ~-10%)."},
            "tightening_existing_rules": {"cumulative_pct": -8.6, "reliability": "single-case",
                "note": "2023-10-17 strengthened rules: -4.68%/-3.96%."},
            "outright_ban_disclosed_after_hours": {"cumulative_pct": -6.87, "reliability": "single-case",
                "note": "2025-04-15 H20 ban + $5.5B charge: same-day figure misleading (news broke "
                        "after close), real reaction is next-day -6.87%, matches reported AH move."},
            "licensing_framework": {"cumulative_pct": -3.07, "reliability": "single-case",
                "note": "2025-01-13 AI Diffusion Rule (case-by-case exceptions possible): -1.97%/-1.10%."},
            "tariff": {"cumulative_pct": 0.69, "reliability": "single-case",
                "note": "2026-01-14 25% tariff on H200-class (sales still permitted): -1.44%/+2.13%, "
                        "net POSITIVE over 2 days -- smallest reaction of any restriction type."},
        },
        "china_restriction_note": "Clean severity ranking, n=6 individually-verified events: the "
            "less final the restriction (more some revenue path remains), the smaller the hit -- "
            "roughly 3x difference between harshest (outright ban) and mildest (tariff) reaction.",
    },
    "MSFT": {"earnings_mean_abs_pct": 4.61, "earnings_n": 18,
             "note": "18-quarter beat/miss split computed but miss n=1 -- not enough to trust a "
                     "beat-vs-miss differential for this name specifically.",
             "cloud_peer_reaction_mean_abs_pct": 1.96, "cloud_peer_reaction_n": 9,
             "cloud_peer_note": "MSFT's reaction to GOOGL earnings (cloud-competitor read), CLEANED "
                 "of a real contamination: MSFT and GOOGL report on the exact same day 13 of 19 "
                 "times in this window, so the naive test (3.28% mean_abs) was mostly MSFT reacting "
                 "to ITSELF, not GOOGL. Restricted to the 9 genuinely non-overlapping dates: 1.96% "
                 "mean_abs vs 4.61% on its own earnings -- a ~2.3x gap, back in line with the "
                 "'own event dominates ~2x' pattern found for NVDA and TSLA. Methodological lesson: "
                 "always check for report-date overlap before trusting a cross-company reaction test."},
    "AAPL": {"earnings_mean_abs_pct": 2.97, "earnings_n": 18,
             "note": "Lowest earnings volatility of the big tech names (2.97%). Same miss-n=1 "
                     "caveat as MSFT. Taiwan/TSMC risk comparative: AAPL showed only -0.93% on "
                     "the Aug 2022 Pelosi visit (the one clean novel-escalation event) -- see "
                     "TAIWAN_RISK below."},
    "AMZN": {"earnings_mean_abs_pct": 6.58, "earnings_n": 18,
             "note": "Reasonable beat(n=15)/miss(n=3) split: beat +1.67%, miss -4.04%. Surprise% "
                     "field from yfinance is UNRELIABLE for AMZN specifically in years with EPS "
                     "near zero (produced nonsensical values like -190%/+657%) -- use price "
                     "reaction directly, not the surprise percentage, for this ticker.",
             "retail_sales_mean_abs_pct": 1.41, "retail_sales_n": 26,
             "retail_sales_note": "AMZN moves ~2.2x more than SPY on Census Bureau Advance Retail "
                 "Sales release days (1.41% vs SPY's 0.65% mean_abs, same 26 real dates) -- a "
                 "genuine magnitude effect (AMZN just moves harder whichever way the day goes), "
                 "not a directional bias -- confirms real macro sensitivity to consumer spending "
                 "data given the retail-heavy business, consistent across the sample not driven "
                 "by one outlier. Known gap: March-August 2025 data-month releases weren't found."},
    "META": {"earnings_mean_abs_pct": 11.46, "earnings_n": 18,
             "note": "Highest earnings volatility of any name tested (std=14.16%, mean_abs=11.46%). Beat +3.59% "
                     "(n=14) vs miss -8.22% (n=4) -- the cleanest beat/miss differential found. "
                     "DATA QUALITY FLAG: yfinance's Surprise(%) for the 2023-02-01 report shows "
                     "-21.53% (implying a big miss) but Meta actually BEAT EPS ($2.72 vs $2.56 "
                     "est.) -- the +23% rally that day was the 'Year of Efficiency' guidance/buyback "
                     "pledge, unrelated to the (mis-recorded) EPS surprise number. Sanity-check any "
                     "large yfinance surprise% outlier against real news before trusting it.",
             "ad_peer_reaction_mean_abs_pct": 1.54, "ad_peer_reaction_n": 33,
             "ad_peer_note": "META's reaction to SNAP/PINS earnings (ad-market health peers), "
                 "deduped by date: only 1.54% mean_abs vs 11.46% on its own earnings -- a ~7-8x "
                 "gap, same 'own event dominates' pattern as NVDA-vs-hyperscalers. One contaminated "
                 "outlier excluded: PINS' 2022-04-27 report coincided exactly with META's OWN "
                 "earnings date -- the raw +17.59% that day was META reacting to itself, not PINS."},
    "GOOGL": {"earnings_mean_abs_pct": 5.26, "earnings_n": 18,
              "antitrust_events": {
                  "2024-08-05_liability_ruling": -4.45,
                  "2025-09-03_remedies_ruling": 9.14,
              },
              "antitrust_note": "Same company, same case, opposite reactions: found guilty (bad "
                  "news in the abstract) -4.45%; remedies turn out milder than feared (Chrome/"
                  "Android breakup avoided) +9.14%. Outcome-relative-to-feared-outcome, not "
                  "headline valence, drove both."},
    "AVGO": {"earnings_mean_abs_pct": 7.39, "earnings_n": 18,
             "custom_silicon_events": {
                 "2025-10-13_openai_deal": 9.88,
                 "2026-04-07_google_tpu_anthropic_deal": 16.0,  # cumulative over 3 days
             },
             "custom_silicon_note": "Hyperscaler customer-contract announcements (9.88%, ~16% "
                 "cumulative) are larger than AVGO's own average earnings move (7.39%) -- unlike "
                 "NVDA/TSLA where the company's own event dominates, AVGO reacts at least as hard, "
                 "and sometimes harder, to customer news than its own earnings."},
    "TSLA": {"earnings_mean_abs_pct": 8.62, "earnings_n": 18,
             "delivery_report_mean_abs_pct": 4.58, "delivery_report_n": 14,
             "delivery_note": "Same pattern as NVDA: own earnings (~8.62%) matter ~2x more than "
                 "the adjacent delivery-report metric (~4.58%), though deliveries alone are still "
                 "a real, tradeable-size catalyst. Concrete raw-number-lies example: 2026-07-02 "
                 "delivery BEAT still caused an 8% intraday reversal/drop -- gap +0.64% on the "
                 "beat, then -8.07% intraday, matching the reported headline exactly."},
    "BRK-B": {"earnings_mean_abs_pct": 0.90, "earnings_n": 18,
              "fomc_reaction_by_regime": {"CUT": -0.83, "HIKE": -0.09, "HOLD": -0.04},
              "fomc_note": "Does NOT confirm the textbook 'rallies on rate hikes via insurance "
                  "float yield' story -- sign is flat-to-backwards. Lowest volatility of every "
                  "stock tested in this whole project (std 0.75-1.49 vs 2.3-4.2 for mega-cap tech "
                  "on the same dates) -- BRK-B just doesn't react much to single-day catalysts."},
    "JPM": {"earnings_mean_abs_pct": 2.93, "earnings_n": 19,
            "earnings_note": "Second-calmest earnings reaction of every name tested this session "
                "(only BRK-B is lower) -- consistent with banks generally having more predictable, "
                "less explosive earnings than tech/growth names.",
            "fomc_reaction_pilot": "See original pilot: only 2 of 6 tested FOMC dates were free of "
            "confounding same-day company-specific news (earnings season, expense guidance). "
            "Cleanest single-name illustration in the whole project of why confounder-checking "
            "matters before trusting a proximity-to-catalyst price move."},
}

# Maps catalyst_terminal.py's IMPORTANCE_KEYWORDS to a researched MACRO_CATALYSTS entry.
# Deliberately incomplete: political/geopolitical keywords (tariff, war, sanctions, trump,
# invasion, executive order) are NOT mapped -- their real market impact is too context-dependent
# (which tariff, which war, which company) to reduce to one number, and forcing a mapping here
# would fabricate precision that doesn't exist. Extend this one verified category at a time, the
# same way every entry above was built -- never add a mapping without the research behind it.
KEYWORD_TO_CATEGORY = {
    "rate hike": "fomc_hike",
    "rate cut": "fomc_cut",
    "cpi": "cpi",
    "inflation": "cpi",
    "jobs report": "nfp",
    "nonfarm payroll": "nfp",
    "unemployment": "nfp",
    "retail sales": "retail_sales",
}


TICKER_KEYWORDS = {
    "nvidia": "NVDA", "nvda": "NVDA", "tesla": "TSLA", "tsla": "TSLA",
    "broadcom": "AVGO", "avgo": "AVGO", "google": "GOOGL", "alphabet": "GOOGL", "googl": "GOOGL",
    "meta": "META", "microsoft": "MSFT", "msft": "MSFT", "apple": "AAPL", "aapl": "AAPL",
    "amazon": "AMZN", "amzn": "AMZN", "berkshire": "BRK-B", "jpmorgan": "JPM", "jpm": "JPM",
}
CATEGORY_KEYWORDS = {
    "export restriction": "china_restriction", "export control": "china_restriction",
    "chip ban": "china_restriction", "china export": "china_restriction",
    "antitrust": "antitrust", "monopoly ruling": "antitrust",
    "custom chip": "custom_silicon", "custom silicon": "custom_silicon", "tpu deal": "custom_silicon",
    "vehicle deliveries": "deliveries", "delivery numbers": "deliveries",
}
TAIWAN_KEYWORDS = {"taiwan"}


def _lookup_company(keyword_hits: list[str]) -> dict:
    """Cross-references ticker mentions with category-phrase mentions in the same headline to
    find the most specific researched calibration available. Falls back to the ticker's own
    earnings-day baseline (as a scale reference, clearly labeled as such) when a company is named
    but no specific researched category matches -- still useful context, just not catalyst-
    specific."""
    tickers = {TICKER_KEYWORDS[kw] for kw in keyword_hits if kw in TICKER_KEYWORDS}
    categories = {CATEGORY_KEYWORDS[kw] for kw in keyword_hits if kw in CATEGORY_KEYWORDS}
    out = {}
    if any(kw in TAIWAN_KEYWORDS for kw in keyword_hits):
        out["taiwan_risk"] = TAIWAN_RISK
    for tk in tickers:
        entry = COMPANY_CATALYSTS.get(tk)
        if not entry:
            continue
        matched = False
        if "china_restriction" in categories and "china_restriction_severity_tiers" in entry:
            out[f"{tk}_china_restriction"] = {"note": entry.get("china_restriction_note"),
                                              "tiers": entry["china_restriction_severity_tiers"]}
            matched = True
        if "antitrust" in categories and "antitrust_events" in entry:
            out[f"{tk}_antitrust"] = {"note": entry.get("antitrust_note"),
                                      "events": entry["antitrust_events"]}
            matched = True
        if "custom_silicon" in categories and "custom_silicon_events" in entry:
            out[f"{tk}_custom_silicon"] = {"note": entry.get("custom_silicon_note"),
                                           "events": entry["custom_silicon_events"]}
            matched = True
        if "deliveries" in categories and "delivery_report_mean_abs_pct" in entry:
            out[f"{tk}_deliveries"] = {"mean_abs_pct": entry["delivery_report_mean_abs_pct"],
                                       "note": entry.get("delivery_note")}
            matched = True
        if not matched and entry.get("earnings_mean_abs_pct") is not None:
            out[f"{tk}_reference_scale"] = {
                "earnings_mean_abs_pct": entry["earnings_mean_abs_pct"],
                "note": "No specific researched catalyst category matched this headline -- shown "
                        "as this ticker's own earnings-day baseline for scale reference only.",
            }
    return out


def summarize(calibration: dict) -> str:
    """One compact, human-readable line per matched calibration entry, for a table cell -- the
    full researched note stays available in the raw dict for a detail view, this is just the
    headline number(s)."""
    if not calibration:
        return ""
    parts = []
    for key, entry in calibration.items():
        if "tiers" in entry:
            vals = [v.get("cumulative_pct") for v in entry["tiers"].values()
                    if v.get("cumulative_pct") is not None]
            parts.append(f"{key}: {min(vals):+.1f}% to {max(vals):+.1f}% (severity-dependent)" if vals else key)
        elif "events" in entry:
            vals = list(entry["events"].values())
            parts.append(f"{key}: historically {min(vals):+.1f}% to {max(vals):+.1f}%" if vals else key)
        elif "mean_abs_pct" in entry:
            parts.append(f"{key}: avg {entry['mean_abs_pct']:.1f}% move")
        elif "earnings_mean_abs_pct" in entry:
            label = key.replace("_reference_scale", "")
            parts.append(f"{label} earnings baseline: {entry['earnings_mean_abs_pct']:.1f}% avg (context only)")
        elif key == "taiwan_risk":
            parts.append("Taiwan risk: historically muted for US-listed names")
        elif "reliability" in entry:
            parts.append(f"{entry.get('category', key)}: n={entry.get('n')}, {entry['reliability']} reliability")
    return " · ".join(parts)


def lookup(keyword_hits: list[str]) -> dict:
    """Best-available historical calibration for a live headline's matched importance keywords --
    macro categories (Fed/CPI/NFP) plus company-specific ones (China restrictions, antitrust,
    custom-silicon deals, deliveries, Taiwan risk). {} if nothing matches -- callers should treat
    an empty result as 'no calibration data,' never as 'this catalyst has no effect.'"""
    out = {}
    for kw in keyword_hits:
        cat = KEYWORD_TO_CATEGORY.get(kw)
        if cat and cat in MACRO_CATALYSTS:
            out[kw] = {"category": cat, **MACRO_CATALYSTS[cat]}
    out.update(_lookup_company(keyword_hits))
    return out


TAIWAN_RISK = {
    "reliability": "single-case + 2 null observations",
    "pelosi_visit_2022_08_02": {"NVDA": 0.46, "AAPL": -0.93, "TSM_ADR": -0.30},
    "note": "The one clean, novel-escalation event (Pelosi's 2022 Taiwan visit) shows only small "
            "moves in US-listed names, sharply contrasting the LOCAL Taiwan market reaction "
            "reported at the time (TWSE -2.1%, TSMC Taiwan-listed -3.1%). Subsequent 'routine' "
            "Taiwan military drill headlines (Oct 2024 'Joint Sword-2024B', Dec 2025 'Justice "
            "Mission 2025') show ZERO measurable reaction -- Taiwan's market closed HIGHER both "
            "times despite the drills. Conclusion: Taiwan geopolitical risk, even for a genuine "
            "first-time escalation, does not reliably transmit into NVDA/AAPL's US-listed price "
            "action at meaningful size, and the market has clearly desensitized to recurring "
            "headline versions of this risk. Score this category LOW weight unless the headline "
            "is genuinely unprecedented (actual conflict, not another drill).",
}
