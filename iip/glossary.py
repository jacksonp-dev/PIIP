"""Central plain-English glossary for every field / column shown in the UI.

One source of truth so hover-tooltips stay consistent. Keys must match the exact column header or
label used in app.py. Keep each description short and jargon-free — the goal is 'a beginner instantly
gets what this number means', not a textbook definition. `help_for(name)` is a safe case-tolerant lookup.
"""
from __future__ import annotations

GLOSSARY: dict[str, str] = {
    # ── options chain ──
    "Strike": "The price you lock in to buy (call) or sell (put) the stock if you exercise. Your bet is the stock crosses it.",
    "Premium": "Price PER SHARE of the option. One contract = 100 shares, so a $0.20 premium costs $20.",
    "Breakeven": "The stock price where you start making money at expiration (strike ± premium). Past this you profit.",
    "% to B/E": "How far the stock must move (in %) to reach your breakeven. Bigger number = a harder bet.",
    "Cost": "Total dollars to buy one contract = premium × 100.",
    "Odds": "How reachable the breakeven is vs the expected move: 🟢 within reach, 🟡 a stretch, 🔴 moonshot (needs a huge move).",
    "P(profit)": "Rough estimated chance the stock is past your breakeven at expiration. NOT a sure thing — direction is ~a coin flip.",

    # ── feed card ──
    "expected move": "How far the options market expects the stock to swing (UP or DOWN) over the period, e.g. ±20%. Bigger = more uncertainty priced in.",
    "Expected move": "How far the options market expects the stock to swing (UP or DOWN) over the period. Bigger = more uncertainty priced in.",
    "Expected move (~30d)": "How far the options market expects the stock to swing (up OR down) over ~30 days.",
    "IV": "Implied volatility — how jumpy options expect the stock to be. Higher IV = pricier options and a bigger expected move.",
    "Implied volatility": "How jumpy options expect the stock to be. Higher = pricier options.",
    "ATM implied volatility": "How jumpy options expect the stock to be, from the at-the-money option. Higher = pricier options.",
    "Trend": "Whether the stock is above or below its recent average prices (up-trend vs down-trend).",
    "Momentum": "Feed: whether recent price action is pushing up or down, and whether it's overbought (stretched) "
               "or oversold. On the 0DTE page: the estimated probability the CURRENT intraday move keeps going, "
               "from its velocity and acceleration.",
    "Model lean": "The tool's mechanical guess at direction. Honestly ~a coin flip and only one input — NOT a prediction.",
    "Alignment": "How strongly the mechanical signals agree with each other. This is NOT the probability of profit.",
    "Volatility": "How pricey/jumpy the options are (IV vs the stock's normal moves). A COST signal, not a direction call.",
    "Catalyst": "A known dated reason the stock could move — earnings, a filing, a trial readout. In the Feed it's "
               "a 🟢/🟡/⚪ flag; in the Watchlist it shows the next known date itself.",

    # ── paper account (positions + totals) ──
    "Contract": "The specific option: ticker, call/put, strike, and expiry date.",
    "Opened": "The date you bought this position.",
    "Expiry": "The date the option expires. After this it's worth only its in-the-money value, or nothing.",
    "Qty": "Number of contracts you hold (each controls 100 shares).",
    "Entry": "The premium PER SHARE you paid when you opened the position.",
    "Now": "The current premium per share (mid-price mark).",
    "Value": "What the position is worth right now = current premium × 100 × contracts.",
    "P&L $": "Profit or loss in dollars since you bought (unrealized — not locked in until you sell).",
    "P&L %": "Profit or loss in percent since you bought.",
    "Exit": "The premium per share you sold at when you closed the position.",
    "Realized P&L": "Locked-in profit or loss on a trade you've already closed.",
    "Cash": "Uninvested dollars sitting in your paper account.",
    "Equity": "Total account value = cash + current value of your open positions.",
    "Total P&L": "Your overall profit/loss vs the $1,000 starting balance.",
    "Open": "How many option positions you currently hold.",
    "Total cost basis": "Total dollars you've paid into all your open positions.",
    "Current value": "What all your open positions are worth right now.",
    "Unrealized P&L": "Paper profit/loss on open positions — not real money until you actually sell.",

    # ── deep research ──
    "Field": "The name of the data point.",
    "What it means": "Plain-English explanation of what the value tells you.",
    "Conf": "How much we trust this data: 🟢 High, 🟡 Medium, 🟠 Low, ⚪ Unknown (no free source).",
    "Source": "Where the data came from.",
    "Research Completeness": "How much of the research we could back with real data. NOT a buy score — it shows where YOUR next research helps most.",

    # ── clinical trials ──
    "Phase": "Trial stage: Phase 1 = safety, Phase 2 = does it work?, Phase 3 = large confirmatory (closest to approval).",
    "Status": "Whether the trial is recruiting, active, completed, etc.",
    "Enroll": "Number of patients in the trial.",
    "Primary completion": "When the trial finishes collecting its main data. The readout/announcement usually follows — NOT an FDA decision date.",
    "NCT": "The trial's unique ID on ClinicalTrials.gov (look it up for full details).",

    # ── SEC filings / catalysts ──
    "Form 4": "A filing insiders (execs/directors) must make within 2 days of buying or selling their own company's stock. A cluster of buys/sells is worth reading.",
    "8-K": "A filing companies must make within days of a material event (management change, new agreement, going-concern warning, etc.) — public but rarely makes headlines right away.",

    # ── the Greeks ──
    "Delta": "How much the option's price moves per $1 the stock moves. A 0.50 call gains ~$0.50 when the stock gains $1. Calls: 0 to 1. Puts: -1 to 0.",
    "Gamma": "How fast Delta itself changes as the stock moves. High gamma = Delta shifts quickly — your risk can change fast, especially near expiry.",
    "Theta": "Money the option loses per day just from time passing (time decay), all else equal. Always working against option BUYERS.",
    "Vega": "How much the option's price moves per 1-point change in implied volatility. High vega = more exposed to IV swings (e.g. earnings IV crush).",
    "Rho": "How much the option's price moves per 1% change in interest rates. Usually the smallest, least-important Greek for short-dated options.",

    # ── Reddit momentum ──
    "Momentum Score": "0-100, built only from real mention counts/rank/cross-subreddit spread — NOT a buy signal, just 'how fast is discussion rising.'",
    "Mention change": "% change in mentions vs 24h ago, straight from the data source — a big number means discussion is accelerating fast.",
    "Cross-subreddit": "How many of the 6 tracked subreddits are discussing this ticker right now. Showing up in several at once is a stronger signal than one noisy sub.",
    "New appearance": "This ticker wasn't logged in our own history before today — first time it's crossed the radar.",
    "Catalyst-linked": "A known dated event (earnings, filing, trial) lines up with this spike — there's a real reason, not just noise.",
    "Unexplained spike": "High Reddit momentum with no known catalyst found. Could be a real early signal or pure hype — research before acting, don't assume either way.",

    # ── scorecard / backtest ──
    "Hit rate": "Share of past predictions that turned out correct.",
    "Brier": "Forecast accuracy score: 0 is perfect, 0.25 is a coin flip — lower is better.",
    "Alpha vs SPY": "Extra return vs simply holding the S&P 500. Positive = beat the market.",
    "Resolved": "How many logged predictions have played out and been graded.",

    # ── 0DTE Intelligence page ──
    "Trade Confidence": "The page's single roll-up number combining Market Bias, Entry Quality, and the options/"
                        "dealer context. Backtested ~57% next-day accuracy at its BEST — statistically the same "
                        "as SPY's own base rate. An agreement heuristic, not a proven edge.",
    "Confluence": "How many of the page's independent signals (bias, breadth, sector, momentum, reversal…) "
                 "actually agree with each other right now. More agreement ≠ higher odds of profit.",
    "Entry Quality": "A 0-100 roll-up of bias + momentum + options liquidity + dealer positioning — flags a "
                     "clean vs messy setup right now, not a probability of profit.",
    "Exit Quality": "Checked once you flag 'I'm in a trade' — watches whether the market's bias has flipped "
                    "against your direction so you notice a real change. Never an auto-exit signal.",
    "Market Bias": "A -100..+100 mechanical lean (VWAP position, EMA alignment, breadth agreement, VIX trend, "
                   "RSI, price vs SMA200) shown as a CALLS/PUTS lean. Still mechanical, still not a prediction.",
    "Breadth (proxy)": "Approximates market-wide participation from a basket (mega-caps + sector ETFs + the "
                       "other 3 indices) — NOT the real NYSE Advance/Decline or TICK. No free source exists for "
                       "those; this is a clearly labeled stand-in.",
    "Sector Health": "% of the 10 tracked sector ETFs confirming the same direction as the index. Broad "
                     "confirmation across sectors is a stronger read than one sector moving alone.",
    "Mega Cap Health": "A weighted composite of the 10 largest index-influencing stocks, using approximate "
                       "(not exact) index weights — shows whether the megacaps that move the index most are "
                       "actually participating.",
    "VWAP": "Volume-Weighted Average Price — the session's running 'fair value,' weighted by how much volume "
           "traded at each level. Above it favors buyers; below it favors sellers.",
    "Relative Volume": "Today's volume vs the 20-day average at this point in the session. Reads artificially "
                       "low before market close — partial-day volume, not time-of-day adjusted.",
    "Previous Day": "Yesterday's high/low/close — common reference levels traders watch for support/resistance "
                    "and gap context.",
    "Opening Range (15m)": "The high/low of the first 15 minutes of the session. A breakout above/below it, "
                           "especially on strong relative volume, is a classic (not guaranteed) continuation signal.",
    "SPY Health": "A 0-100 composite of trend, momentum, buying pressure, structure, and relative volume for SPY.",
    "QQQ Health": "A 0-100 composite of trend, momentum, buying pressure, structure, and relative volume for QQQ.",
    "IWM Health": "A 0-100 composite of trend, momentum, buying pressure, structure, and relative volume for IWM.",
    "DIA Health": "A 0-100 composite of trend, momentum, buying pressure, structure, and relative volume for DIA.",
    "Reversal Risk": "Estimates whether the CURRENT move looks like it's strengthening or running out of steam "
                     "(price/RSI divergence, VWAP overextension) — does NOT predict tops or call exact turns.",
    "Options Health": "Liquidity/execution quality of the options chain right now — tight spreads and real open "
                      "interest/volume mean you can actually get in and out at a fair price.",
    "Dealer Positioning": "An ESTIMATE of options dealers' net gamma exposure (GEX) from open interest. Positive/"
                          "'long gamma' regimes tend to dampen moves; negative/'short gamma' can amplify them. "
                          "Not live/intraday OI, and the dealer-sign convention is a simplification.",
    "Net GEX": "Estimated net gamma exposure in dollars — the scale behind the Dealer Positioning read. An "
              "estimate from end-of-prior-day open interest, not a live number.",
    "10-Year Treasury Yield": "The benchmark U.S. bond yield. Often moves SPY more than people expect, "
                             "especially around CPI/Fed days — rising yields are typically a headwind for stocks.",
    "Dollar Index (DXY)": "Tracks the US dollar against a basket of currencies. A strong dollar is historically "
                          "a headwind for large-cap/multinational earnings (and SPY).",
    "WTI Crude Oil": "West Texas Intermediate front-month crude price — context for Energy-sector-driven moves.",
    "Put/Call OI Ratio": "Total put open interest ÷ call open interest across the chain. Above ~1.1 is put-heavy "
                        "(more downside hedging/bets), below ~0.9 is call-heavy — a noisier sentiment read than "
                        "Dealer Positioning, not a strong signal alone.",
    "Equal-Weight vs Cap-Weight": "Compares RSP (equal-weight S&P 500) to SPY (cap-weight) on the day. RSP "
                                 "lagging SPY means the move is concentrated in a few mega-caps, not broad "
                                 "participation.",
    "Market DNA": "A same-session classification of today's day-type (Range Bound, Trend Day, Gap & Go, etc.) "
                 "built from gap behavior, range vs normal, and how one-sided price has stayed vs VWAP.",
    "Session Length": "How many minutes of today's session have printed so far — Market DNA needs at least 30 "
                      "before it will commit to a day-type label.",
    "Overnight Gap": "% difference between today's open and yesterday's close. Whether this gap HOLDS or gets "
                    "reversed is a key input to the Market DNA label.",
    "Gap Held": "Whether price has stayed on the gap side of yesterday's close all session (Yes) or traded back "
               "through it at some point (No) — the difference between a Gap & Go day and an Opening Reversal.",
    "Range vs Normal (ATR)": "Today's high-low range divided by the 14-day average true range (ATR). Above ~1.3x "
                            "with little net progress reads as choppy/wide two-way action.",
    "Net Move vs Range": "How much of today's total range was 'net' directional progress vs just back-and-forth "
                        "chop. Feeds both the Trend Day and Slow Grind Market DNA labels.",
    "VWAP Consistency": "% of today's bars that closed on the same side of the running VWAP as the current "
                       "price. A trend day stays on one side almost all session; chop crosses back and forth.",
    "Day Change": "Today's % change from the prior close to the current price.",

    # ── Home KPI tiles ──
    "Today": "Your account's profit/loss vs where it stood at yesterday's close. Shows once a prior day has "
            "been logged.",
    "Cash available": "Uninvested dollars sitting in your paper account, ready to open new positions.",
    "Open positions": "How many option positions you currently hold, and their current total value.",

    # ── Screener / Watchlist columns ──
    "Symbol": "The stock's ticker — the short code it trades under.",
    "Company": "The company's full name.",
    "Sector": "The broad industry group the company belongs to — useful context for spotting sector-wide moves "
             "vs single-stock news.",
    "Chart": "A quick sparkline of recent price action — shape only, use the Ticker Page for the real chart.",
    "Price": "The stock's latest traded price.",
    "1D %": "How much the stock has moved today.",
    "Volume": "Shares traded today. Unusually high volume alongside a price move adds conviction to that move.",
    "Reddit": "Total Reddit mentions today across the tracked subreddits, and how many different subreddits are "
             "talking about it.",
}


# ── Rich, example-driven definitions for the dedicated 📖 Glossary tab ──
# Ordered list of (category, [(term, long_description_with_example), ...]).
GLOSSARY_GROUPS: list = [
    ("Options — the basics", [
        ("Call", "A bet a stock goes UP — the right to BUY 100 shares at the strike price. "
                 "Example: you buy a GRAB $4 call for $0.20 (= $20). If GRAB rises above $4.20 by expiry "
                 "you make money; if it stays under $4, the call expires worthless and you lose the $20."),
        ("Put", "A bet a stock goes DOWN — the right to SELL 100 shares at the strike. "
                "Example: a NIO $4.50 put for $0.19. You profit if NIO falls below $4.31; if NIO stays "
                "above $4.50, you lose the premium."),
        ("Premium", "The price of the option, quoted PER SHARE. One contract controls 100 shares, so a "
                    "$0.20 premium costs $20. You pay the premium to open; it's the most you can lose on a buy."),
        ("Strike", "The price the option is 'aimed at.' A $4 call pays off above $4; a $4 put pays off below $4. "
                   "Strikes further from today's price are cheaper but need a bigger move to pay."),
        ("Expiration (expiry)", "The option's deadline. After it, the option settles and is gone. "
                                "'8/7' = expires August 7. Less time = cheaper but riskier (less room for your move); "
                                "more time = pricier but more room."),
        ("Contract", "One option = 100 shares of exposure. 'Qty 3' means 3 contracts = 300 shares of exposure."),
        ("Exercise", "Turning an option into actual shares. Almost never worth doing early — SELLING the option "
                     "usually gives you more, because you keep its leftover time value. Exercising a $4 call also "
                     "means paying $400 for 100 shares."),
    ]),
    ("Reading the options chain", [
        ("Breakeven", "The stock price where you start making money at expiry. Call = strike + premium; "
                      "put = strike − premium. A $4 call bought for $0.20 breaks even at $4.20."),
        ("% to breakeven", "How far the stock must move to reach breakeven. '+9%' means the stock has to rise 9%. "
                           "The bigger this is, the harder the bet."),
        ("Cost", "Total dollars for one contract = premium × 100. A $0.20 premium = $20."),
        ("Odds (🟢 🟡 🔴)", "How reachable breakeven is vs the expected move. 🟢 within ~1 expected move (realistic), "
                            "🟡 a stretch (up to 2×), 🔴 moonshot — needs a huge move, usually a near-certain loss."),
        ("P(profit)", "A rough estimate of the chance the stock is past your breakeven at expiry. 25% ≈ a 1-in-4 shot. "
                      "It does NOT tell you which direction — that part is ~a coin flip."),
        ("⭐ vs green ★", "⭐ = the highest-odds contract in the whole list (usually deep in-the-money and expensive). "
                          "green ★ = the best odds among the strikes you can actually AFFORD on your budget — your realistic 'best way in.'"),
        ("Expected move", "How far the options market expects the stock to swing — UP or DOWN — over the period. "
                          "'±20%' means about a 20% move is priced in, either direction. Bigger = more uncertainty priced in."),
        ("Implied volatility (IV)", "How jumpy options expect the stock to be. High IV = expensive options + a bigger "
                                    "expected move. After big events (like earnings) IV often 'crushes' (drops sharply), "
                                    "which can lose you money even if you guessed direction right."),
        ("⚠️ no bid / stale", "'no bid' = nobody's currently buying, so you may be unable to sell/exit. "
                              "'stale' = the shown price isn't a live quote (market closed or barely-traded strike) — "
                              "the real cost to buy is usually higher. Avoid these for real trades."),
    ]),
    ("The Greeks", [
        ("Delta", "How much the option's price moves per $1 the stock moves — its 'speed.' A 0.50 call gains "
                  "roughly $0.50 for every $1 the stock rises. Calls run 0 to 1; puts run -1 to 0. Also a rough "
                  "'chance of finishing in-the-money' estimate (a 0.30 delta ≈ ~30% odds), though that's an "
                  "approximation, not a real probability."),
        ("Gamma", "How fast Delta ITSELF changes as the stock moves — the option's 'acceleration.' High gamma "
                  "(common near expiry / near the strike) means Delta can swing quickly, so your position's risk "
                  "can change fast even if the stock only moves a little."),
        ("Theta", "How much value the option loses PER DAY just from time passing, all else equal (time decay). "
                  "Theta is always working against option BUYERS — the clock is ticking whether the stock moves "
                  "or not. Decay accelerates the closer you get to expiration."),
        ("Vega", "How much the option's price moves per 1-point move in implied volatility (IV). High vega = more "
                 "exposed to IV swings — this is why options often lose value right after earnings even if you "
                 "guessed direction correctly (IV 'crushes' once the event has passed)."),
        ("Rho", "How much the option's price moves per 1% change in interest rates. Usually the smallest and "
                "least important Greek day-to-day, especially for short-dated options — it matters more for "
                "long-dated (LEAPS) contracts."),
    ]),
    ("The Feed & its signals", [
        ("Trend", "Whether the stock is above or below its recent average prices — a simple up-trend vs down-trend read."),
        ("Momentum", "Whether recent price action is pushing up or down, and whether it's overbought (stretched, "
                     "can pull back) or oversold (beaten down, can bounce)."),
        ("Model lean", "The tool's mechanical direction guess. It's honestly ~a coin flip and only one input — NOT a prediction."),
        ("Alignment", "How strongly the mechanical signals AGREE with each other. 'Strong' just means they're consistent — "
                      "it is NOT a higher chance of winning. Direction is still a coin flip."),
        ("Catalyst tag", "Whether there's a dated reason to move. 🟢 = a known event (like earnings) is coming. "
                         "🟡 = options are pricey but there's no known reason (likely just noise, unless YOU know one). "
                         "⚪ = nothing scheduled."),
        ("Cheapest sensible entry", "The least money for a realistic (within-expected-move, non-lottery) contract on a card — "
                                    "e.g. '$5 put @ 2026-08-07.' Helps you see what your budget can actually trade."),
    ]),
    ("Your paper account", [
        ("Entry / Now", "Entry = the premium per share you paid to open. Now = the current premium (mid-price). "
                        "If Now > Entry you're up on the mark."),
        ("Cost / Value", "Cost = what you paid in (entry × 100 × contracts). Value = what it's worth now "
                         "(current premium × 100 × contracts)."),
        ("Unrealized P&L", "Paper profit/loss on positions you STILL HOLD. It's not real money until you sell — and "
                           "you'd sell at the bid, a bit below the shown mark. Example: mark says +$5, but selling at "
                           "the bid might only give +$3."),
        ("Realized P&L", "Locked-in profit/loss from trades you've already CLOSED. This is real."),
        ("Cash / Equity", "Cash = uninvested dollars. Equity = total account value = cash + current value of open positions."),
        ("Total P&L", "Your overall gain/loss vs the $1,000 starting balance (includes both open and closed)."),
    ]),
    ("Deep Research & financials", [
        ("Market cap", "The price of the WHOLE company = share price × total shares. A $2B market cap means buying "
                       "every share would cost about $2 billion. Big-cap = huge/established; small-cap = smaller/riskier."),
        ("Revenue", "Total sales over the last 12 months. Growing revenue is good; shrinking is a yellow flag."),
        ("EPS (earnings per share)", "Profit divided by number of shares. Negative EPS = the company is losing money."),
        ("Profit margin", "How many cents of profit the company keeps per $1 of sales. Negative = unprofitable."),
        ("Free cash flow", "Cash left after running and investing in the business. Positive = self-funding; "
                           "negative = burning cash (may need to raise money)."),
        ("Cash runway", "For a money-losing company, how long its cash lasts before it must raise more — which "
                        "dilutes existing shareholders. '~1.6 years' is a warning clock (common for early biotech)."),
        ("P/E ratio", "Price ÷ yearly profit per share. A P/E of 20 means you pay $20 for each $1 the company earns a "
                      "year. ~20 is typical for the market; lower can be 'cheaper.' No P/E = not profitable yet."),
        ("Price/Sales", "Price ÷ yearly sales — the go-to valuation gauge when a company has no profit yet. Lower is cheaper."),
        ("Analyst target", "Where Wall Street analysts think the stock goes in ~12 months. Useful context, but targets "
                           "skew optimistic and the high/low range shows how much they disagree."),
        ("Short % of float", "How many shares are bet AGAINST the stock. ~5% is normal; 40% is heavily shorted — lots of "
                             "downside bets, but also 'squeeze fuel' if the stock rises and shorts scramble to buy back."),
        ("Institutional ownership", "How much is held by big funds (vs everyday investors). High = 'smart money' is involved."),
        ("Confidence (🟢🟡🟠⚪)", "How much we trust a data point: 🟢 High (verified number), 🟡 Medium, 🟠 Low, "
                                 "⚪ Unknown (no free source — we won't fake it)."),
    ]),
    ("Biotech & clinical trials", [
        ("Trial phase", "Drug-testing stages. Phase 1 = is it safe? Phase 2 = does it work? Phase 3 = large final proof "
                        "before the FDA. Phase 3 results move biotech stocks the most."),
        ("Primary completion", "When a trial finishes collecting its main data. The public RESULTS usually come weeks to "
                               "months later — that's the real catalyst. It is NOT an FDA approval date."),
        ("NCT number", "A trial's unique ID on ClinicalTrials.gov. Search it there for full details."),
    ]),
    ("Track record (Scorecard)", [
        ("Hit rate", "Share of past predictions that came true. Above 50% is better than a coin flip — but check the "
                     "confidence interval; small samples lie."),
        ("Brier score", "Forecast accuracy: 0 is perfect, 0.25 is a coin flip, lower is better."),
        ("Alpha vs SPY", "Extra return vs simply holding the S&P 500. Positive = you beat just buying the index."),
        ("Research Completeness", "How much of a research dossier we could back with real data. It is NOT a buy score — "
                                  "it just shows where YOUR next research adds the most."),
    ]),
    ("0DTE Intelligence page", [
        ("Trade Confidence", "The page's single roll-up number (0-100) blending Market Bias, Entry Quality, and the "
                             "options/dealer context into one 'how much do the signals agree' read. Backtested "
                             "Jul 2023-2026 (546 days, same scoring code): even its highest bucket hit ~57% next-day "
                             "directional accuracy — statistically indistinguishable from SPY's own ~57.3% base rate. "
                             "Treat it as an internal-agreement heuristic, not a proven edge."),
        ("Confluence", "How many of the page's independent signals (bias, breadth, sector, mega-cap, momentum, "
                       "reversal…) actually point the same direction right now, e.g. '6/8 signals agree.' More "
                       "agreement means the picture is internally consistent — it does NOT raise your odds of "
                       "profit on its own."),
        ("Entry Quality", "A 0-100 roll-up of bias, momentum, options liquidity, and dealer positioning meant to "
                          "flag whether right now looks like a clean, tradeable setup or a messy/illiquid one — "
                          "not a probability of profit."),
        ("Exit Quality", "Unlocked once you flip 'I'm in a trade.' Watches whether the market's bias has flipped "
                         "against your chosen direction so you notice a real change of conditions — it never "
                         "auto-exits you, that call stays yours."),
        ("Market Bias", "A mechanical -100..+100 lean built from VWAP position, EMA20/50/200 alignment, breadth "
                        "agreement, the VIX trend, RSI, and price vs the 200-day average — shown as a CALLS/PUTS "
                        "lean with a confidence %. Still mechanical, still roughly a coin flip on direction."),
        ("Breadth (proxy)", "Approximates whether the WHOLE market is participating using a basket of mega-caps + "
                            "sector ETFs + the other 3 major indices — there is no free source for the real NYSE "
                            "Advance/Decline line or TICK, so this is a clearly-labeled stand-in, not the genuine "
                            "market-wide reading."),
        ("Sector Health", "% of the 10 tracked sector ETFs (XLK, XLF, XLE, etc.) moving the same direction as the "
                          "index. Broad sector participation is a stronger read than a move driven by one sector alone."),
        ("Mega Cap Health", "A weighted composite of the 10 largest index-moving stocks (AAPL, MSFT, NVDA…), using "
                            "approximate index weights — shows whether the names that actually move the index are "
                            "participating in today's move."),
        ("VWAP", "Volume-Weighted Average Price — the session's running 'fair value,' weighted by how much volume "
                "traded at each price. Above VWAP favors buyers; below favors sellers. Institutional desks watch "
                "this level closely."),
        ("Relative Volume", "Today's volume vs the 20-day average AT THIS POINT in the session. Reads artificially "
                            "low before market close since it's partial-day volume — don't read a low number as "
                            "'quiet' if the session just started."),
        ("Previous Day levels", "Yesterday's high/low/close — the reference levels traders commonly watch for "
                                "support, resistance, and gap context."),
        ("Opening Range (15m)", "The high/low of the session's first 15 minutes. A breakout above or below it — "
                                "especially on strong relative volume — is a classic continuation signal, though "
                                "not a guaranteed one."),
        ("{Ticker} Health", "A 0-100 composite of trend, momentum, buying pressure, structure, and relative volume "
                            "for the selected index (SPY/QQQ/IWM/DIA)."),
        ("Momentum (continuation)", "On this page, the estimated probability the CURRENT move keeps going, from "
                                    "the intraday velocity (recent rate of change) and acceleration (is velocity "
                                    "itself speeding up or slowing down)."),
        ("Reversal Risk", "Estimates whether the current move looks like it's strengthening or running out of "
                          "steam, from price/RSI divergence and how far price has stretched from VWAP. Does NOT "
                          "predict exact tops or bottoms."),
        ("Options Health", "Liquidity/execution quality of the options chain right now — tight bid/ask spreads and "
                           "real open interest/volume mean you can actually get filled at a fair price, not just a "
                           "quoted one."),
        ("Dealer Positioning / Net GEX", "An ESTIMATE of options dealers' net gamma exposure, built from open "
                                        "interest (not live/intraday data). Positive ('long gamma') regimes tend to "
                                        "dampen price swings; negative ('short gamma') can amplify them. Gamma wall "
                                        "strikes shown are context — support/resistance framing, never a guaranteed level."),
        ("Macro Context (10Y yield / DXY / WTI / Put-Call OI / Equal-vs-Cap-Weight)",
         "Slower-moving, day-regime context, not a timing signal: 10-Year Treasury Yield (rising yields are "
         "typically a stock headwind), Dollar Index/DXY (a strong dollar pressures large-cap earnings), WTI Crude "
         "(context for Energy-sector moves), Put/Call OI Ratio (call-heavy vs put-heavy positioning — noisier than "
         "Dealer Positioning), and Equal-Weight (RSP) vs Cap-Weight (SPY) — RSP lagging SPY means the move is "
         "concentrated in a few mega-caps, not broad participation."),
    ]),
    ("Market DNA (day-type read)", [
        ("Market DNA", "A same-session classification of today's day-type — Range Bound, Trend Day, Gap & Go, etc. "
                       "— built purely from today's own gap behavior, range vs normal, and how one-sided price has "
                       "stayed vs the running VWAP. Logged every refresh so the read can be reviewed after the fact."),
        ("Range Bound", "The default label: no clear gap, trend, or chop signature was detected — price is "
                        "chopping within a fairly normal range."),
        ("Gap & Go", "Opened with a significant gap from yesterday's close that has HELD (never traded back "
                    "through it) and kept extending in the same direction."),
        ("Opening Reversal", "Opened with a significant gap that has since been filled/reversed — the early move "
                             "faded rather than continuing."),
        ("Trend Day", "Price has stayed on one side of VWAP for most of the session and is sitting near its high "
                      "(or low) — a real net-directional day, not just chop."),
        ("High Volatility Chop", "A wide range relative to the 14-day ATR but with little NET progress — lots of "
                                 "two-way movement that ends up going nowhere."),
        ("Slow Grind Higher / Lower", "Steady single-direction drift without the range expansion of a full Trend "
                                      "Day — a quieter, grindier version of the same idea."),
        ("Insufficient Evidence", "Shown for the first ~30 minutes of the session (or if daily/intraday data isn't "
                                  "available yet) — deliberately refuses to guess a day-type before there's enough "
                                  "of the session to actually read."),
        ("Session Length", "How many minutes of today's session have printed so far — Market DNA needs at least "
                           "30 before it will commit to a label."),
        ("Overnight Gap", "% difference between today's open and yesterday's close. Whether this gap HOLDS or gets "
                          "reversed is a key input to the day-type label."),
        ("Gap Held", "Whether price has stayed on the gap side of yesterday's close all session (Yes) or traded "
                     "back through it at some point (No) — the difference between Gap & Go and Opening Reversal."),
        ("Range vs Normal (ATR)", "Today's high-low range divided by the 14-day average true range (ATR). Above "
                                  "~1.3x with little net progress reads as choppy/wide two-way action."),
        ("Net Move vs Range", "How much of today's total range was 'net' directional progress (|last − open| ÷ "
                              "range) vs just back-and-forth chop. High values feed both Trend Day and Slow Grind."),
        ("VWAP Consistency", "% of today's bars that closed on the same side of the running VWAP as the current "
                             "price. A trend day stays on one side almost all session; chop crosses back and forth."),
        ("Day Change", "Today's % change from the prior close to the current price — the same number used in the "
                       "Market Bias and Mega Cap Health reads."),
    ]),
]


def help_for(name) -> str:
    """Case-tolerant lookup; '' if the term isn't in the glossary (so callers can pass anything)."""
    if not name:
        return ""
    if name in GLOSSARY:
        return GLOSSARY[name]
    low = str(name).lower()
    for k, v in GLOSSARY.items():
        if k.lower() == low:
            return v
    return ""
