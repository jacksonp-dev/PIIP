# PIIP — Personal Investment Intelligence Platform

## Welcome

PIIP is a free, research-and-paper-trading tool for US stocks and options. It's built for
**learning by doing, without risking real money** — a deterministic engine computes real numbers
(technicals, greeks, implied volatility, expected move, historical odds) from free market data,
an optional AI layer helps interpret them, and a paper-trading account lets you practice actual
decisions and track how they play out.

**This app runs entirely on YOUR computer.** It's not a hosted website — there's no server
anywhere else involved. Your data, your account, and any API keys you enter stay on this machine;
nothing is uploaded anywhere by this platform itself. Every clone runs its own fully independent
copy.

Some real data is genuinely paywalled — true real-time quotes, institutional order flow, the real
NYSE advance/decline line — and this tool doesn't have access to it. Where a gap like that exists,
it's disclosed right on the page, not silently filled in or estimated without a label.

**This is for research and paper trading only.** It does not execute real trades, and nothing here
is financial advice.

## Why I built this

I'm a developer with 8+ years of experience. I wanted to learn options trading, but the amount I
needed to understand felt like climbing a mountain — so I built PIIP as a research and
paper-trading tool to actually learn by doing, without risking real money. I'm sharing it in case
it helps others in the same boat.

I hope you find it useful and educational — please leave feedback or report bugs (there's a
built-in 🐛 Feedback page for that) so I can keep improving it for everyone.

## Quick start (Windows, no Python install needed)

1. Clone this repo, or click GitHub's **Code → Download ZIP** and extract it.
2. Double-click **`launch_piip.bat`** in the repo root.
3. First launch installs everything it needs automatically — needs internet, takes a few minutes,
   only happens once. A portable Python runtime is already included in `python-embed/`, so nothing
   is installed system-wide and nothing else needs to be on your machine first.
4. Your browser opens to the app. The first time, it'll ask — optionally, every field is skippable
   — for your own Anthropic/Finnhub API keys, with links to go get them. Skip anything you don't
   have; you can always add it later by editing the `.env` file this step creates.

That's it — no `pip install`, no terminal commands, no separate Python setup.

## What each key is for

None of these are required to use PIIP — skip any of them and the app still works, just with that
one feature turned off.

| Key | What it powers | Cost |
|---|---|---|
| `ANTHROPIC_API_KEY` | The optional AI-interpretation layer on top of the free deterministic engine. Skip it and PIIP still works fully on free data — the AI layer just stays off. | Pay-as-you-go, a few cents per research run (dry-run/free by default until you turn it on). |
| `FINNHUB_API_KEY` | The 0DTE Intelligence page's Catalyst Terminal news feed. Skip it and that one section shows a setup hint instead of headlines; everything else works normally. | Free tier. |
| `RESEARCH_CONTACT_EMAIL` | Sent as a courtesy contact in requests to Wikipedia and SEC EDGAR — both sites' own fair-access policies ask automated tools to include a real way to reach whoever's running them. Not shared with anyone else, not used for anything but that. | Free (it's just your email, not a paid key). |

## What's inside

- **Home** — your account at a glance: equity, today's P&L, open positions.
- **Watchlist / Screener** — track tickers you care about, or scan the full S&P 500 + major
  ETFs by price and today's move.
- **Feed** — a ranked list of names with an unusually large expected move and why.
- **Reddit Momentum** — detects fast-rising Reddit discussion across 6 subreddits (not a stock
  picker — a "worth a look" signal).
- **0DTE Intelligence** — a same-day market-condition dashboard (bias, breadth, options/dealer
  positioning, Market DNA day-type read) for SPY/QQQ/IWM/DIA — research context, not a trading
  signal.
- **Research / Deep Research / Ticker Page** — per-ticker technicals, options chain, greeks,
  fundamentals, clinical trials (biotech), and SEC filing catalysts.
- **Paper trading** — a real, persistent $1,000 paper options account. Every trade is saved and
  survives a restart.
- **Journal** — log your thesis, what would prove you wrong, and your exit plan; review the
  outcome honestly afterward.
- **Scorecard** — every prediction this tool makes gets graded against reality, SPY, and a coin
  flip. If it can't beat a dumb baseline, that's shown plainly, not hidden.
- **Glossary** — every term on the platform explained in plain English with examples.
- **Feedback** — report a bug or share an idea; opens a pre-filled GitHub issue in your browser.

## This is a fully self-contained local instance

- Every clone runs **entirely on your own machine** with **your own API keys** — there is no
  shared backend, server, or account. Nothing you do here touches anyone else's copy of this app.
- Local data (`iip.db` — your paper-trading account, watchlist, journal, prediction history) is
  **gitignored** and created fresh, empty, on first run. It is never committed and never shared
  between clones.
- The only paid API is Anthropic (Claude), and only if you turn on the AI layer — see **Cost**
  below. Every other data source (Yahoo Finance, Wikipedia, FRED, SEC EDGAR, ClinicalTrials.gov,
  ApeWisdom, Finnhub's free tier) is free and keyless except Finnhub, which needs its own free
  API key — again, yours alone, not shared.
- **Never commit your `.env` file** — it holds your real keys. `.env.example` is the safe template
  every clone starts from.

## Manual setup (any OS, or if you'd rather manage your own Python env)

```bash
git clone <this repo's URL>
cd PIIP
python -m venv .venv
.venv\Scripts\activate      # Windows — use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. Same first-run onboarding screen either way — `.env` isn't
something you need to hand-create; the app writes it for you the first time it launches.

## CLI (optional, same engine as the app)

```bash
python -m iip.cli research AAPL                  # deterministic only  (FREE)
python -m iip.cli research AAPL --llm            # + AI layer, DRY-RUN (FREE, tests wiring)
python -m iip.cli research AAPL --llm --live-llm # + AI layer for REAL (spends, cost-governed)
python -m iip.cli score                          # resolve due predictions + print scorecard
python -m iip.cli report                         # print scorecard
```

## Cost

The AI layer is **dry-run (free) by default**. A real run = a handful of Claude calls (~$0.02–0.03
on Haiku), governed by a daily cap, per-run cap, and calls-per-run cap so it can't run away on you.
The deterministic engine — the actual computation behind every number on the platform — is always
free.

## Methodology — the honesty rules baked in

- **The deterministic engine computes; the AI only interprets.** Every numeric metric (technicals,
  greeks, IV, expected move, historical odds) comes from code, never from the AI — the AI narrates
  what the numbers already say, and is scored against the deterministic baseline rather than
  trusted by default.
- **Every prediction is logged and graded** — against reality, against SPY, and against a coin
  flip. Today's honest baseline is close to a coin flip on direction; the Scorecard says so plainly
  instead of flattering the tool.
- **Data gaps are disclosed, not papered over.** Where PIIP can't get real data for free (order
  flow, true real-time quotes, the real NYSE breadth line), it labels the substitute as an estimate
  or proxy right on the page rather than presenting a guess as fact.

See [BUILD_PLAN.md](BUILD_PLAN.md) for the deeper design rationale behind these decisions.

## Feedback & license

Found a bug or have an idea? Use the in-app **🐛 Feedback** page, or open an issue directly on
GitHub. See [LICENSE](LICENSE) for usage terms.
