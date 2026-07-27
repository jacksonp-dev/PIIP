"""PIIP command line.

  python -m iip.cli research AAPL                 # deterministic only (free)
  python -m iip.cli research AAPL --llm           # + LLM layer, DRY-RUN (still free, tests wiring)
  python -m iip.cli research AAPL --llm --live-llm# + LLM layer for REAL (spends, cost-governed)
  python -m iip.cli score                         # resolve due predictions + print scorecard
  python -m iip.cli report                        # print scorecard only
"""
from __future__ import annotations

import argparse

from . import backtest, orchestrator, scorer
from . import predictions as pred


def _fmt(entry):
    fc, opt = entry["fc"], entry["opt"]
    return (f"  [{entry['source']:<13} H{entry['H']:>2}d] {fc['stock_direction']:<8} "
            f"move~{fc.get('stock_expected_move_pct')}% pUp={fc.get('stock_prob_up')} "
            f"conf={fc.get('confidence')}  opt={opt.get('option_type')} {opt.get('option_strike')} "
            f"@ ${opt.get('option_entry_premium')} (id={entry['id']})")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="iip")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("research", help="run research on a ticker")
    r.add_argument("ticker")
    r.add_argument("--llm", action="store_true", help="enable LLM layer (dry-run unless --live-llm)")
    r.add_argument("--live-llm", action="store_true", help="actually spend on Claude calls")
    r.add_argument("--horizons", default="7,30,90")
    r.add_argument("--db", default=pred.DB_PATH)

    b = sub.add_parser("backtest", help="point-in-time deterministic backtest (real results now)")
    b.add_argument("--tickers", default="AAPL,MSFT,NVDA,AMZN,GOOGL,META,JPM,XOM")
    b.add_argument("--db", default="iip_backtest.db")

    s = sub.add_parser("score", help="resolve due predictions then print scorecard")
    s.add_argument("--db", default=pred.DB_PATH)
    p = sub.add_parser("report", help="print scorecard")
    p.add_argument("--db", default=pred.DB_PATH)

    a = ap.parse_args(argv)
    if a.cmd == "research":
        horizons = tuple(int(x) for x in a.horizons.split(","))
        res = orchestrator.research(a.ticker.upper(), horizons=horizons, use_llm=a.llm,
                                    db=a.db, dry_run=(not a.live_llm))
        print(f"\n{res['ticker']} @ ${res['spot']:.2f}")
        for e in res["deterministic"]:
            print(_fmt(e))
        if a.llm:
            mode = "LIVE" if a.live_llm else "DRY-RUN (free)"
            print(f"  --- LLM layer ({mode}) ---")
            for e in res["llm"]:
                print(_fmt(e))
            print(f"  LLM cost this run: ${res['cost_usd']:.4f}")
        print("\nLogged as pre-registered experiments. Run `score` after horizons elapse.")
    elif a.cmd == "backtest":
        tks = [t.strip().upper() for t in a.tickers.split(",")]
        print(f"Point-in-time deterministic backtest over {len(tks)} tickers (this takes a moment)...")
        db, n = backtest.run_backtest(tks, db=a.db)
        print(f"Backtested {n} deterministic predictions (stock thesis only; options are forward-only).\n")
        print(scorer.format_report(db))
    elif a.cmd == "score":
        n = scorer.resolve_due(a.db)
        print(f"Resolved {n} newly-due prediction(s).\n")
        print(scorer.format_report(a.db))
    elif a.cmd == "report":
        print(scorer.format_report(a.db))


if __name__ == "__main__":
    main()