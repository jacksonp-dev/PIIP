"""Cost governor — budget is architecture, not polish (LOCKED SPEC).

Hard caps: daily spend, per-run spend, calls-per-run. Persists today's spend so restarts don't
reset the cap. Every LLM call must pass through can_spend()/record().
"""
from __future__ import annotations

import json
import os
from datetime import date

STATE = "cost_state.json"

# rough output-heavy price per 1M tokens (input+output blended); interpretation calls are small
PRICES_PER_MTOK = {
    "claude-haiku-4-5-20251001": 1.5,
    "claude-sonnet-5": 6.0,
    "claude-opus-4-8": 30.0,
}


def est_call_cost(model: str, in_tok: int = 1500, out_tok: int = 500) -> float:
    blended = PRICES_PER_MTOK.get(model, 3.0)
    return (in_tok + out_tok) / 1_000_000 * blended


class Budget:
    def __init__(self, path: str = STATE, daily_cap: float = 3.0,
                 per_run_cap: float = 0.15, max_calls_per_run: int = 4):
        self.path = path
        self.daily_cap = daily_cap
        self.per_run_cap = per_run_cap
        self.max_calls_per_run = max_calls_per_run
        self.run_spent = 0.0
        self.run_calls = 0

    def _today(self) -> dict:
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                s = json.load(f)
            if s.get("date") == date.today().isoformat():
                return s
        return {"date": date.today().isoformat(), "spent": 0.0}

    def spent_today(self) -> float:
        return self._today()["spent"]

    def start_run(self) -> None:
        self.run_spent = 0.0
        self.run_calls = 0

    def can_spend(self, est: float) -> tuple[bool, str]:
        if self.run_calls >= self.max_calls_per_run:
            return False, f"per-run call cap reached ({self.max_calls_per_run})"
        if self.run_spent + est > self.per_run_cap:
            return False, f"per-run ${self.per_run_cap:.2f} cap would be exceeded"
        if self.spent_today() + est > self.daily_cap:
            return False, f"daily ${self.daily_cap:.2f} cap would be exceeded"
        return True, "ok"

    def record(self, cost: float) -> None:
        self.run_spent += cost
        self.run_calls += 1
        s = self._today()
        s["spent"] = round(s["spent"] + cost, 6)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(s, f)