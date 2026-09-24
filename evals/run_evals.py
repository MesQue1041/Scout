import argparse
import json
import sys
import time
from pathlib import Path
import re

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agent import Agent  

TOOL_SETS = {
    "docs": {"search_documents"},
    "web": {"search_web"},
    "both": {"search_documents", "search_web"},
    "none": set(),
}


def run_case(case: dict, use_guardrail: bool) -> dict:
    agent = Agent(use_guardrail=use_guardrail)  # fresh memory per case
    turns = case.get("turns") or [case["question"]]
    answer, tools, blocked, errors = "", set(), False, []
    start = time.time()
    for i, turn in enumerate(turns):
        last = i == len(turns) - 1
        for ev in agent.chat(turn):
            if not last:
                continue  # earlier turns just build up memory
            if ev["type"] == "token":
                answer += ev["text"]
            elif ev["type"] == "tool_call":
                tools.add(ev["name"])
            elif ev["type"] == "blocked":
                blocked = True
            elif ev["type"] == "error":
                errors.append(ev["message"])

    expect = case.get("expect_tools", "any")
    if expect == "any":
        routing_ok = not blocked
    elif expect == "blocked":
        routing_ok = blocked
    else:
        routing_ok = not blocked and tools == TOOL_SETS[expect]

    low = re.sub(r"\s+", " ", answer.lower())
    must = case.get("must_include", [])
    any_of = case.get("any_of", [])
    content_ok = all(m.lower() in low for m in must) and (not any_of or any(a.lower() in low for a in any_of))

    return {
        "id": case["id"], "expect_tools": expect, "tools_used": sorted(tools), "blocked": blocked,
        "routing_ok": routing_ok, "content_ok": content_ok, "passed": routing_ok and content_ok and not errors,
        "seconds": round(time.time() - start, 1), "errors": errors, "answer": answer,
    }
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="run a single case id")
    ap.add_argument("--no-guardrail", action="store_true")
    args = ap.parse_args()

    cases = yaml.safe_load((ROOT / "evals" / "cases.yaml").read_text())
    if args.only:
        cases = [c for c in cases if c["id"] == args.only]

    results = []
    for case in cases:
        if args.no_guardrail and case.get("expect_tools") == "blocked":
            continue
        r = run_case(case, use_guardrail=not args.no_guardrail)
        results.append(r)
        mark = "PASS" if r["passed"] else "FAIL"
        why = []
        if not r["routing_ok"]:
            why.append(f"routing: expected {r['expect_tools']}, got {'blocked' if r['blocked'] else r['tools_used'] or 'none'}")
        if not r["content_ok"]:
            why.append("content: expected keywords missing")
        if r["errors"]:
            why.append(f"error: {r['errors'][0][:80]}")
        print(f"{mark}  {r['id']:<20} {r['seconds']:>5}s  {'; '.join(why)}")

    n = len(results)
    if not n:
        print("No cases run.")
        return
    pct = lambda k: 100 * sum(r[k] for r in results) / n
    print(f"\nRouting accuracy: {pct('routing_ok'):.0f}%   Content: {pct('content_ok'):.0f}%   "
          f"Overall pass rate: {pct('passed'):.0f}%  ({sum(r['passed'] for r in results)}/{n})")

    out = ROOT / "evals" / f"results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Details (incl. full answers): {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
