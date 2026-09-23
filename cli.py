import argparse

from agent import Agent

DIM, CYAN, YELLOW, RESET = "\033[2m", "\033[36m", "\033[33m", "\033[0m"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-guardrail", action="store_true")
    args = ap.parse_args()

    agent = Agent(use_guardrail=not args.no_guardrail)
    print("Research assistant. Ask about your files or the web. /reset clears memory, /quit exits.\n")
    while True:
        try:
            q = input(f"{CYAN}you>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q == "/quit":
            break
        if q == "/reset":
            agent.reset()
            print(f"{DIM}(memory cleared){RESET}\n")
            continue

        print(f"{CYAN}bot>{RESET} ", end="", flush=True)
        for ev in agent.chat(q):
            if ev["type"] == "token":
                print(ev["text"], end="", flush=True)
            elif ev["type"] == "tool_call":
                print(f"\n{DIM}  -> {ev['name']}({ev['args']}){RESET}", flush=True)
            elif ev["type"] == "tool_result":
                print(f"{DIM}  <- {len(ev['preview'])}+ chars back{RESET}\n     ", end="", flush=True)
            elif ev["type"] == "blocked":
                print(f"{YELLOW}[blocked: {ev['reason']}]{RESET} ", end="")
            elif ev["type"] == "error":
                print(f"{YELLOW}[error] {ev['message']}{RESET}")
        print("\n")


if __name__ == "__main__":
    main()
