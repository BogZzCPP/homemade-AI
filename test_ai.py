import sys
import time
import json
import traceback

try:
    import requests
except ImportError:
    print("ERROR: pip install requests")
    input("\nPress Enter to exit...")
    sys.exit()

# ─── CONFIG ───────────────────────────────────────────────
MODEL = "qwen3:4b-instruct"
BASE_URL = "http://localhost:11434"
MAX_HISTORY = 8
NUM_CTX = 4096
NUM_THREAD = 6
MAX_TOKENS = 512
TEMPERATURE = 0.3

RED = "\033[38;2;139;0;0m"
CYAN = "\033[38;2;0;200;255m"
DIM = "\033[2m"
RESET = "\033[0m"
GREEN = "\033[38;2;0;200;80m"
YELLOW = "\033[38;2;255;200;0m"

SYSTEM = """You are a nice but strict tutor/teacher, you are made for giving responses to questions, the questions can be either math, physics, engineering, fragrances, cars, just anything, you can answer anything.

Rules:
- For simple questions, give the simple answer. "2+2" -> "4". One sentence max.
- For complex questions, explain thoroughly: show every step, state the law/principle, explain WHY.
- You are NOT a cheerleader. Never praise unless genuinely correct.
- If the user is wrong, say "You're wrong because..." and explain why.
- If the user's reasoning has a gap, point it out.
- For code: explain WHY it works or is broken. Point out unasked bugs.
- For EE/physics: derive from first principles. Make them do skipped steps.
- Be direct. No filler. No "Great question!"
- If uncertain, say so. Don't guess.
- Match response length to question complexity. Simple in -> simple out.
- Keep responses under 200 words unless the user explicitly asks for a full derivation or detailed explanation."""

HELP = f"""
{CYAN}Commands:{RESET}
  {DIM}/clear{RESET}       - Reset conversation
  {DIM}/fast{RESET}        - Switch to 4b (fast)
  {DIM}/smart{RESET}       - Switch to 8b (smarter)
  {DIM}/model{RESET}       - Show current model + settings
  {DIM}/tokens <n>{RESET}  - Set max response length (e.g. /tokens 512)
  {DIM}/help{RESET}        - Show this
  {DIM}exit{RESET}         - Quit
"""


def trim(h, n):
    return h[-n:] if len(h) > n else h


def warmup(model):
    try:
        r = requests.post(
            f"{BASE_URL}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "think": False,
                "keep_alive": -1,
                "options": {"num_predict": 1},
            },
            timeout=60,
        )
        return r.status_code == 200
    except Exception:
        return False


def ask(question, history, model):
    msgs = [{"role": "system", "content": SYSTEM}] + history + [{"role": "user", "content": question}]
    payload = {
        "model": model,
        "messages": msgs,
        "stream": True,
        "think": False,
        "keep_alive": -1,
        "options": {
            "num_ctx": NUM_CTX,
            "num_thread": NUM_THREAD,
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS,
        },
    }

    r = requests.post(
        f"{BASE_URL}/api/chat",
        json=payload,
        stream=True,
        timeout=120,
    )
    r.raise_for_status()

    full = ""
    for line in r.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        if data.get("message", {}).get("content"):
            full += data["message"]["content"]
        if data.get("done"):
            break

    print(f"\n{RED}AI:{RESET} {full}\n")
    return full


# ─── STARTUP ──────────────────────────────────────────────
print(f"{CYAN}═════════ AI Homemade ═════════{RESET}\n")

try:
    requests.get(f"{BASE_URL}/api/tags", timeout=5)
except Exception:
    print(f"\033[31m✗ Ollama is not running.\033[0m")
    print("  Start it: open Start Menu -> type 'Ollama' -> click it.")
    print("  Then run this script again.\n")
    input("Press Enter to exit...")
    sys.exit()

print(f"{GREEN}✓ Ollama connected{RESET}")

if not warmup(MODEL):
    print(f"{YELLOW}⚠ Model '{MODEL}' may not be pulled. Trying anyway...{RESET}")

print(f"{GREEN}✓ Model ready: {MODEL}{RESET}\n")
print(f"{DIM}/help for commands | 'exit' to quit{RESET}\n")

# ─── MAIN LOOP ────────────────────────────────────────────
history = []

while True:
    try:
        q = input(f"{CYAN}You:{RESET} ").strip()
        if not q:
            continue

        if q.lower() in ("exit", "quit"):
            break

        if q.lower() == "/clear":
            history = []
            print(f"{DIM}  Cleared.{RESET}\n")
            continue

        if q.lower() == "/help":
            print(HELP)
            continue

        if q.lower() == "/model":
            print(f"{DIM}  Model: {MODEL}{RESET}")
            print(f"{DIM}  Max tokens: {MAX_TOKENS}{RESET}")
            print(f"{DIM}  Context: {NUM_CTX}{RESET}")
            print(f"{DIM}  Threads: {NUM_THREAD}{RESET}\n")
            continue

        if q.lower() == "/fast":
            if MODEL == "qwen3:4b-instruct":
                print(f"{DIM}  Already on 4b.{RESET}\n")
                continue
            MODEL = "qwen3:4b-instruct"
            history = []
            print(f"{DIM}  Switching to {MODEL}...{RESET}")
            if warmup(MODEL):
                print(f"{GREEN}  ✓ Ready.{RESET}\n")
            else:
                print(f"{YELLOW}  ⚠ Not pulled. Run: ollama pull qwen3:4b-instruct{RESET}\n")
            continue

        if q.lower() == "/smart":
            if MODEL == "qwen3:8b":
                print(f"{DIM}  Already on 8b.{RESET}\n")
                continue
            MODEL = "qwen3:8b"
            history = []
            print(f"{DIM}  Switching to {MODEL}...{RESET}")
            if warmup(MODEL):
                print(f"{GREEN}  ✓ Ready.{RESET}\n")
            else:
                print(f"{YELLOW}  ⚠ Not pulled. Run: ollama pull qwen3:8b{RESET}\n")
            continue

        if q.lower().startswith("/tokens"):
            parts = q.split()
            if len(parts) == 2:
                try:
                    MAX_TOKENS = int(parts[1])
                    print(f"{DIM}  Max tokens set to {MAX_TOKENS}.{RESET}\n")
                except ValueError:
                    print(f"{DIM}  Use a number. e.g. /tokens 512{RESET}\n")
            else:
                print(f"{DIM}  Usage: /tokens 512{RESET}\n")
            continue

        # ── Ask AI ──
        t0 = time.time()
        try:
            resp = ask(q, history, MODEL)
        except requests.exceptions.ConnectionError:
            print(f"\n\033[31m  Ollama disconnected. Restart it and try again.\033[0m\n")
            continue
        except Exception as e:
            print(f"\n\033[31m  {e}\033[0m\n")
            continue

        dt = time.time() - t0
        print(f"{DIM}  {dt:.1f}s{RESET}\n")

        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": resp})
        history = trim(history, MAX_HISTORY)

    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"\n\033[31m{e}\033[0m")
        traceback.print_exc()

print(f"\n{DIM}Goodbye.{RESET}")
input("\nPress Enter to exit...")