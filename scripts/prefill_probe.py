#!/usr/bin/env python3
"""Drive ninfer-serve prefill and report measured prefill throughput and energy.

Sends one chat completion per target prompt size with max_tokens=1 so the
measurement isolates prefill. Rates come from the structured request log, never
from the HTTP round trip, which also contains prepare and vision time.

With --power-limit-sweep the same probe is repeated across board power limits and
reports the throughput/energy trade-off. Energy per token is a function of the
power ceiling, not a fixed property of the engine, so the curve is the result and
a single number is not.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
import urllib.request

WORDS = None


def corpus():
    global WORDS
    if WORDS is None:
        # Deterministic pseudo-text with a realistic token/word ratio and no
        # long repeated span, so nothing upstream can shortcut the prefill.
        src = "/usr/share/dict/words"
        if os.path.exists(src):
            WORDS = [w.strip() for w in open(src) if w.strip().isalpha()]
        else:
            WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
                     "hotel", "india", "juliet", "kilo", "lima", "mike", "november",
                     "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform",
                     "victor", "whiskey", "xray", "yankee", "zulu"]
    return WORDS


def make_prompt(words, seed):
    rng = random.Random(seed)
    w = corpus()
    return " ".join(rng.choice(w) for _ in range(words))


def post(url, payload, timeout):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    return body, time.time() - t0


def tail_log(path, since):
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                # A torn trailing line is normal while the server is writing.
                continue
            out.append(rec)
    return out[since:]


def prefill_rate(rec):
    """Prefill tok/s from a request_done record.

    docs/performance.md defines this as computed prefill tokens over prefill
    seconds. Both come from the record; deriving it from the HTTP round trip
    would fold in prepare and vision time and overstate nothing but the wall.
    """
    if rec.get("event") != "request_done":
        return None
    result = rec.get("result") or {}
    timings = rec.get("timings_seconds") or {}
    tokens = result.get("computed_prefill_tokens") or 0
    seconds = timings.get("prefill") or 0.0
    if tokens <= 0 or seconds <= 0.0:
        return None
    return tokens / seconds


def server_ttft_ms(rec):
    timings = rec.get("timings_seconds") or {}
    prepare = timings.get("prepare") or 0.0
    vision = timings.get("vision") or 0.0
    prefill = timings.get("prefill") or 0.0
    return 1000.0 * (prepare + vision + prefill)


def reused_tokens(rec):
    """Prompt tokens this request did not have to prefill.

    Nonzero means the prompt was served from cached state instead of being
    prefilled, which invalidates the measurement rather than improving it.
    """
    if rec.get("event") != "request_done":
        return 0
    result = rec.get("result") or {}
    cache = rec.get("continuation_cache") or {}
    hit = result.get("prefix_cache_hit_tokens") or 0
    restored = cache.get("restored_tokens") or 0
    return hit + restored


def window_energy(records):
    """Sum board energy and tokens across the throughput records in a window."""
    joules = prefill_j = decode_j = idle_j = 0.0
    prefill_tokens = decode_tokens = 0
    residual = 0.0
    seen = False
    for rec in records:
        if rec.get("event") != "throughput":
            continue
        energy = rec.get("energy")
        if not energy:
            continue
        seen = True
        joules += energy["board_joules"]
        prefill_j += energy["prefill_joules"]
        decode_j += energy["decode_joules"]
        idle_j += energy["idle_joules"]
        residual += energy["residual_joules"]
        prefill_tokens += rec["tokens"]["computed_prefill"]
        decode_tokens += rec["tokens"]["committed_decode"]
    if not seen:
        return None
    return {"board_joules": joules, "prefill_joules": prefill_j, "decode_joules": decode_j,
            "idle_joules": idle_j, "residual_joules": residual,
            "prefill_tokens": prefill_tokens, "decode_tokens": decode_tokens}


def smi(query):
    out = subprocess.run(["nvidia-smi", f"--query-gpu={query}",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip().splitlines()[0].split(",")


def board_temperature():
    return float(smi("temperature.gpu")[0])


def current_power_limit():
    return float(smi("power.limit")[0])


def set_power_limit(watts):
    """Returns True when the board accepted the new ceiling.

    Changing the limit needs elevated privileges. Failing loudly matters: a
    sweep that silently ran every point at the same ceiling would produce a flat
    curve that looks like a finding.
    """
    result = subprocess.run(["nvidia-smi", "-pl", str(int(watts))],
                            capture_output=True, text=True)
    if result.returncode != 0:
        return False
    return abs(current_power_limit() - watts) < 1.0


def settle(target_c, timeout_s, poll_s=2.0):
    """Wait for the board to cool to a repeatable starting temperature.

    Energy efficiency drifts as the board heats. Without this, thermal state
    aliases onto whichever axis the sweep varies and the resulting curve
    describes the cooling system rather than the power ceiling.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if board_temperature() <= target_c:
            return True
        time.sleep(poll_s)
    return False


def run_point(args, targets, seen):
    """One pass over every target prompt size. Returns (rows, new_seen)."""
    rows = []
    for target in targets:
        words = int(target * args.words_per_token)
        prompt = make_prompt(words, args.seed * 100003 + target)
        payload = {"model": args.model, "max_tokens": args.max_tokens,
                   "temperature": 0.0,
                   "messages": [{"role": "user", "content": prompt}]}
        try:
            body, wall = post(args.url, payload, args.timeout)
        except Exception as exc:
            print(f"  target {target}: REQUEST FAILED: {exc}", file=sys.stderr)
            continue
        ptok = body.get("usage", {}).get("prompt_tokens", 0)
        # The reporter emits on its own interval, so the record covering the tail
        # of this request has not necessarily been written yet.
        time.sleep(args.settle_seconds)
        recs = tail_log(args.log, seen)
        seen += len(recs)
        rate = ttft = float("nan")
        reused = 0
        for rec in recs:
            r = prefill_rate(rec)
            if r is not None:
                rate = r
                ttft = server_ttft_ms(rec)
                reused += reused_tokens(rec)
        rows.append({"target": target, "prompt_tokens": ptok, "prefill_tok_s": rate,
                     "server_ttft_ms": ttft, "wall_s": wall, "reused_tokens": reused,
                     "energy": window_energy(recs)})
    return rows, seen


def summarize(rows):
    rates = [r["prefill_tok_s"] for r in rows if r["prefill_tok_s"] == r["prefill_tok_s"]]
    joules = tokens = 0.0
    reused = 0
    for row in rows:
        reused += row["reused_tokens"]
        energy = row["energy"]
        if energy:
            joules += energy["board_joules"]
            tokens += energy["prefill_tokens"] + energy["decode_tokens"]
    return {"prefill_tok_s": sum(rates) / len(rates) if rates else float("nan"),
            "board_joules": joules,
            "joules_per_token": joules / tokens if tokens else float("nan"),
            "reused_tokens": reused}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--log", default="/tmp/ninfer-reqlog.jsonl")
    ap.add_argument("--targets", default="8192,32768,102400")
    ap.add_argument("--words-per-token", type=float, default=0.75)
    ap.add_argument("--max-tokens", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="")
    ap.add_argument("--settle-seconds", type=float, default=6.0,
                    help="wait for the reporter interval to cover the request tail")
    ap.add_argument("--power-limit-sweep", default="",
                    help="comma-separated board power limits in watts, e.g. 250,320,400,480")
    ap.add_argument("--sweep-repeats", type=int, default=1)
    ap.add_argument("--cool-to", type=float, default=45.0,
                    help="board temperature each sweep point starts from")
    ap.add_argument("--cool-timeout", type=float, default=300.0)
    ap.add_argument("--json", default="", help="write the sweep report to this path")
    args = ap.parse_args()

    targets = [int(x) for x in args.targets.split(",")]
    seen = len(tail_log(args.log, 0))

    if not args.power_limit_sweep:
        print(f"{'label':10s} {'target':>8s} {'prompt_tok':>10s} {'prefill_tok_s':>14s} "
              f"{'ttft_ms':>10s} {'wall_s':>8s} {'J/tok':>8s} {'board_J':>9s}")
        rows, seen = run_point(args, targets, seen)
        for row in rows:
            energy = row["energy"]
            tokens = (energy["prefill_tokens"] + energy["decode_tokens"]) if energy else 0
            jpt = energy["board_joules"] / tokens if energy and tokens else float("nan")
            board_j = energy["board_joules"] if energy else float("nan")
            print(f"{args.label:10s} {row['target']:8d} {row['prompt_tokens']:10d} "
                  f"{row['prefill_tok_s']:14.1f} {row['server_ttft_ms']:10.1f} "
                  f"{row['wall_s']:8.2f} {jpt:8.3f} {board_j:9.1f}")
        if any(row["reused_tokens"] for row in rows):
            print("WARNING: prompt tokens were served from cache; this is not a prefill "
                  "measurement. Disable the continuation cache and prefix reuse.", file=sys.stderr)
        return

    limits = [float(x) for x in args.power_limit_sweep.split(",")]
    original = current_power_limit()
    print(f"board power limit {original:.0f} W; sweeping {limits}", file=sys.stderr)

    # Randomized point order across repeats. Sweeping monotonically would let the
    # board's rising temperature correlate with the power axis and be read as an
    # efficiency trend that is really a thermal one.
    schedule = []
    for repeat in range(args.sweep_repeats):
        order = list(limits)
        random.Random(args.seed + repeat).shuffle(order)
        schedule.extend((repeat, watts) for watts in order)

    points = []
    try:
        print(f"{'watts':>7s} {'rep':>4s} {'prefill_tok_s':>14s} {'J/tok':>8s} "
              f"{'board_J':>9s} {'temp_c':>7s}")
        for repeat, watts in schedule:
            if not set_power_limit(watts):
                print(f"FAILED to set the power limit to {watts:.0f} W; nvidia-smi -pl needs "
                      f"elevated privileges. Aborting the sweep rather than reporting every "
                      f"point at {current_power_limit():.0f} W.", file=sys.stderr)
                return 1
            if not settle(args.cool_to, args.cool_timeout):
                print(f"  warning: board did not reach {args.cool_to:.0f} C before this point",
                      file=sys.stderr)
            start_temp = board_temperature()
            rows, seen = run_point(args, targets, seen)
            summary = summarize(rows)
            summary.update({"power_limit_watts": watts, "repeat": repeat,
                            "start_temperature_c": start_temp})
            points.append(summary)
            print(f"{watts:7.0f} {repeat:4d} {summary['prefill_tok_s']:14.1f} "
                  f"{summary['joules_per_token']:8.3f} {summary['board_joules']:9.1f} "
                  f"{start_temp:7.1f}")
            sys.stdout.flush()
    finally:
        set_power_limit(original)

    if any(point["reused_tokens"] for point in points):
        print("WARNING: prompt tokens were served from cache; this is not a prefill "
              "measurement. Disable the continuation cache and prefix reuse.", file=sys.stderr)

    if args.json:
        report = {"artifact_type": "ninfer_power_limit_sweep", "schema_version": 1,
                  "model": args.model, "targets": targets,
                  "original_power_limit_watts": original,
                  "cool_to_c": args.cool_to, "points": points}
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
