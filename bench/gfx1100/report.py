"""Step 5's table: plain vs drafted decode on gfx1100 from bench_concurrent receipts, beside the gfx1151 figures.

  report.py STEP5_DIR
"""
import glob
import json
import os
import statistics
import sys

# PR #3665's recipe (docs/recipes/qwen3.8-27b.md, commit 8/8): gfx1151, T=0, plain -> drafted tok/s
GFX1151 = {("code", 1): (14.4, 34.8), ("chat", 1): (14.8, 28.0), ("code", 2): (27.2, 49.2), ("chat", 2): (27.5, 38.1),
           ("code", 4): (46.2, 54.5), ("chat", 4): (45.8, 44.7)}


def cells(path):
    return json.load(open(path))["cells"]


def main(d):
    runs = {}  # (model, arm) -> {(prompt, streams, temp): [aggregate tok/s per round], ttft}
    for f in sorted(glob.glob(os.path.join(d, "bench-*.json"))):
        tag = os.path.basename(f)[6:-5]
        parts = tag.split("-")
        if parts[-1] == "contended":
            parts[1] += " (CPU contended)"
        if parts[0] == "llama":
            model, arm = parts[1], "llama.cpp"
        else:
            model, arm = parts[0], parts[1]
        for c in cells(f):
            k = (c["prompt"], c["streams"], c["temperature"])
            r = runs.setdefault((model, arm), {}).setdefault(k, {"tps": [], "ttft": []})
            r["tps"].append(c["aggregate_tps"])
            r["ttft"].append(c["request"]["ttft_s"])
    acc = {}
    for f in glob.glob(os.path.join(d, "drafts-*-*-*.json")):
        _, model, arm = os.path.basename(f)[:-5].split("-")[:3]
        if arm == "plain":
            continue
        x = json.load(open(f))
        a = acc.setdefault((model, arm), [0, 0, 0])
        a[0] += x["rounds"]; a[1] += x["accepted"]; a[2] += x["drafted"]
    med = lambda v: statistics.median(v) if v else float("nan")
    fmt = lambda x: "–" if x != x else f"{x:.1f}"
    for model in sorted({m for m, _ in runs}):
        print(f"\n### {model}: aggregate decode tok/s (median of alternating rounds), TTFT s")
        print("| prompt | streams | T | plain | trees | chains | trees speed-up | llama.cpp | TTFT plain / trees | contended plain / trees | gfx1151 plain -> drafted |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        keys = sorted({k for (m, _), v in runs.items() if m == model for k in v})
        for k in keys:
            pc = runs.get((model, "plain (CPU contended)"), {}).get(k, {"tps": []})
            qc = runs.get((model, "drafted (CPU contended)"), {}).get(k, {"tps": []})
            p = runs.get((model, "plain"), {}).get(k, {"tps": [], "ttft": []})
            q = runs.get((model, "drafted"), {}).get(k, {"tps": [], "ttft": []})
            ch = runs.get((model, "chains"), {}).get(k, {"tps": []})
            l = runs.get((model, "llama.cpp"), {}).get(k, {"tps": []})
            ref = GFX1151.get((k[0], k[1])) if k[2] == 0 else None
            up = med(q["tps"]) / med(p["tps"]) if p["tps"] and q["tps"] else float("nan")
            print(f"| {k[0]} | {k[1]} | {k[2]:g} | {fmt(med(p['tps']))} | {fmt(med(q['tps']))} | {fmt(med(ch['tps']))} | {up:.2f}x | "
                  f"{fmt(med(l['tps']))} | {med(p['ttft']):.2f} / {med(q['ttft']):.2f} | {fmt(med(pc['tps']))} / {fmt(med(qc['tps']))} | "
                  f"{'%.1f -> %.1f' % ref if ref else ''} |")
        for arm in ("drafted", "chains"):
            if (model, arm) in acc:
                r, k, t = acc[(model, arm)]
                print(f"\nacceptance ({model}, {'trees' if arm == 'drafted' else 'chains'}): {k}/{t} drafts kept in {r} rounds; "
                      f"mean acceptance length {(k + r) / max(r, 1):.2f} tokens a round (kept drafts + bonus)")
    for f in sorted(glob.glob(os.path.join(d, "exact-*.json"))):
        bad = [(c["prompt"], c["streams"], c["temperature"], c["alone"], c["serial"]) for c in cells(f)
               if c.get("alone", {}).get("unequal") or c.get("serial", {}).get("unequal")]
        print(f"exactness {os.path.basename(f)}: " + ("drafted == alone == serial in every cell" if not bad else f"UNEQUAL {bad}"))


if __name__ == "__main__":
    main(sys.argv[1])
