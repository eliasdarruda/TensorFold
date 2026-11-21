"""Greedy output agreement between two OpenAI-compatible servers (TensorFold and llama.cpp), standard library only.

  agree.py collect URL MODEL OUT.json   # greedy chat replies to PROMPTS, saved with their texts
  agree.py compare A.json B.json        # per prompt: common prefix (chars, words) and word-level match

Two quantizations of one model drift apart after some tokens even when both are right, so the useful numbers are
how long the replies agree from the start, and how many words match position by position.
"""
import json
import sys
import urllib.request

PROMPTS = [
    "Write a short Python function that computes the Fibonacci sequence and explain it.",
    "Explain how matrix multiplication uses a GPU in plain English, then give a small numerical example.",
    "Write a bash one-liner that counts the lines of every .py file under the current directory, sorted.",
    "What are the three laws of thermodynamics? One sentence each.",
    "Translate to French: 'The meeting was moved to Thursday because the lab was closed.'",
    "Write a SQL query that returns the five customers with the highest total order value.",
    "Summarize the causes of the 2008 financial crisis in four bullet points.",
    "Implement binary search in C and state its time complexity.",
]


def chat(url, model, prompt, tokens=256):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": tokens,
            "temperature": 0, "seed": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    m = d["choices"][0]["message"]
    return (m.get("reasoning_content") or "") + (m.get("content") or "")


def collect(url, model, out):
    json.dump([{"prompt": p, "text": chat(url, model, p)} for p in PROMPTS], open(out, "w"), indent=1)


def compare(a_path, b_path):
    a, b = json.load(open(a_path)), json.load(open(b_path))
    rows = []
    for x, y in zip(a, b):
        s, t = x["text"], y["text"]
        c = next((i for i, (p, q) in enumerate(zip(s, t)) if p != q), min(len(s), len(t)))
        ws, wt = s.split(), t.split()
        cw = next((i for i, (p, q) in enumerate(zip(ws, wt)) if p != q), min(len(ws), len(wt)))
        same = sum(p == q for p, q in zip(ws, wt)) / max(1, max(len(ws), len(wt)))
        rows.append((x["prompt"][:48], c, cw, same))
    print(f"agreement {a_path.rsplit('/', 1)[-1]} vs {b_path.rsplit('/', 1)[-1]}:")
    for p, c, cw, same in rows:
        print(f"  {p:50s} common prefix {c:5d} chars / {cw:4d} words; words equal by position {same:6.1%}")
    print(f"  median common prefix {sorted(r[2] for r in rows)[len(rows) // 2]} words; "
          f"mean positional word match {sum(r[3] for r in rows) / len(rows):.1%}")


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        collect(*sys.argv[2:5])
    elif sys.argv[1] == "compare":
        compare(*sys.argv[2:4])
    else:
        sys.exit(__doc__)
