# TensorFold DFlash2 on gfx1100: runner and sandbox evidence

`run.sh 1` … `run.sh 5` runs the five steps on the GPU machine, one at a time, each ending with a summary
(`$OUT/stepN/summary.txt`; step 5 also writes `report.md`). Settings are environment variables listed at the top
of `run.sh`; the usual ones:

```
export TF=the checkout this folder is in HIPCC=/opt/rocm/bin/hipcc
./run.sh 1 && ./run.sh 2 && ./run.sh 3
ADAPTER=~/adapters/pi DENSE=~/models/Qwen3.8-27B-bf16 ./run.sh 4     # DENSE optional, see below
LLAMA_SERVER=~/llama.cpp/build/bin/llama-server GGUF=~/models/Qwen3.8-27B-Q4_K_S.gguf GGUF_LORA=~/adapters/pi.gguf ./run.sh 5
```

Branches on github.com/eliasdarruda/TensorFold: `hip-dflash2` (PRs #3665 + #3918 on abb3665) and `brrr` (that plus
DFlash2 draft trees on HIP). On `brrr` trees are on by default; `TF_NO_TREES=1` gives chains, `TF_DRAFT_NODES=N`
sets the tree's nodes (15 by default), and `TF_TREE_LOG=1` logs every tree window and kept path. Step 4 checks both,
and step 5 measures plain, trees and chains (`ARMS`).

Notes the sandbox run turned up:

- **24 GiB and `check --drafter`.** On a virtual 24 GiB gfx1100 the check's radix phases run out of device memory
  with the drafter (5 FAIL with HipFailed/StreamOpen), and pass without it (46/0/7). With the drafter the check
  passed from 26 GiB up (51 passed, 0 failed, 2 skipped). A real card also loses some memory to the runtime and the
  desktop, so run it headless. Expect the lanes checks (drafted == serial, solo == together, graph == eager,
  resumed drafted == fresh drafted) to pass before the radix ones fail, and read those lines on their own.
- **Serving memory on 24 GiB** (the real checkpoint's tensor sizes): `--no-drafts` fits 3 streams at 32k context;
  `--drafter` fits 2 at 32k, 5 at 16k, 7 at 8k. That's why both speed arms use `--context 8192 --parallel 4`, so
  they compare at the same 4 streams.
- **Merge mode.** `merge MODEL ADAPTER OUT` folds the LoRA into the shipped 4-bit codes. TensorFold's docs measure
  about 0.1% of a rank-64 adapter landing that way at g64. `DENSE=<bf16 source>` uses `--quantize` instead (99.6%).
  Without `DENSE` the "merged" checkpoint is close to the base model.
- **llama.cpp bits per weight.** MLX affine 4-bit at g64 is 4.5 bpw. Q4_K_S (about 4.6) is the closest k-quant;
  Q4_K_M is about 4.8.

`sandbox/` has what produced the evidence:

- `fakehip.c` is a host-only `libamdhip64.so` that presents one gfx1100. It loads the real code objects, resolves
  every kernel the engine asks for, and checks every launch's block and LDS against gfx11 limits, but runs no
  kernels. Use it via `TENSORFOLD_HIP=/path/libamdhip64.so`.
- `fakeckpt.py` writes sparse checkpoints with the 27B's and the drafter's real tensor names and shapes (zeros, no
  download).
- `coinfo.py` reads each kernel's registers, spills, LDS and scratch from the code objects.
