# Handoff: TensorFold + DFlash2 on gfx1100

Context for continuing this evaluation in a local Claude Code session on the gfx1100 machine.

## Goal

Evaluate TensorFold's native HIP engine with DFlash2 speculative decoding on an AMD gfx1100 (7900 XTX, 24 GB),
against a tuned llama.cpp. Model: Qwen3.8-27B (qwen3_5 family: 48 Gated DeltaNet + 16 full-attention layers)
with the user's LoRA adapter "pi". Then make it as fast as possible.

## Constraints from the user

- One GPU rank only (no `--tp`). 4-bit drafter.
- Originally "chains not trees"; later the user approved porting draft-tree verification. Both are benchmarked
  (`ARMS="plain drafted chains"`, chains = `TF_NO_TREES=1`).
- Do not post anything upstream (ashhart/TensorFold) without asking. If results are good, draft a comment for
  PR #3665 with the gfx1100 numbers for the user to review.
- Ask the user for the pi adapter path at step 4 (and an optional bf16 source for `--quantize`).

## Repo state

- Fork: github.com/eliasdarruda/TensorFold. Remote `upstream` = ashhart/TensorFold.
- `hip-dflash2`: abb3665 + cherry-picked `f0a54bd..pr3918` (DFlash2 on HIP, PRs #3665/#3918, by AbrahamCete,
  measured only on gfx1151).
- `brrr` (this branch): hip-dflash2 + tree verification on the dense qwen3_5 HIP engine (commit c404a8b97) +
  this runner.

### What the tree port does (c404a8b97)

- Tree rows sit at physical cache positions base+i; logical position = base+depth (RoPE uses logical).
- Attention remaps keys >= base through each row's ancestor path (4-bit nibbles, max 16 rows):
  `plan.hpp` (`plan_see`, `plan_key`, `plan_sees`, `plan_parent`, `plan_kept`), `plan.hip` score/apply/attend.
- Conv (`decode.hip conv_split_body`) and GDN (`gated_delta.hip`) rebuild from the window-start state along
  ancestors when parent != r-1.
- Keep follows the kept path: conv window, GDN replay of path rows, final row = last path row, and the new
  `tf_plan_keep_kv` kernel moves kept K/V to their logical positions.
- Host: `forward/plan.zig` (depth/path/kept/full in Layout, `treeRows`, `keptPath`), `engine/round.zig`
  (`markPath`, paths upload), `engine/hip_lanes.zig` (`keepPath`, refuses trees under TP), `hip_serve/backend.zig`
  (`windowRows(trees)`, `Pending.tree`, `TreeNotKept`), drafter `draft.zig` (`tree_nodes` from `TF_DRAFT_NODES`,
  `walker.grow`, landing chances). `TF_TREE_LOG=1` logs tree windows and keeps.
- Aim: bit-exact with serial decode.

### What was verified without a GPU (sandbox, fake HIP runtime)

- All host test suites pass; all kernels compile for gfx1100; no scratch/spills (apply VGPRs 57 -> 42).
- `check --drafter` on fake checkpoints: 51/0/2 with trees and with chains.
- NOT verified: numerical exactness on real hardware, and speed. That is what steps 2-5 are for.

## Runner (`bench/gfx1100/run.sh`)

`./bench/gfx1100/run.sh N` for N = 1..5; logs go to a dated dir, `summary.txt` and `report.md`. See README.md.

1. Setup checks: ROCm, hipcc, Zig >= 0.17.0 (PyPI 0.17.0 is yanked; use 0.17.0.post0).
2. Kernel correctness: `zig build hip-kernel-test -Dhip=true -Dhip-arch=gfx1100 -Dhipcc=...`; on the simm16
   error (#1521) retry with `-Dhip-relax-branches=true`.
3. Builds: `tf-qwen35-test` and `native-hip -Doptimize=ReleaseFast`.
4. Exactness: `tf-qwen35-test check <27B> --drafter <DFlash2>` (gfx1151 reference: 49/0/1), chains and plain
   too; then `tensorfold merge` the pi adapter and recheck. Known: `check --drafter` OOMs at 24 GiB (it passed
   at 26 GiB in the sandbox) — a real finding to fix or report.
5. Speed: alternate plain / drafted / chains (A1 B1 A2 B2) at 1, 2, 4 streams, code and chat prompts. Reports
   tok/s, TTFT, acceptance; exactness pass; llama.cpp at matched bits per weight (`LLAMA_SERVER`, `GGUF`,
   `GGUF_LORA`) with output agreement; optional FreeToken arm (`FT`, `FT_MODEL`, `FT_DRAFT`; its ROCm support
   is experimental, install from its docs/install_amd.md with arch gfx1100, no `[accel]` extras).

gfx1151 reference (1 stream): code 14.4 -> 34.8 tok/s, chat 14.8 -> 28.0 tok/s.

## Expectations (estimates, not measurements)

- Plain decode ~32 tok/s; llama.cpp ~35-40; drafted average ~55 tok/s (range 40-70) for the pi harness at
  20-50k context; 80-130 tok/s possible on short predictable code. 256k context doesn't fit on 24 GB
  (KV ~64 KB/token, model+drafter 15.2 GiB).

## Next steps

1. Run steps 2-5 in order; stop and show the user the results after each step.
2. Fix any FAIL lines (kernel mismatches, exactness divergences) before trusting speed numbers.
3. After results, further speed work:
   - #3811: share K/V reads across rows in the decode walk.
   - Tune `TF_DRAFT_NODES` (try 7 and 11).
   - Lower-bit weights.
   - Profile gaps between rounds.
   - Try `--recall` on the real pi workload.
   - Fix the `check --drafter` OOM at 24 GiB.
4. If results are good, draft (do not post) the PR #3665 comment with the gfx1100 numbers.
