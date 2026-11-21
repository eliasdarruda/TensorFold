#!/usr/bin/env bash
# Steps 1-5 of the gfx1100 DFlash2 evaluation of TensorFold PRs #3665 + #3918 (branch hip-dflash2).
# Run one step at a time and read its summary before the next:   ./run.sh 1   ...   ./run.sh 5
# Every step logs to $OUT/stepN/ and prints a short summary at the end.
#
# Settings (environment; defaults in brackets):
#   TF        TensorFold checkout on branch hip-dflash2                        [~/TensorFold]
#   HIPCC     hipcc                                                             [/opt/rocm/bin/hipcc]
#   HIP_DEVICE_LIB  device bitcode dir, only if the build cannot find it       [unset]
#   OUT       where logs and results go                                         [~/tf-gfx1100-<date>]
#   MODEL     27B checkpoint dir     [resolved from the HF cache after `pull TensorFold/Qwen3.8-27B-MLX-4bit`]
#   DRAFTER   DFlash2 drafter dir    [resolved from the HF cache after `pull z-lab/Qwen3.8-27B-DFlash2`]
#   ADAPTER   the pi PEFT LoRA folder (step 4 merge)                            [required for the merge]
#   DENSE     the 27B's bf16 source checkpoint, for `merge --quantize`          [unset: fold merge, see below]
#   MERGED    merged checkpoint dir                                             [$OUT/Qwen3.8-27B-pi-MLX-4bit]
#   CONTEXT   served context for both speed arms (24 GiB fits 4 streams at 8192) [8192]
#   PARALLEL  streams both speed arms serve                                     [4]
#   LEVELS    concurrent streams measured                                       [1,2,4]
#   TOKENS    tokens a request                                                  [256]
#   REPS      repetitions a cell                                                [3]
#   TEMPS     temperatures                                                      [0]
#   PORT      server port                                                       [18080]
#   ARMS      speed arms in order: plain (--no-drafts), drafted (DFlash2 trees), chains (TF_NO_TREES=1)
#             [plain drafted chains]
#   INTERFERE CPU hogs (busy processes) for an extra contended round per arm, Blink-style; 0 = none [0]
#   LLAMA_SERVER  llama.cpp's llama-server binary (step 5 comparison)           [unset: skipped]
#   GGUF      the 27B GGUF at the closest bits per weight (Q4_K_S ~4.6 bpw vs MLX 4-bit g64's 4.5)
#   GGUF_LORA the pi adapter as a GGUF LoRA, for the merged comparison          [unset]
#   LLAMA_ARGS extra llama-server flags                                         [-ngl 999 -fa on -c 8192 -np 4]
#   FT         FreeToken's `ft` binary (step 5 comparison)                       [unset: skipped]
#   FT_MODEL   .gguf or HF dir for FreeToken; FT_DRAFT bf16 DFlash2 dir; FT_ARGS extra flags
set -uo pipefail

STEP=${1:?usage: run.sh 1|2|3|4|5}
TF=${TF:-$(cd "$(dirname "$0")/../.." && pwd)}
HIPCC=${HIPCC:-/opt/rocm/bin/hipcc}
OUT=${OUT:-$HOME/tf-gfx1100-$(date +%Y%m%d)}
CONTEXT=${CONTEXT:-8192}
PARALLEL=${PARALLEL:-4}
LEVELS=${LEVELS:-1,2,4}
TOKENS=${TOKENS:-256}
REPS=${REPS:-3}
TEMPS=${TEMPS:-0}
PORT=${PORT:-18080}
INTERFERE=${INTERFERE:-0}
LLAMA_ARGS=${LLAMA_ARGS:--ngl 999 -fa on -c 8192 -np 4}
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=$TF/zig-out/native-hip/bin/tensorfold-native
CHECK=$TF/zig-out/bin/tf-qwen35-test
HIPFLAGS=(-Dhip=true -Dhip-arch=gfx1100 "-Dhipcc=$HIPCC")
[ -n "${HIP_DEVICE_LIB:-}" ] && HIPFLAGS+=("-Dhip-device-lib=$HIP_DEVICE_LIB")
[ -n "${RELAX:-}" ] && HIPFLAGS+=(-Dhip-relax-branches=true)
D=$OUT/step$STEP
mkdir -p "$D"
say() { printf '%s\n' "$*" | tee -a "$D/summary.txt"; }

# a cached Hugging Face repo's newest snapshot directory
snapshot() {
  local hub=${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}
  ls -d "$hub/models--${1/\//--}/snapshots/"* 2>/dev/null | tail -1
}
resolve_models() {
  MODEL=${MODEL:-$(snapshot TensorFold/Qwen3.8-27B-MLX-4bit)}
  DRAFTER=${DRAFTER:-$(snapshot z-lab/Qwen3.8-27B-DFlash2)}
  MERGED=${MERGED:-$OUT/Qwen3.8-27B-pi-MLX-4bit}
}

# serve <log> <args...>: start a server and wait for it; sets SERVER_PID
serve() {
  local log=$1; shift
  env ${SERVE_ENV:-} "$BIN" serve "$@" --name bench --port "$PORT" >"$log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 600); do
    grep -q "serving bench at" "$log" && return 0
    kill -0 $SERVER_PID 2>/dev/null || { say "server exited: $(grep -m1 'tensorfold:' "$log")"; return 1; }
    sleep 1
  done
  say "server did not come up in 600 s"; return 1
}
stop() { kill -INT "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; sleep 3; }
metric() { curl -s "http://127.0.0.1:$PORT/metrics" | awk -v k="tensorfold:$1" '$1==k {print $2}'; }

case $STEP in
1)
  say "== step 1: toolchain and branch ($(date))"
  say "branch: $(git -C "$TF" rev-parse --abbrev-ref HEAD) $(git -C "$TF" rev-parse --short HEAD); base abb3665: $(git -C "$TF" merge-base --is-ancestor abb3665b58c3920145e7ff0abc287375c7f9b7a8 HEAD && echo ancestor || echo MISSING)"
  say "PR commits on top: $(git -C "$TF" rev-list --count abb3665b58c3920145e7ff0abc287375c7f9b7a8..HEAD) (expect 8)"
  say "ROCm: $(cat /opt/rocm/.info/version 2>/dev/null || echo unknown)"
  say "hipcc: $HIPCC -> $("$HIPCC" --version 2>&1 | grep -m1 -iE 'clang version')"
  say "HIP: $("$HIPCC" --version 2>&1 | grep -m1 'HIP version')"
  say "zig: $(zig version) (need >= 0.17.0)"
  say "GPUs: $(rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | sort | uniq -c | tr '\n' ' ')"
  say "VRAM: $(rocm-smi --showmeminfo vram 2>/dev/null | grep -E 'Total Memory|Total Used' | sed 's/.*: //' | tr '\n' ' ')"
  "$HIPCC" --version >"$D/hipcc.txt" 2>&1; rocminfo >"$D/rocminfo.txt" 2>&1
  ;;
2)
  say "== step 2: kernels against their host references on the GPU ($(date))"
  cd "$TF"
  for s in hip-gpu-test hip-affine-test hip-kernel-test; do
    zig build "${HIPFLAGS[@]}" $s --summary all >"$D/$s.log" 2>&1; rc=$?
    if [ $rc -ne 0 ] && grep -q "simm16" "$D/$s.log" && [ -z "${RELAX:-}" ]; then
      say "$s: branch size exceeds simm16 (#1521): retrying with -Dhip-relax-branches=true"
      zig build "${HIPFLAGS[@]}" -Dhip-relax-branches=true $s --summary all >"$D/$s-relaxed.log" 2>&1; rc=$?
      tail -3 "$D/$s-relaxed.log" | tee -a "$D/summary.txt"
    fi
    say "$s: exit $rc; $(grep -m1 -E 'Build Summary' "$D/$s.log")"
    grep -E "^(FAIL|fail)|error:|expected|mismatch" "$D/$s.log" | head -10 >>"$D/summary.txt"
  done
  ;;
3)
  say "== step 3: builds ($(date))"
  cd "$TF"
  zig build "${HIPFLAGS[@]}" tf-qwen35-test >"$D/tf-qwen35-test.log" 2>&1; say "tf-qwen35-test: exit $? -> $CHECK"
  zig build "${HIPFLAGS[@]}" native-hip -Doptimize=ReleaseFast >"$D/native-hip.log" 2>&1; say "native-hip: exit $? -> $BIN"
  say "capabilities: $("$BIN" capabilities --json | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["backends"], "qwen3_5:", len(d["families"].get("qwen3_5",[])), "formats")')"
  ;;
4)
  say "== step 4: exactness under DFlash2 ($(date))"
  resolve_models
  [ -d "$MODEL" ] || { "$BIN" pull TensorFold/Qwen3.8-27B-MLX-4bit | tee "$D/pull-model.log"; MODEL=$(snapshot TensorFold/Qwen3.8-27B-MLX-4bit); }
  [ -d "$DRAFTER" ] || { "$BIN" pull z-lab/Qwen3.8-27B-DFlash2 | tee "$D/pull-drafter.log"; DRAFTER=$(snapshot z-lab/Qwen3.8-27B-DFlash2); }
  say "model $MODEL"; say "drafter $DRAFTER"
  check() { # check <name> <dir> [--drafter DIR]
    local name=$1 dir=$2; shift 2
    "$CHECK" check "$dir" "$@" >"$D/check-$name.log" 2>&1
    say "check $name: $(grep -E '^check (PASS|FAIL)' "$D/check-$name.log")"
    grep -E "^FAIL" "$D/check-$name.log" | tee -a "$D/summary.txt" | head -20 >/dev/null
    if grep -q "HipFailed\|StreamOpen" "$D/check-$name.log"; then
      say "  note: HipFailed/StreamOpen failures are device-memory exhaustion (the radix checks hold the most streams;"
      say "  with --drafter each also holds tap rings). On the fake 24 GiB card this needed ~26 GiB; run headless or see the README."
    fi
  }
  check base-drafter "$MODEL" --drafter "$DRAFTER"
  # the brrr branch drafts trees by default: the same checks on chains (TF_NO_TREES=1) for comparison
  TF_NO_TREES=1 check base-drafter-chains "$MODEL" --drafter "$DRAFTER"
  check base-plain "$MODEL"
  if [ -z "${ADAPTER:-}" ]; then say "ADAPTER unset: merge skipped. Set ADAPTER=/path/to/pi-adapter and rerun step 4."; exit 0; fi
  if [ ! -d "$MERGED" ]; then
    if [ -n "${DENSE:-}" ]; then
      say "merge: fresh quantization from the bf16 source (merge DENSE ADAPTER OUT --quantize SHIPPED)"
      "$BIN" merge "$DENSE" "$ADAPTER" "$MERGED" --quantize "$MODEL" >"$D/merge.log" 2>&1; say "merge: exit $?"
      (cd "$TF" && zig build tf-merge-check >/dev/null 2>&1 && ./zig-out/bin/tf-merge-check "$DENSE" "$ADAPTER" "$MERGED" "$MODEL") >"$D/merge-check.log" 2>&1
    else
      say "merge: FOLD into the shipped 4-bit codes (DENSE unset). docs/quantization.md: a fold lands ~0.1% of a rank-64"
      say "  adapter at g64 (codes move only past half a step); set DENSE=<bf16 Qwen3.8-27B> for --quantize (99.6%)."
      "$BIN" merge "$MODEL" "$ADAPTER" "$MERGED" >"$D/merge.log" 2>&1; say "merge: exit $?"
      (cd "$TF" && zig build tf-merge-check >/dev/null 2>&1 && ./zig-out/bin/tf-merge-check "$MODEL" "$ADAPTER" "$MERGED") >"$D/merge-check.log" 2>&1
    fi
    say "tf-merge-check: exit $? ($(tail -1 "$D/merge-check.log"))"
  fi
  check merged-drafter "$MERGED" --drafter "$DRAFTER"
  ;;
5)
  say "== step 5: speed, plain (--no-drafts) vs drafted (--drafter), alternating A1 B1 A2 B2 ($(date))"
  resolve_models
  models=(base:"$MODEL"); [ -d "$MERGED" ] && models+=(merged:"$MERGED")
  for m in "${models[@]}"; do
    name=${m%%:*}; dir=${m#*:}
    rounds=(1 2); [ "$INTERFERE" -gt 0 ] && rounds+=(contended)
    for round in "${rounds[@]}"; do
      for arm in ${ARMS:-plain drafted chains}; do
        tag=$name-$arm-$round
        # drafted: trees (the brrr branch's default); chains: the same drafter with TF_NO_TREES=1
        SERVE_ENV=""
        if [ $arm = plain ]; then a=(--no-drafts); else a=(--drafter "$DRAFTER"); fi
        [ $arm = chains ] && SERVE_ENV="TF_NO_TREES=1"
        serve "$D/serve-$tag.log" "$dir" "${a[@]}" --context "$CONTEXT" --parallel "$PARALLEL" || continue
        grep -m1 "streams at once" "$D/serve-$tag.log" | sed 's/^\[tensorfold\] //' >>"$D/summary.txt"
        hogs=(); if [ "$round" = contended ]; then for _ in $(seq 1 "$INTERFERE"); do yes >/dev/null & hogs+=($!); done; fi
        r0=$(metric decode_rounds_total); k0=$(metric mtp_accepted_total); t0=$(metric mtp_drafted_total)
        (cd "$TF/tools" && python3 bench_concurrent.py "http://127.0.0.1:$PORT" bench --levels "$LEVELS" --tokens "$TOKENS" \
          --reps "$REPS" --temperatures "$TEMPS" --label "$tag" --output "$D/bench-$tag.json") >"$D/bench-$tag.log" 2>&1
        [ ${#hogs[@]} -gt 0 ] && kill "${hogs[@]}" 2>/dev/null
        r1=$(metric decode_rounds_total); k1=$(metric mtp_accepted_total); t1=$(metric mtp_drafted_total)
        echo "{\"rounds\": $((r1 - r0)), \"accepted\": $((k1 - k0)), \"drafted\": $((t1 - t0))}" >"$D/drafts-$tag.json"
        # drafted replies against the same requests alone and with drafts off (token_sha256), outside the timed runs
        [ $arm != plain ] && (cd "$TF/tools" && python3 bench_concurrent.py "http://127.0.0.1:$PORT" bench --levels 1,"$PARALLEL" \
          --tokens "$TOKENS" --reps 1 --temperatures 0,1.0 --alone --serial --label "$tag-exact" --output "$D/exact-$tag.json") >"$D/exact-$tag.log" 2>&1
        [ $arm = drafted ] && [ $round = 1 ] && python3 "$HERE/agree.py" collect "http://127.0.0.1:$PORT" bench "$D/text-tf-$name.json"
        stop
      done
    done
  done
  if [ -n "${LLAMA_SERVER:-}" ] && [ -n "${GGUF:-}" ]; then
    for name in base merged; do
      lora=(); [ $name = merged ] && { [ -n "${GGUF_LORA:-}" ] || continue; lora=(--lora "$GGUF_LORA"); }
      # shellcheck disable=SC2086
      "$LLAMA_SERVER" -m "$GGUF" "${lora[@]}" $LLAMA_ARGS --port "$PORT" --alias bench >"$D/llama-$name.log" 2>&1 &
      SERVER_PID=$!
      for _ in $(seq 1 600); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break; sleep 1; done
      (cd "$TF/tools" && python3 bench_concurrent.py "http://127.0.0.1:$PORT" bench --levels "$LEVELS" --tokens "$TOKENS" \
        --reps "$REPS" --temperatures "$TEMPS" --label "llama-$name" --output "$D/bench-llama-$name.json") >"$D/bench-llama-$name.log" 2>&1
      python3 "$HERE/agree.py" collect "http://127.0.0.1:$PORT" bench "$D/text-llama-$name.json"
      stop
      [ -f "$D/text-tf-$name.json" ] && python3 "$HERE/agree.py" compare "$D/text-tf-$name.json" "$D/text-llama-$name.json" | tee -a "$D/summary.txt"
    done
  else
    say "LLAMA_SERVER or GGUF unset: llama.cpp comparison skipped"
  fi
  # FreeToken (github.com/FlashML-org/FreeToken): same 27B + DFlash2, ROCm path is experimental.
  # FT_MODEL: a .gguf (e.g. the merged pi model at Q4_K_M) or an HF dir; FT_DRAFT: the bf16 DFlash2 checkpoint
  # (FreeToken refuses quantized safetensors drafts). FT_ARGS: extra flags.
  if [ -n "${FT:-}" ] && [ -n "${FT_MODEL:-}" ]; then
    for arm in plain drafted; do
      spec=(); [ $arm = drafted ] && { [ -n "${FT_DRAFT:-}" ] || continue; spec=(--speculative-algorithm dflash --speculative-draft-model-path "$FT_DRAFT"); }
      # shellcheck disable=SC2086
      "$FT" serve --model "$FT_MODEL" "${spec[@]}" --served-model-name bench --port "$PORT" --max-running-requests "$PARALLEL" \
        ${FT_ARGS:-} >"$D/ft-$arm.log" 2>&1 &
      SERVER_PID=$!
      for _ in $(seq 1 1200); do curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && break; sleep 1; done
      (cd "$TF/tools" && python3 bench_concurrent.py "http://127.0.0.1:$PORT" bench --levels "$LEVELS" --tokens "$TOKENS" \
        --reps "$REPS" --temperatures "$TEMPS" --label "ft-$arm" --output "$D/bench-ft-$arm.json") >"$D/bench-ft-$arm.log" 2>&1
      python3 "$HERE/agree.py" collect "http://127.0.0.1:$PORT" bench "$D/text-ft-$arm.json"
      grep -i "accept len" "$D/ft-$arm.log" | tail -3 | tee -a "$D/summary.txt"
      stop
      [ -f "$D/text-tf-merged.json" ] && python3 "$HERE/agree.py" compare "$D/text-tf-merged.json" "$D/text-ft-$arm.json" | tee -a "$D/summary.txt"
    done
  else
    say "FT or FT_MODEL unset: FreeToken comparison skipped"
  fi
  python3 "$HERE/report.py" "$D" | tee "$D/report.md"
  ;;
*) echo "usage: run.sh 1|2|3|4|5"; exit 2 ;;
esac
say "== step $STEP done; logs in $D"
