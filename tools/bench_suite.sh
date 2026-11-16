#!/bin/bash
# Run every benchmark with fixed arguments and store the raw output under logs/.
# Usage: tools/bench_suite.sh [tag]
set -u
cd "$(dirname "$0")/.."
TAG=${1:-$(date -u +%Y%m%dT%H%M)}
OUT=logs/suite_$TAG
mkdir -p "$OUT"
B=./build/bin

run() {
  local name=$1; shift
  echo "== $name"
  timeout 3600 "$@" > "$OUT/$name.txt" 2>&1
  echo "   exit=$? $(tail -1 "$OUT/$name.txt" 2>/dev/null | cut -c1-140)"
}

run ctest ctest --test-dir build --output-on-failure
run sim $B/sim 1 3000 0 0 12
run curriculum $B/curriculum
run sequence $B/sequence
run textmodel $B/textmodel
run streamlm $B/streamlm data/wikitext2.txt 3000 0.005 1 800000 1 "" 1 0 0 "the" 1.0 1 0 0 1
run charlm_text $B/charlm data/wikitext2_head.txt 0 0.004 0 "" 0 "" 0 4
run charlm_code $B/charlm data/code_train.txt 0 0.004 0 data/code_valid.txt 0 "" 1 6 16
run sdm $B/sdm
run novelty $B/novelty
run contextfree $B/contextfree
run cleanup $B/cleanup
run powermem $B/powermem
run swebench $B/swebench data/swebench_lite.tsv 1024 2
run wordspeech $B/wordspeech
run phonotactics $B/phonotactics
run audit $B/audit

echo "suite written to $OUT"