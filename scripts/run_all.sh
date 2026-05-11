#!/usr/bin/env bash
# rag_test — run all 4 RAG systems on NarrativeQA dev_10_doc (293 queries).
#
# Prerequisites:
#   * scripts/setup_env.sh has been run (4 sibling clones + installs)
#   * OPENAI_API_KEY exported
#   * (optional) WHICH="hipporag2 raptor comorag arag"  ← subset to run
#
# Each experiment writes its own results/ inside experiments/exp-XXX-.../

set -euo pipefail
THIS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RAG_TEST="$(cd "$THIS/.." && pwd)"
cd "$RAG_TEST"

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY not set."
    exit 1
fi
export ARAG_API_KEY="$OPENAI_API_KEY"     # A-RAG batch_runner reads either name

WHICH="${WHICH:-hipporag2 raptor comorag arag}"

log() { printf "\033[1;34m[run]\033[0m %s\n" "$*"; }

for sys in $WHICH; do
    case "$sys" in
        hipporag2)
            log "exp-003-hipporag2-gpt"
            python3 experiments/exp-003-hipporag2-gpt/run.py
            ;;
        raptor)
            log "exp-004-raptor-gpt"
            python3 experiments/exp-004-raptor-gpt/run.py
            ;;
        comorag)
            log "exp-005-comorag-narrativeqa"
            python3 experiments/exp-005-comorag-narrativeqa/run.py
            ;;
        arag)
            log "exp-006-arag-narrativeqa"
            python3 experiments/exp-006-arag-narrativeqa/run.py
            ;;
        *)
            echo "WARN: unknown system '$sys' — skipping"
            ;;
    esac
done

log "✓ all done. building comparison table…"
python3 "$RAG_TEST/scripts/build_comparison.py"
