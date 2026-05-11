#!/usr/bin/env bash
# rag_test — clone + install the four external RAG systems.
#
# Assumes 4 sibling clones already exist (or will be made), as documented in
# EXTERNAL_REPOS.md.  If a sibling is missing, this script clones it. Then it
# checks out a known-good commit, applies any patches (ComoRAG), and pip-installs.
#
# Required env: none. Run from rag_test/ root.
# Optional: set $PARENT to override the parent directory used for sibling clones.

set -euo pipefail

THIS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RAG_TEST="$(cd "$THIS/.." && pwd)"
PARENT="${PARENT:-$(dirname "$RAG_TEST")}"

# (repo dir, github URL, frozen commit, install cmd)  — keep in sync with EXTERNAL_REPOS.md
SYSTEMS=(
    "HippoRAG|OSU-NLP-Group/HippoRAG|main|pip install -e ."
    "raptor|parthsarthi03/raptor|main|pip install -e ."
    "ComoRAG|EternityJune25/ComoRAG|a4f8433|pip install -e ."
    "arag|Ayanami0730/arag|a44de6b|pip install -e ."
)

log() { printf "\033[1;34m[setup]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[setup-warn]\033[0m %s\n" "$*"; }
die() { printf "\033[1;31m[setup-err]\033[0m %s\n" "$*"; exit 1; }

log "rag_test root: $RAG_TEST"
log "sibling parent: $PARENT"

# 1) Common Python deps used by rag_test wrappers + eval
log "(1/4) installing rag_test common deps"
pip install -q --upgrade pip
pip install -q numpy pandas pyyaml sentence-transformers tiktoken openai tqdm

# 2) Clone + checkout + install each system
i=0
for sys_spec in "${SYSTEMS[@]}"; do
    i=$((i+1))
    IFS='|' read -r dir url hash install <<< "$sys_spec"
    target="$PARENT/$dir"
    log "(2/4) [$i/${#SYSTEMS[@]}] $dir  →  $target"

    if [ ! -d "$target" ]; then
        log "    cloning https://github.com/$url"
        git clone --quiet "https://github.com/$url.git" "$target"
    else
        log "    already exists at $target — skipping clone"
    fi

    if [ "$hash" != "main" ] && [ "$hash" != "" ]; then
        log "    checking out $hash"
        (cd "$target" && git fetch --quiet && git checkout --quiet "$hash") \
            || warn "    checkout of $hash failed — staying on current HEAD"
    fi
done

# 3) Apply ComoRAG patches (after checkout, before install)
log "(3/4) applying ComoRAG patches"
COMORAG_DIR="$PARENT/ComoRAG"
if [ -d "$COMORAG_DIR" ]; then
    for p in "$RAG_TEST"/patches/comorag/*.patch; do
        [ -f "$p" ] || continue
        # idempotent: check if already applied
        if (cd "$COMORAG_DIR" && git apply --reverse --check "$p") 2>/dev/null; then
            log "    skipping $(basename "$p") — already applied"
        else
            log "    applying $(basename "$p")"
            (cd "$COMORAG_DIR" && git apply "$p") \
                || warn "    patch failed — manual merge may be required"
        fi
    done
    # New file (not a diff): copy if missing
    SBERT_DST="$COMORAG_DIR/src/comorag/embedding_model/SBERTEmbedding.py"
    if [ ! -f "$SBERT_DST" ]; then
        log "    copying SBERTEmbedding.py"
        cp "$RAG_TEST/patches/comorag/SBERTEmbedding.py" "$SBERT_DST"
    fi
fi

# 4) Install each system (separate so patches are already applied for ComoRAG)
log "(4/4) pip install each external system"
for sys_spec in "${SYSTEMS[@]}"; do
    IFS='|' read -r dir url hash install <<< "$sys_spec"
    target="$PARENT/$dir"
    log "    installing $dir"
    ( cd "$target" && eval "$install" )
done

log "✓ setup complete."
log "Verify path resolution:  python3 $RAG_TEST/utils/paths.py"
log "Next: export OPENAI_API_KEY=...; then bash $RAG_TEST/scripts/run_all.sh"
