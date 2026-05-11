"""Post-hoc cleaner: strip A-RAG's verbose-citation tail to make answers comparable
to baseline RAG systems (HippoRAG2/RAPTOR/ComoRAG) that produce concise answers.

A-RAG default prompt asks the agent to "Cite the specific chunks", so its answers
contain (Chunk ID: ...), (Chunk 4b30...), "This is supported by...", "This is
indicated by...", etc. Token-level F1 penalises this verbosity heavily.

This cleaner:
  1) drops chunk-citation parentheticals: `(Chunk ID: ...)`, `(Chunk ...)`, `[Chunk ...]`
  2) drops "supporting" / "indicated by" / "according to" trailing sentences
  3) trims to ≤ 2 sentences (concise answer convention)
"""
import re

CHUNK_PAT = re.compile(r"\(\s*chunk[^)]*\)|\[\s*chunk[^]]*\]", re.IGNORECASE)
TAIL_KILL = [
    "this is supported by", "this is indicated by", "this is shown by",
    "this is confirmed by", "this is mentioned in", "this is stated in",
    "this information is supported", "this information is indicated",
    "according to the chunks", "according to chunk",
    "as supported by", "as indicated by", "as mentioned in",
    "for more details", "if you want", "would you like", "i can try",
]

def clean_answer(text: str, max_sents: int = 2) -> str:
    if not text:
        return ""
    s = text.strip()
    # 1) strip parenthetical chunk citations
    s = CHUNK_PAT.sub("", s)
    # 2) tail-kill sentences after a "supporting evidence" cue
    sents = re.split(r"(?<=[.!?])\s+", s)
    clean = []
    for sent in sents:
        low = sent.lower().strip(' .,;:"')
        if any(low.startswith(k) or k in low[:60] for k in TAIL_KILL):
            break
        clean.append(sent)
    out = " ".join(clean[:max_sents]).strip()
    # tidy: extra spaces from removed parens
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+([.,;:!?])", r"\1", out)
    return out.strip()


if __name__ == "__main__":
    tests = [
        ("Mary Horowitz's job is a news anchor. This is supported by the text mentioning \"CNN ANCHOR Mary Horowitz\" and \"NEWS ANCHOR Mary Horowitz\" in the broadcast context. (Chunk ID: 4b30ab1c49b62dc59b9773954958d9ac6807a865_187)",
         "Mary Horowitz's job is a news anchor."),
        ("Mary's blind date is Howard. He is a retired mine supervisor. This is supported by the text in chunk 4b30ab1c49b62dc59b9773954958d9ac6807a865_206 which states: \"RETIRED MINE SUPERVISOR ... Howard stands at the barrier looking more worried than anybody.\" Additionally, Mary meets Howard through Elizabeth, and Howard has a Plymouth Colt car from 1989.",
         "Mary's blind date is Howard. He is a retired mine supervisor."),
        ("The provided chunks do not contain information about why Mary got fired. They mostly describe interactions and events involving Mary but do not mention her being fired or the reasons behind it. If you want, I can try a different search approach or keywords to find relevant information. Would you like me to do that?",
         "The provided chunks do not contain information about why Mary got fired."),
    ]
    for raw, expect in tests:
        got = clean_answer(raw)
        ok = "✓" if got.startswith(expect[:40]) else "✗"
        print(f"{ok}\n  RAW: {raw[:120]}\n  CLN: {got}\n  EXP: {expect[:120]}\n")
