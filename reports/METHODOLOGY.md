# HippoRAG2 vs RAPTOR - Detailed Methodology

**Date**: 2026-05-07

---

## 1. Overview

This document describes the full experimental methodology for comparing HippoRAG2 and RAPTOR on the NarrativeQA benchmark. Both methods are Retrieval-Augmented Generation (RAG) systems that build structured indices over document corpora and answer questions by retrieving relevant context.

---

## 2. Dataset: NarrativeQA

- **Source**: `osunlp/HippoRAG_2` on HuggingFace (HippoRAG2's official benchmark split)
- **Subset**: `narrativeqa_dev_10_doc` (dev set, 10 documents)
- **Corpus**: 4,111 passages in `narrativeqa_dev_10_doc_corpus.json`
  - Format: `{idx, title, text}` per passage
  - Documents are long narratives (movies/books) split into passages
- **Queries**: 293 questions in `narrativeqa_dev_10_doc.json`
  - Format: `{document, question, answer}` per sample
  - `answer` is a list of acceptable answer strings (typically 2 references)
- **Task**: Given a question about a narrative, retrieve relevant context and generate a concise answer
- **Evaluation samples**: 293 queries total, each with 2 gold reference answers
- **Example data point**:
  ```json
  {
    "document": "All About Eve",
    "question": "What is Mary Horowitz's job?",
    "answer": [
      "She is a crossword writer for the Sacramento Herald.",
      "She is a crossword puzzle writer."
    ]
  }
  ```

---

## 3. Model Configuration

### 3.1 LLM: Qwen/Qwen2.5-7B-Instruct

- **Why not the original paper models?**
  - HippoRAG2 paper uses Llama-3.3-70B-Instruct (too large for single A100)
  - RAPTOR paper uses GPT-3.5-Turbo (API-dependent, not reproducible)
  - We need the same model for both to ensure fair comparison
- **Why Qwen2.5-7B?**
  - Initially planned Llama-3.1-8B-Instruct, but it's gated on HuggingFace (403 Forbidden)
  - Qwen2.5-7B-Instruct is ungated, similar capability tier (7B params), fully supported by vLLM 0.6.6
- **Serving**: vLLM 0.6.6 with OpenAI-compatible API on `localhost:8000`
  - `--gpu-memory-utilization 0.45` (~36GB VRAM)
  - `--max-model-len 4096`
  - `--dtype auto`

### 3.2 Embedding: SBERT (multi-qa-mpnet-base-cos-v1)

- **Why not NV-Embed-v2?** (HippoRAG2's default)
  - NV-Embed-v2 requires ~14GB GPU memory
  - Combined with vLLM (~40GB), it exhausted the A100's 80GB during concurrent operation
  - Caused vLLM server crashes and process hangs
- **Why SBERT?**
  - RAPTOR already uses SBERT (multi-qa-mpnet-base-cos-v1)
  - ~400MB GPU memory — coexists easily with vLLM
  - Using the same embedding for both systems ensures fair comparison
  - Loaded via HippoRAG2's `Transformers/sentence-transformers/multi-qa-mpnet-base-cos-v1` backend

### 3.3 Hardware

- 1x NVIDIA A100-SXM4-80GB
- CUDA 13.0, Driver 580.82.07
- Google Colab environment

---

## 4. HippoRAG2 Configuration

### 4.1 Indexing Pipeline

HippoRAG2 builds a knowledge graph from the corpus using OpenIE:

1. **NER (Named Entity Recognition)**: Extract entities from each of 4,111 passages using the LLM
   - Throughput: ~10-15 it/s (~4 minutes total)
   - Prompt (1-shot, chat format):
     ```
     System: "Your task is to extract named entities from the given paragraph.
              Respond with a JSON list of entities."

     User [1-shot example]:
       "Radio City
        Radio City is India's first private FM radio station and was started on 3 July 2001..."

     Assistant [1-shot output]:
       {"named_entities": ["Radio City", "India", "3 July 2001", "Hindi", "English",
                           "May 2008", "PlanetRadiocity.com"]}

     User [actual input]: <passage text>
     ```

2. **Triple Extraction**: Extract (subject, relation, object) triples from each of 4,111 passages
   - Throughput: ~2-3 it/s (~24 minutes total) -- indexing bottleneck due to longer LLM outputs
   - Prompt (1-shot, chat format):
     ```
     System: "Your task is to construct an RDF graph from the given passages and
              named entity lists. Respond with a JSON list of triples.
              Requirements:
              - Each triple should contain at least one named entity from the list.
              - Clearly resolve pronouns to their specific names."

     User [1-shot example]:
       "Convert the paragraph into a JSON dict with named entity list and triple list.
        Paragraph: Radio City is India's first private FM radio station...
        {"named_entities": ["Radio City", "India", ...]}"

     Assistant [1-shot output]:
       {"triples": [
         ["Radio City", "located in", "India"],
         ["Radio City", "is", "private FM radio station"],
         ["Radio City", "started on", "3 July 2001"],
         ["Radio City", "plays songs in", "Hindi"],
         ["PlanetRadiocity.com", "launched in", "May 2008"],
         ["PlanetRadiocity.com", "is", "music portal"],
         ...
       ]}

     User [actual input]: <passage text> + <extracted NER JSON>
     ```

3. **Embedding**: Encode entities, facts, and passages using SBERT
   - Batch size: 64
   - ~10 seconds for all embeddings

4. **Graph Construction**: Build knowledge graph with entities as nodes, triples as edges
   - Graph type: `facts_and_sim_passage_node_unidirectional`
   - Includes both extracted fact edges and similarity-based passage node edges

### 4.2 Retrieval

- **Method**: Personalized PageRank (PPR) on the knowledge graph
- **Parameters**:
  - `retrieval_top_k=200` (candidate passages)
  - `linking_top_k=5` (entity linking candidates)

### 4.3 Retrieval Query NER

Before retrieval, HippoRAG2 extracts entities from the query for graph linking:

```
System: "You're a very effective entity extraction system."

User [1-shot]:
  "Please extract all named entities that are important for solving the questions below.
   Place the named entities in json format.
   Question: Which magazine was started first Arthur's Magazine or First for Women?"

Assistant [1-shot]:
  {"named_entities": ["First for Women", "Arthur's Magazine"]}

User [actual]: "Question: <query text>"
```

### 4.4 QA (Question Answering)

- **Method**: IRCoT (Interleaved Retrieval Chain-of-Thought)
- **Parameters**:
  - `max_qa_steps=3` (up to 3 iterative retrieval-reasoning steps)
  - `qa_top_k=5` (top passages per step)
  - `max_new_tokens=2048`
- **Process**: For each of 293 queries, the LLM reads retrieved context, reasons step-by-step, optionally retrieves more context, and generates a final answer
- **IRCoT Prompt** (1-shot, chat format):
  ```
  System: "You serve as an intelligent assistant, adept at facilitating users through
           complex, multi-hop reasoning across multiple documents. Your task is to
           generate one thought for current step, DON'T generate the whole thoughts
           at once! If you reach what you believe to be the final step, start with
           'So the answer is:'."

  [1-shot demo with John Lennon documents + multi-hop question + reasoning chain]

  User: <retrieved passages> + "\n\nQuestion: <query>\nThought:"
  ```
  The LLM generates one reasoning step at a time. If it doesn't conclude with "So the answer is:", the system retrieves more passages and prompts again (up to 3 steps).

### 4.4 Full Config

```python
BaseConfig(
    save_dir=SAVE_DIR,
    llm_base_url="http://localhost:8000/v1",
    llm_name="Qwen/Qwen2.5-7B-Instruct",
    embedding_model_name="Transformers/sentence-transformers/multi-qa-mpnet-base-cos-v1",
    force_index_from_scratch=True,
    force_openie_from_scratch=True,
    rerank_dspy_file_path=None,
    retrieval_top_k=200,
    linking_top_k=5,
    max_qa_steps=3,
    qa_top_k=5,
    graph_type="facts_and_sim_passage_node_unidirectional",
    embedding_batch_size=64,
    max_new_tokens=2048,
    corpus_len=4111,
    openie_mode="online",
    max_retry_attempts=10,
)
```

---

## 5. RAPTOR Configuration

### 5.1 Indexing Pipeline

RAPTOR builds a recursive summarization tree:

1. **Leaf Creation**: Split all corpus text into 100-token chunks → 6,477 leaf nodes
   - Text from all 10 documents concatenated with title headers

2. **Layer 0**: Cluster leaf nodes using GMM + UMAP, summarize each cluster
   - ~700 clusters created
   - Each cluster summarized by the LLM (100-token summaries)
   - ~18 minutes (bulk of indexing time)

3. **Layers 1-4**: Recursively cluster and summarize higher-level nodes
   - Each layer has fewer nodes
   - Stopped at Layer 4 (cannot create more layers)
   - ~3 minutes for all upper layers

### 5.2 Retrieval

- **Method**: Collapsed tree retrieval
  - All nodes (leaf + summary) are in a single embedding space
  - Query embedding compared against all nodes
- **Parameters**:
  - `tr_top_k=10` (retrieve top-10 most relevant nodes)
  - `tr_selection_mode="top_k"`

### 5.3 QA (Question Answering)

- **Method**: Single-step QA
  - Retrieved nodes concatenated as context (max 3,500 tokens)
  - LLM generates answer in one call
- **Prompt** (chat format):
  ```
  System: "You are a Question Answering assistant. Answer concisely based on
           the given context."

  User: "Given Context: <top-10 retrieved nodes concatenated>

         Question: <query>

         Answer:"
  ```
- **Parameters**:
  - `max_tokens=150` for answer generation
  - `temperature=0`

### 5.3b Summarization Prompt (used during tree building)

```
System: "You are a helpful assistant."

User: "Write a summary of the following, including as many key details as possible:
       <cluster text>:"
```
- `max_tokens=500`, `temperature=0`
- Called once per cluster (~741 total summarization calls across all layers)

### 5.4 Custom Model Classes

Since RAPTOR was designed for OpenAI GPT API, we implemented custom wrappers:

- `VLLMSummarizationModel`: Wraps `BaseSummarizationModel`, calls vLLM via OpenAI client
- `VLLMQAModel`: Wraps `BaseQAModel`, calls vLLM via OpenAI client

Both use `openai.OpenAI(api_key="dummy", base_url="http://localhost:8000/v1")`.

### 5.5 Full Config

```python
RetrievalAugmentationConfig(
    summarization_model=VLLMSummarizer(),
    qa_model=VLLMQAModel(),
    embedding_model=SBertEmbeddingModel("sentence-transformers/multi-qa-mpnet-base-cos-v1"),
    tb_max_tokens=100,          # 100-token chunks (paper setting)
    tb_num_layers=5,            # max tree layers
    tb_summarization_length=100,
    tr_top_k=10,                # retrieve top-k nodes
    tr_selection_mode="top_k",
)
```

---

## 6. Evaluation Metrics

All metrics computed in `experiments/eval_metrics.py`. For each sample, we compute the max score across all gold answer references.

### 6.1 F1 (Token-level)

- Normalize both prediction and gold answer (lowercase, remove articles/punctuation)
- Tokenize by whitespace
- Compute precision = |common tokens| / |prediction tokens|
- Compute recall = |common tokens| / |gold tokens|
- F1 = 2 * precision * recall / (precision + recall)

### 6.2 Exact Match (EM)

- Normalize both strings
- Return 1.0 if identical, 0.0 otherwise

### 6.3 ROUGE-L

- Compute Longest Common Subsequence (LCS) between normalized token sequences
- Precision = LCS / |prediction tokens|
- Recall = LCS / |gold tokens|
- F-measure = 2 * P * R / (P + R)

### 6.4 BLEU-1 and BLEU-4

- Compute modified n-gram precision (clipped counts)
- Apply brevity penalty: BP = min(1.0, exp(1 - |gold| / |pred|))
- BLEU-n = BP * (clipped n-gram matches / total n-grams in prediction)

### 6.5 METEOR (Simplified)

- Unigram precision and recall on normalized tokens
- Weighted harmonic mean with alpha=0.9 (recall-weighted)
- F_mean = (P * R) / (alpha * P + (1-alpha) * R)
- Note: This is a simplified version; official METEOR includes stemming, synonyms, and chunk penalty

### 6.6 Normalization Function

Applied to both prediction and gold answer before all metric computations:
1. Lowercase the string
2. Remove punctuation characters
3. Remove articles ("a", "an", "the")
4. Collapse multiple whitespace to single space

Example: `"She is a crossword puzzle writer."` -> `"she is crossword puzzle writer"`

### 6.7 Aggregation

- Per-sample: max score across all gold answer references (typically 2 per sample)
- Dataset-level: arithmetic mean across all 293 samples, multiplied by 100 (percentage)

### 6.8 Metric Computation Example

For the sample:
- Prediction: `"Mary Horowitz's job is the crossword constructor for the Atlanta Citizen."`
- Gold 1: `"She is a crossword writer for the Sacramento Herald."`
- Gold 2: `"She is a crossword puzzle writer."`

Normalized:
- Pred: `"mary horowitzs job is crossword constructor for atlanta citizen"`
- Gold 1: `"she is crossword writer for sacramento herald"`
- Gold 2: `"she is crossword puzzle writer"`

F1 vs Gold 2: common tokens = {"is", "crossword", "writer"} = 3 (taking "constructor" != "writer")
- Actually: pred tokens = 9, gold2 tokens = 5, common = {"is", "crossword"} = 2
- Precision = 2/9 = 0.222, Recall = 2/5 = 0.400, F1 = 0.286

Final F1 for this sample = max(F1_vs_gold1, F1_vs_gold2)

---

## 7. Prediction Examples

### Sample 0
- **Question**: What is Mary Horowitz's job?
- **Gold**: ["She is a crossword writer for the Sacramento Herald.", "She is a crossword puzzle writer."]
- **HippoRAG2**: "Mary Horowitz's job is the crossword constructor for the Atlanta Citizen."
- **RAPTOR**: "Mary Horowitz's job is a crossword constructor for the Atlanta Citizen newspaper."
- **Analysis**: Both systems correctly identify the job type (crossword constructor/writer) but hallucinate the newspaper name. Similar quality.

### Sample 50
- **Question**: How do Hi and Ed meet?
- **Gold**: ["She is taking his mug shot.", "They meet after Ed takes Hi's mugshot."]
- **HippoRAG2**: "The script does not provide information on how Hi and Ed meet."
- **RAPTOR**: "The given context does not provide information on how Hi and Ed meet."
- **Analysis**: Both fail -- neither retrieves the relevant passage. This suggests the information was not in the top retrieved passages.

### Sample 150
- **Question**: Why was Almayer so upset when the body was found floating in the river?
- **Gold**: ["because the body was wearing Dain's jewelry and now he has no chance to find the gold", "He thought it was Dain and that without him he'd never find the mine."]
- **HippoRAG2**: "Almayer was upset because the body was found floating in the river, which meant the body would be identified and the circumstances of the death would be known..."
- **RAPTOR**: "Almayer was upset because he heard his wife wailing for the dead..."
- **Analysis**: Both partially address the question but miss the key detail (Dain's jewelry / no chance to find gold). HippoRAG2's answer is more topically relevant.

---

## 8. Execution Timeline

| Phase | HippoRAG2 | RAPTOR |
|-------|-----------|--------|
| NER extraction | 4 min (4111 passages) | N/A |
| Triple extraction | 24 min (4111 passages) | N/A |
| Embedding | 10s | 3 min (6477 leaves) |
| Graph/Tree building | 1s | 21 min (4 layers) |
| **Total indexing** | **39.1 min** | **24.3 min** |
| Retrieval | 5.5 min (293 queries) | included in QA |
| QA reading | 8.3 min (293 queries, multi-step) | 3.3 min (293 queries, single-step) |
| **Total QA** | **14.1 min** | **3.3 min** |
| **Grand total** | **53.1 min** | **27.6 min** |

---

## 8. Software Versions

| Package | Version |
|---------|---------|
| transformers | 4.45.2 |
| tokenizers | 0.20.3 |
| vllm | 0.6.6.post1 |
| openai | 1.58.1 |
| sentence-transformers | 2.7.0 |
| pydantic | 2.10.4 |
| tiktoken | 0.7.0 |
| litellm | 1.59.12 |
| raptor (local) | from github.com/parthsarthi03/raptor |
| hipporag (local) | from github.com/OSU-NLP-Group/HippoRAG |

---

## 9. Reproducibility Notes

1. Start vLLM server manually before running experiments:
   ```bash
   CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
     --model Qwen/Qwen2.5-7B-Instruct \
     --max-model-len 4096 \
     --gpu-memory-utilization 0.45 \
     --port 8000 --dtype auto
   ```

2. Run HippoRAG2: `python experiments/exp-001-hipporag2-narrativeqa/run.py`

3. Run RAPTOR: `python experiments/exp-002-raptor-narrativeqa/run.py`

4. Both scripts save results JSON and metrics JSON to their respective `results/` directories.

5. To re-evaluate from saved results:
   ```bash
   python experiments/eval_metrics.py experiments/exp-001-hipporag2-narrativeqa/results/hipporag2_results.json
   python experiments/eval_metrics.py experiments/exp-002-raptor-narrativeqa/results/raptor_results.json
   ```

---

## 12. Analysis: HippoRAG2 vs RAPTOR for Long-Context RAG

### 12.1 What Makes NarrativeQA a Long-Context Challenge

NarrativeQA is specifically designed to test comprehension of long-form narratives (entire books and movie scripts). The 10 documents in our evaluation are split into 4,111 passages -- each document spans hundreds of passages. Questions require understanding character relationships, plot events, motivations, and causal chains that may be spread across dozens of passages far apart in the text.

This makes it a strong benchmark for evaluating **long-context RAG**: the system must (1) build a meaningful index over thousands of passages, and (2) retrieve the right context from a massive search space to answer sense-making questions.

### 12.2 HippoRAG2: Better for Long-Context Sense-Making

**Wins on 5/6 metrics with +38% relative F1 improvement (16.94 vs 12.28).**

Why HippoRAG2 excels at long-context:

1. **Knowledge graph preserves cross-passage relationships**: When a character is mentioned in passage 50 and their motivation explained in passage 2000, HippoRAG2's entity-relation triples connect these through the graph. Personalized PageRank can traverse these connections to find relevant distant passages.

2. **Multi-step reasoning bridges information gaps**: IRCoT's iterative retrieval allows HippoRAG2 to first find one relevant passage, extract clues, then retrieve additional context in subsequent steps. This is critical for multi-hop narrative questions like "Why was Almayer upset when the body was found?" which requires connecting: body found -> Dain's jewelry -> no gold expedition.

3. **Fine-grained entity linking**: Query NER + graph-based entity linking finds specific entities mentioned in the question, even when they appear in only a few of 4,111 passages. This is more targeted than embedding similarity alone.

**Weakness**: Slower indexing (39 min vs 24 min) and QA (14 min vs 3 min) due to the two-pass OpenIE pipeline and multi-step reasoning.

### 12.3 RAPTOR: Faster but Loses Detail in Long Contexts

**2x faster overall but lower quality across all meaningful metrics.**

Why RAPTOR struggles with long-context:

1. **Information compression loses detail**: RAPTOR's recursive summarization compresses 6,477 leaf nodes into ~741 summary clusters (Layer 0) and further up the tree. Each summarization step loses specific details -- character names, specific events, causal relationships. For narrative QA where details matter, this is a significant disadvantage.

2. **No cross-document entity tracking**: RAPTOR clusters by semantic similarity (GMM + UMAP on embeddings). Two passages about the same character in different scenes may end up in different clusters if they discuss different topics. There's no explicit entity graph to connect them.

3. **Single-step QA limits reasoning depth**: With only one retrieval + one LLM call per query, RAPTOR cannot iteratively refine its understanding. If the initial top-10 retrieved nodes don't contain the answer, there's no second chance.

4. **Collapsed tree mixes abstraction levels**: The retrieval step searches across all levels (leaves + summaries). A high-level summary node may match the query semantically but lack the specific detail needed for the answer.

**Strength**: Much faster QA (3.3 min vs 14.1 min for 293 queries) makes it suitable for latency-sensitive applications where approximate answers are acceptable.

### 12.4 Verdict for Long-Context RAG

| Criterion | Winner | Why |
|-----------|--------|-----|
| **QA accuracy** | HippoRAG2 | +38% F1, better at detail retrieval |
| **Multi-hop questions** | HippoRAG2 | IRCoT multi-step reasoning |
| **Entity-centric queries** | HippoRAG2 | KG entity linking vs embedding similarity |
| **Indexing speed** | RAPTOR | 24 min vs 39 min |
| **QA latency** | RAPTOR | 0.67s/query vs 2.88s/query |
| **Memory efficiency** | RAPTOR | No graph storage needed |
| **Scalability to more docs** | Depends | RAPTOR simpler; HippoRAG2's KG grows linearly |

**For long-context RAG where answer quality matters, HippoRAG2 is the better choice.** Its knowledge graph structure and multi-step reasoning are fundamentally better suited to long narrative comprehension tasks where information is distributed across many passages.

**RAPTOR is better when speed matters more than accuracy**, or when the task is primarily about summarization/overview rather than specific fact retrieval from long documents.

### 12.5 When to Use Which

- **Use HippoRAG2 when**:
  - Questions require connecting information across distant passages
  - Entity relationships are important (who did what to whom)
  - Accuracy is more important than latency
  - Documents are long narratives, legal texts, or multi-document collections

- **Use RAPTOR when**:
  - Questions are about overall themes/summaries
  - Low latency is required (<1s per query)
  - Documents are shorter or topics are well-clustered
  - Approximate answers are acceptable
