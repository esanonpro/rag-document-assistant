# Reproduce the evaluation

The benchmark is reviewed in [BENCHMARK_REVIEW.md](BENCHMARK_REVIEW.md).
It contains 12 questions covering five of the 11 supplied PDFs. All 11 PDFs
participate in retrieval. Each reference has a document filename and exact PDF
pages. PDFs are not redistributed; use the same local copies and compare their
SHA-256 hashes with the run manifest in `results.json`.

## Environment and local model

Use Python 3.12 on Linux and install the measured CPU environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r evaluation/requirements.lock.txt
pip check
```

RAGAS is pinned to **0.4.3**. The script uses its modern
`ragas.metrics.collections` API (`ContextRecall`, `Faithfulness`,
`AnswerRelevancy`). The previous import of `ContextRecall` from `ragas.metrics`
is incompatible with this API. `langchain-community==0.3.31` also preserves
modules imported by RAGAS; the newest community package removes one of these
modules. Maintained Hugging Face and Ollama integrations replace deprecated
LangChain integration imports.

Install Ollama **0.35.0**. The measured model is
**Mistral-7B-Instruct-v0.3, Q4_K_M**, imported under `mistral-eval-v03`.
Its pinned download revision and SHA-256 are in [model_provenance.json](model_provenance.json).
The ordinary production default remains `mistral` via Ollama.
The registry download was unavailable in Work, so the evaluation used the
documented GGUF import instead:

```bash
curl -fL 'https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/61fd4167fff3ab01ee1cfe0da183fa27a944db48/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf' \
  -o evaluation/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf
echo '1270d22c0fbb3d092fb725d4d96c457b7b687a5f5a715abe1e818da303e562b6  evaluation/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf' | sha256sum -c -
LLAMA_ARG_CACHE_RAM=0 OLLAMA_NUM_PARALLEL=1 OLLAMA_CONTEXT_LENGTH=8192 ollama serve
# In a second terminal:
ollama create mistral-eval-v03 -f evaluation/Modelfile
```

The Modelfile reduces the internal prefill batch to 128 for the Work CPU memory
limit. This does not change chunking or retrieval. Use a sufficiently sized
machine and avoid concurrent builds during inference.
The RAM prompt cache is disabled to prevent growth across independent requests.

## Run

Place all 11 PDFs in `data/`, then run from the repository root:

```bash
RAGAS_DO_NOT_TRACK=true python -m evaluation.evaluate_rag \
  --corpus data \
  --rag-model mistral-eval-v03 \
  --judge-model mistral-eval-v03 \
  --embedding-revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
  --output evaluation/rerun
```

Use a new output directory for each run. Existing published results are never
silently overwritten. The full package lock, corpus hashes, benchmark hash,
model digest, Ollama template and evaluation-source hashes identify the run.
If generation is interrupted, `--resume` verifies the benchmark, corpus, package
versions, model configuration and actual retrieved texts before reusing completed
responses. Resume history and server settings are recorded. In the published
run, the first five responses used the default RAM prompt cache; later generation
and all judging disable it. Model weights, prompts and decoding settings are
identical across the resume.
The script builds a **fresh** FAISS index and passes the recorded top-five
contexts directly to the generation chain, avoiding a second retrieval and
preventing reuse of a stale index.

Pipeline: `PyPDFLoader → RecursiveCharacterTextSplitter (1000/200) →
all-MiniLM-L6-v2 → FAISS IndexFlatL2 → top-k=5 → Mistral/Ollama`.
Embeddings are not normalized. Generation uses temperature 0.2, seed 42,
8,192 context tokens, a 2,048-token output ceiling and eight CPU threads.
Other sampler settings are recorded in `results.json`.

## Judge and scoring

The **RAG LLM** generates answers. The **judge LLM** performs separate calls
at temperature 0 with seed 42, structured JSON decoding and a 4,096-token
output ceiling. These roles share the same Mistral weights in this run, an
explicit source of correlated evaluation error. The judge is **not ground
truth**. The reference answers, separately checked against PDFs, remain the
source annotations.

The local adapter passes unmodified RAGAS prompts and Pydantic output schemas
to Ollama. RAGAS itself decomposes claims, checks support, regenerates questions
and computes scores; the adapter does not assign scores. Full judge requests
and raw responses are preserved in `judge_trace.jsonl`.

- **Context Recall** checks which reference-answer claims are present in the
  retrieved context. This implementation also uses the judge.
- **Faithfulness** checks the generated answer's claims against that context.
- **Answer Relevancy** regenerates three questions from the answer and measures
  their cosine similarity to the original question using MiniLM. It is a
  relevance proxy, not a correctness or answer-accuracy measure.

Each reported value is the unweighted mean of all 12 questions. Failed,
truncated or nonfinite judge outputs stop publication; no rows are silently
dropped and no replacement scores are supplied.

The benchmark is small and exploratory. Questions and references are French;
most source text and RAGAS default judge prompts are English. MiniLM is not
multilingual, so cross-language reformulations can reduce Answer Relevancy.
Single-run scores do not establish absence of hallucination or generalize to
all scientific corpora.
