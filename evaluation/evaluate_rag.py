"""Reproducible RAG evaluation using RAGAS 0.4.3 collections.

Run from the repository root: python -m evaluation.evaluate_rag --corpus data
An Ollama server and explicitly named local models must already be available.
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import httpx
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.metrics.collections import ContextRecall, Faithfulness, AnswerRelevancy
from app.rag_pipeline import load_pdfs_from_folder, split_documents, create_llm, create_answer_chain, format_docs
from evaluation.adapters import OllamaJudge, LocalEmbeddings

METRICS = ('context_recall', 'faithfulness', 'answer_relevancy')

def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def load_examples(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Benchmark must be nonempty with unique IDs')
    for row in rows:
        if not row.get('question') or not row.get('reference_answer') or not row.get('sources'):
            raise ValueError('Every entry needs a question, reference answer and verified sources')
    return rows

def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)

def canonical_config(value):
    """Ollama prints identical parameter maps in nondeterministic line order."""
    if isinstance(value, dict):
        return {k: sorted(v.splitlines()) if k == 'parameters' and isinstance(v, str) else canonical_config(v) for k, v in value.items()}
    if isinstance(value, list):
        return [canonical_config(v) for v in value]
    return value

async def main(args):
    started = datetime.now(timezone.utc).isoformat()
    examples = load_examples(args.dataset)
    corpus = sorted(Path(args.corpus).glob('*.pdf'))
    if not corpus:
        raise ValueError('No PDF corpus supplied')
    manifest = [{'document': p.name, 'sha256': sha256(p)} for p in corpus]
    docs = load_pdfs_from_folder(Path(args.corpus))
    page_counts = {p.name: sum(Path(d.metadata['source']).name == p.name for d in docs) for p in corpus}
    for row in examples:
        for source in row['sources']:
            if source['document'] not in page_counts or any(p < 1 or p > page_counts[source['document']] for p in source['pdf_pages']):
                raise ValueError('Reference source missing or invalid page: ' + row['id'])
    packages = {p: importlib.metadata.version(p) for p in ['ragas','langchain-community','langchain-core','langchain-ollama','langchain-huggingface','langchain-text-splitters','sentence-transformers','transformers','torch','faiss-cpu','pypdf','fonttools']}
    async with httpx.AsyncClient(trust_env=False, timeout=120) as client:
        tags = (await client.get(args.ollama_url + '/api/tags')).json()['models']
        models = {}
        for role, name in [('rag', args.rag_model), ('judge', args.judge_model)]:
            tag = next((t for t in tags if t['name'] in [name, name + ':latest']), None)
            if tag is None:
                raise ValueError('Ollama model unavailable: ' + name)
            show = await client.post(args.ollama_url + '/api/show', json={'model': name})
            show.raise_for_status()
            models[role] = {'name': name, 'digest': tag['digest'], 'details': tag.get('details'), 'parameters': show.json().get('parameters'), 'template': show.json().get('template')}
        version = (await client.get(args.ollama_url + '/api/version')).json()['version']
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / 'results.json'
    if results_path.exists():
        raise FileExistsError('Choose a clean output directory; never overwrite measured results implicitly')
    embedding = HuggingFaceEmbeddings(model_name=args.embedding_model, model_kwargs={'device': 'cpu', 'revision': args.embedding_revision}, encode_kwargs={'normalize_embeddings': False})
    # Always build a fresh index from the hashed corpus. No existing pickle is loaded.
    chunks = split_documents(docs, chunk_size=1000, chunk_overlap=200)
    vectorstore = FAISS.from_documents(chunks, embedding)
    retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 5})
    config = {'loader': 'PyPDFLoader', 'chunk_size': 1000, 'chunk_overlap': 200, 'separators': ['\n\n','\n','. ',' ',''], 'top_k': 5, 'retrieval': 'FAISS IndexFlatL2, unnormalized embeddings, similarity', 'embedding_model': args.embedding_model, 'embedding_revision': args.embedding_revision, 'embedding_normalize': False, 'rag_llm': models['rag'], 'rag_temperature': 0.2, 'rag_seed': 42, 'num_ctx': 8192, 'rag_num_predict': 2048, 'n_documents': len(corpus), 'n_pages': len(docs), 'n_chunks': len(chunks)}
    config['rag_prompt_sha256'] = sha256(Path(__file__).parents[1] / 'app/rag_pipeline.py')
    config['rag_options'] = {'temperature': 0.2, 'seed': 42, 'num_ctx': 8192, 'num_predict': 2048, 'num_thread': 8, 'top_p': 0.9, 'top_k': 40, 'repeat_penalty': 1.1}
    run_info = {'pipeline': config, 'judge': models['judge'], 'packages': packages, 'corpus': manifest, 'benchmark_sha256': sha256(args.dataset), 'started_at': started}
    responses_path = out / 'responses.json'
    rows = []
    resume_info = []
    if args.resume and responses_path.exists():
        previous = json.loads((out / 'run_config.json').read_text())
        for key in ('pipeline', 'judge', 'packages', 'corpus', 'benchmark_sha256'):
            if canonical_config(previous[key]) != canonical_config(run_info[key]):
                raise ValueError('Cannot resume a different evaluation: ' + key)
        rows = json.loads(responses_path.read_text())
        if len(rows) > len(examples):
            raise ValueError('Checkpoint has too many rows')
        for saved, example in zip(rows, examples):
            if any(saved.get(key) != value for key, value in example.items()):
                raise ValueError('Checkpoint benchmark entry changed: ' + example['id'])
            if not saved.get('response') or saved['retrieved_contexts'] != [d.page_content for d in retriever.invoke(example['question'])]:
                raise ValueError('Checkpoint response missing or contexts changed: ' + example['id'])
        resume_info = previous.get('resumes', []) + [{'previous_started_at': previous['started_at'], 'completed_responses': len(rows), 'previous_server_environment': previous.get('server_environment', {'LLAMA_ARG_CACHE_RAM': 'default (8192 MiB in Ollama 0.35.0)'})}]
        print('resumed', len(rows), 'verified responses', flush=True)
    elif responses_path.exists():
        raise FileExistsError('Partial responses already exist; use --resume or a clean output directory')
    run_info['resumes'] = resume_info
    run_info['server_environment'] = {key: os.getenv(key) for key in ('OLLAMA_NUM_PARALLEL', 'OLLAMA_CONTEXT_LENGTH', 'LLAMA_ARG_CACHE_RAM', 'OMP_NUM_THREADS')}
    atomic_json(out / 'run_config.json', run_info)
    llm = create_llm(args.rag_model, base_url=args.ollama_url, **config['rag_options'])
    chain = create_answer_chain(llm)
    for row in examples[len(rows):]:
        retrieved = retriever.invoke(row['question'])
        # The same passages are used once for generation and recorded for scoring.
        answer = chain.invoke({'question': row['question'], 'context': format_docs(retrieved)})
        record = {**row, 'response': answer, 'retrieved_contexts': [d.page_content for d in retrieved], 'retrieved_sources': [{'document': Path(d.metadata['source']).name, 'pdf_page': d.metadata['page'] + 1} for d in retrieved]}
        rows.append(record)
        atomic_json(responses_path, rows)
        print('generated', row['id'], flush=True)
    judge_options = {'temperature': 0, 'seed': 42, 'num_ctx': 8192, 'num_predict': 4096, 'top_p': 1, 'top_k': 40, 'repeat_penalty': 1.1, 'num_thread': 8}
    judge = OllamaJudge(args.judge_model, args.ollama_url, judge_options, out / 'judge_trace.jsonl')
    metrics = [ContextRecall(llm=judge), Faithfulness(llm=judge), AnswerRelevancy(llm=judge, embeddings=LocalEmbeddings(embedding), strictness=3)]
    scores = []
    for row in rows:
        inputs = [{'user_input': row['question'], 'retrieved_contexts': row['retrieved_contexts'], 'reference': row['reference_answer']}, {'user_input': row['question'], 'response': row['response'], 'retrieved_contexts': row['retrieved_contexts']}, {'user_input': row['question'], 'response': row['response']}]
        scored = {'id': row['id']}
        for metric, inp in zip(metrics, inputs):
            value = float((await metric.ascore(**inp)).value)
            if not math.isfinite(value):
                raise ValueError('Undefined metric for ' + row['id'] + ': ' + metric.name + '; aggregate not published')
            scored[metric.name] = value
            print('scored', row['id'], metric.name, value, flush=True)
        scores.append(scored)
        atomic_json(out / 'per_question.json', scores)
    summary = {key: sum(s[key] for s in scores) / len(scores) for key in METRICS}
    payload = {'status': 'completed', 'n_questions': len(scores), 'metrics': summary, 'aggregation': 'unweighted arithmetic mean across all benchmark questions; no failed rows omitted', 'pipeline': config, 'judge': {**models['judge'], 'provider': 'Ollama local', 'adapter': 'Ollama structured JSON schema / RAGAS collections', 'options': judge_options, 'prompts': 'Unmodified RAGAS 0.4.3 default English prompts; full request/output trace in judge_trace.jsonl', 'is_ground_truth': False, 'same_model_as_generator': args.rag_model == args.judge_model}, 'answer_relevancy': {'strictness': 3, 'embedding_model': args.embedding_model, 'embedding_revision': args.embedding_revision}, 'evaluation': {'started_at_utc': started, 'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'ollama_version': version, 'packages': packages, 'benchmark_sha256': sha256(args.dataset), 'corpus': manifest, 'evaluator_sha256': sha256(__file__), 'adapters_sha256': sha256(Path(__file__).with_name('adapters.py')), 'server_environment': run_info['server_environment'], 'resumes': resume_info}, 'artifacts': {'responses': 'responses.json', 'per_question': 'per_question.json', 'judge_trace': 'judge_trace.jsonl'}, 'limitations': ['Small exploratory benchmark, not a population-level performance estimate.', 'Generator and judge may share weights; judge is not ground truth and can misclassify support.', 'French questions and reference answers; English scientific corpus and default RAGAS judge prompts.', 'MiniLM is not multilingual; relevance similarities may penalize cross-language reformulations.', 'Single seeded run; quantized CPU inference and dependency versions affect reproducibility.']}
    atomic_json(results_path, payload)
    print(json.dumps(summary), flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', default='data')
    p.add_argument('--dataset', default='evaluation/eval_dataset.jsonl')
    p.add_argument('--output', default='evaluation')
    p.add_argument('--rag-model', default='mistral')
    p.add_argument('--judge-model', required=True, help='Explicit judge model, never an implicit provider default')
    p.add_argument('--ollama-url', default='http://127.0.0.1:11434')
    p.add_argument('--embedding-model', default='sentence-transformers/all-MiniLM-L6-v2')
    p.add_argument('--embedding-revision', required=True, help='Pinned Hugging Face model commit')
    p.add_argument('--resume', action='store_true', help='Reuse only verified completed responses from an identical benchmark, corpus, pipeline and judge configuration')
    asyncio.run(main(p.parse_args()))
