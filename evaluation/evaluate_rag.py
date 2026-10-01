"""Evaluate the RAG pipeline with three complementary RAGAS metrics.

The evaluation dataset is intentionally versioned separately from the public demo
corpus. Fill evaluation/eval_dataset.jsonl with reviewed reference examples before
publishing scores.

Required environment variables for judge-based metrics depend on the LLM provider
configured for RAGAS.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from ragas import EvaluationDataset, evaluate
from ragas.metrics import ContextRecall, Faithfulness, ResponseRelevancy

from app.rag_pipeline import build_or_load_vectorstore, create_rag_chain, VECTORSTORE_PATH

DATASET_PATH = Path("evaluation/eval_dataset.jsonl")
RESULTS_PATH = Path("evaluation/results.json")


def load_examples(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Create reviewed reference examples before evaluation."
        )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Evaluation dataset is empty.")
    return rows


def build_eval_rows(examples: list[dict]) -> list[dict]:
    vectorstore = build_or_load_vectorstore(VECTORSTORE_PATH)
    retriever, rag_chain = create_rag_chain(vectorstore)
    rows = []

    for item in examples:
        question = item["question"]
        docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]
        answer = rag_chain.invoke(question)
        rows.append(
            {
                "user_input": question,
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": item["reference_answer"],
            }
        )
    return rows


def main() -> None:
    examples = load_examples(DATASET_PATH)
    dataset = EvaluationDataset.from_list(build_eval_rows(examples))
    result = evaluate(
        dataset=dataset,
        metrics=[ContextRecall(), Faithfulness(), ResponseRelevancy()],
    )

    frame = result.to_pandas()
    metric_columns = [
        col for col in frame.columns
        if col in {"context_recall", "faithfulness", "answer_relevancy", "response_relevancy"}
    ]
    summary = {col: float(frame[col].mean()) for col in metric_columns}
    payload = {
        "n_questions": len(frame),
        "metrics": summary,
        "note": "Publish only after the reference dataset has been manually reviewed.",
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()
