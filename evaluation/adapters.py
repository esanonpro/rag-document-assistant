"""Local adapters for the RAGAS 0.4 collections API; scores remain RAGAS code."""
import asyncio
import json
from pathlib import Path

import httpx
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.embeddings.base import BaseRagasEmbedding


class OllamaJudge(InstructorBaseRagasLLM):
    def __init__(self, model, base_url, options, trace_path):
        self.model, self.base_url, self.options = model, base_url, options
        self.trace_path = Path(trace_path)

    def generate(self, prompt, response_model):
        return asyncio.run(self.agenerate(prompt, response_model))

    async def agenerate(self, prompt, response_model):
        # Ollama constrains decoding to the unmodified RAGAS output schema.
        payload = {
            "model": self.model, "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "format": response_model.model_json_schema(),
            "options": self.options,
        }
        async with httpx.AsyncClient(timeout=1200, trust_env=False) as client:
            result = await client.post(self.base_url + "/api/chat", json=payload)
            result.raise_for_status()
            raw = result.json()
        with self.trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"schema": response_model.__name__, "prompt": prompt,
                                     "request_options": self.options, "output": raw}, ensure_ascii=False) + "\n")
        if raw.get("done_reason") == "length":
            raise RuntimeError("Judge output truncated; no aggregate will be published")
        return response_model.model_validate_json(raw["message"]["content"])


class LocalEmbeddings(BaseRagasEmbedding):
    def __init__(self, embeddings):
        super().__init__()
        self.embeddings = embeddings

    def embed_text(self, text, **kwargs):
        return self.embeddings.embed_query(text)

    async def aembed_text(self, text, **kwargs):
        return self.embed_text(text)

    def embed_texts(self, texts, **kwargs):
        return self.embeddings.embed_documents(texts)

    async def aembed_texts(self, texts, **kwargs):
        return self.embed_texts(texts)
