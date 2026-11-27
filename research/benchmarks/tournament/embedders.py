"""Pinned, local sentence embedders for the many-option comparison.

These adapters are research dependencies, not part of Laya's runtime API. The pooling
and prefixes follow each model card; the short intent labels are not long passages.
"""
from dataclasses import asdict, dataclass

import numpy as np

from laya.shortlist import cached_embed_fn


@dataclass(frozen=True)
class EmbeddingSpec:
    repo: str
    revision: str
    pooling: str
    dimensions: int
    max_length: int
    query_prefix: str = ""
    option_prefix: str = ""


SPECS = {
    "nomic": EmbeddingSpec("nomic-ai/modernbert-embed-base",
                           "d556a88e332558790b210f7bdbe87da2fa94a8d8", "mean", 768, 512,
                           "search_query: ", "search_document: "),
    "bge": EmbeddingSpec("BAAI/bge-small-en-v1.5",
                         "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a", "cls", 384, 512),
    "minilm": EmbeddingSpec("sentence-transformers/all-MiniLM-L6-v2",
                            "1110a243fdf4706b3f48f1d95db1a4f5529b4d41", "mean", 384, 256),
}


def pool(hidden, mask, method):
    """Pool only real tokens, then normalize in fp32 like the model cards."""
    import torch
    import torch.nn.functional as F

    hidden = hidden.float()
    if method == "cls":
        vectors = hidden[:, 0]
    elif method == "mean":
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        vectors = (hidden * weights).sum(1) / weights.sum(1).clamp(min=1)
    else:
        raise ValueError("unknown pooling method %r" % method)
    if not torch.isfinite(vectors).all() or (vectors.norm(dim=1) == 0).any():
        raise ValueError("embedder returned a non-finite or zero vector")
    return F.normalize(vectors, p=2, dim=1)


class TextEncoder:
    """Encode already-prefixed strings; their positions carry no role information."""

    def __init__(self, spec, device="cpu", batch_size=16, local_files_only=False):
        import torch
        from transformers import AutoModel, AutoTokenizer

        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.spec, self.batch_size = spec, batch_size
        self.device = torch.device(device)
        options = dict(revision=spec.revision, local_files_only=local_files_only,
                       trust_remote_code=False)
        self.tokenizer = AutoTokenizer.from_pretrained(spec.repo, **options)
        self.model = AutoModel.from_pretrained(spec.repo, use_safetensors=True, **options)
        self.model.to(device=self.device, dtype=torch.float32).eval()
        self.encoded_texts = 0
        self.calls = 0

    def __call__(self, texts):
        import torch

        if not texts:
            return np.empty((0, self.spec.dimensions), dtype=np.float32)
        self.calls += 1
        parts = []
        for start in range(0, len(texts), self.batch_size):
            chunk = list(texts[start:start + self.batch_size])
            encoded = self.tokenizer(chunk, padding=True, truncation=True,
                                     max_length=self.spec.max_length, return_tensors="pt")
            # Count truncations separately from the live path in the experiment manifest;
            # doing an unbounded second tokenization here would distort online latency.
            self.encoded_texts += len(chunk)
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                hidden = self.model(**encoded).last_hidden_state
                vectors = pool(hidden, encoded["attention_mask"], self.spec.pooling)
            result = vectors.cpu().numpy()
            if result.shape != (len(chunk), self.spec.dimensions):
                raise ValueError("unexpected embedding shape %r" % (result.shape,))
            parts.append(result)
        return np.concatenate(parts)


class RoleEmbedder:
    """Apply query/option prefixes BEFORE caching can deduplicate or reorder misses."""

    def __init__(self, encoder, spec, maxsize=4096):
        self.encoder, self.spec = encoder, spec
        self.cached = cached_embed_fn(encoder, maxsize=maxsize)

    def __call__(self, texts):
        if not texts:
            return np.empty((0, self.spec.dimensions), dtype=np.float32)
        return self.cached([self.spec.query_prefix + texts[0]] +
                           [self.spec.option_prefix + text for text in texts[1:]])

    def warm_options(self, texts):
        return self.cached([self.spec.option_prefix + text for text in texts])

    def cache_clear(self):
        self.cached.cache_clear()

    def cache_info(self):
        return self.cached.cache_info()

    def identity(self):
        return dict(asdict(self.spec), dtype="float32", device=str(self.encoder.device))


def truncation_summary(encoder, cases, question):
    """Count input lengths offline; never put the extra tokenization in live timings."""
    from laya.common import render_options, serialize_state

    spec = encoder.spec
    queries = [spec.query_prefix + question["instructions"] + "\n" + serialize_state(c["state"]) for c in cases]
    options = [spec.option_prefix + text for text in render_options({"t": "choice", "crit": question["criteria"]})]
    result = {}
    for role, texts in (("queries", queries), ("options", options)):
        lengths = [len(encoder.tokenizer(text, truncation=False)["input_ids"]) for text in texts]
        result[role] = {"texts": len(texts), "max_tokens": max(lengths),
                        "truncated": sum(n > spec.max_length for n in lengths)}
    return result
