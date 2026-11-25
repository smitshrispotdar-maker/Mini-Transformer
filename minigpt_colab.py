"""Shared Mini-GPT architecture, byte-level BPE tokenizer, and Colab checkpoint helpers.

This module is the single source of truth for:
  * MiniGPT_Colab_Training.ipynb (GPU training/export in Colab)
  * app.py (local Streamlit inference)

No pretrained language-model weights are downloaded or used. The Transformer
definition is the same nanoGPT-style model already present in the original app;
only generation options, checkpoint helpers, and a byte-level BPE tokenizer are
added around it.
"""

from __future__ import annotations

import csv
import hashlib
import heapq
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Token constants
# ---------------------------------------------------------------------------

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAD_ID, BOS_ID, EOS_ID, UNK_ID = range(len(SPECIAL_TOKENS))
_SPECIAL_IDS = {PAD_ID, BOS_ID, EOS_ID, UNK_ID}

# ---------------------------------------------------------------------------
# Reproducibility and text utilities
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_text(text: str) -> str:
    """Normalize Unicode/quotes/whitespace while preserving paragraph breaks."""
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
        "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
        "\u2013": "-", "\u2014": "-", "\u2212": "-",
        "\u00a0": " ", "\u200b": "", "\ufeff": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_document(text: str, min_chars: int = 60) -> str:
    """Keep only reasonably sized paragraphs; retain '\n\n' boundaries."""
    text = normalize_text(text)
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) >= min_chars]
    return "\n\n".join(paragraphs)


def _dedupe_key(text: str) -> str:
    normalized = re.sub(r"\W+", " ", text.lower()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def dedupe_documents(documents: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate records using normalized text hashes."""
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    for doc in documents:
        key = _dedupe_key(doc["text"])
        if key in seen:
            continue
        seen.add(key)
        kept.append(doc)
    return kept


def split_documents(
    documents: Sequence[dict[str, Any]],
    val_fraction: float = 0.05,
    test_fraction: float = 0.05,
    seed: int = 1337,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split whole documents so no document leaks across splits."""
    docs = list(documents)
    random.Random(seed).shuffle(docs)
    n = len(docs)
    if n < 3:
        raise ValueError("Need at least 3 clean documents for train/val/test split.")
    n_test = max(1, int(round(n * test_fraction)))
    n_val = max(1, int(round(n * val_fraction)))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError("Dataset too small after split; lower the split fractions.")
    return docs[:n_train], docs[n_train:n_train + n_val], docs[n_train + n_val:]


def document_text(documents: Iterable[dict[str, Any]]) -> str:
    return "\n\n".join(doc["text"] for doc in documents)


# ---------------------------------------------------------------------------
# Programmatic dataset loaders (streaming; no full-corpus download)
# ---------------------------------------------------------------------------

DATASET_CARDS = {
    "tinystories": {
        "path": "roneneldan/TinyStories",
        "name": None,
        "license": "cdla-sharing-1.0",
        "url": "https://huggingface.co/datasets/roneneldan/TinyStories",
    },
    "wikitext103": {
        "path": "Salesforce/wikitext",
        "name": "wikitext-103-v1",
        "license": "see dataset card (WikiText-103 derived from Wikipedia)",
        "url": "https://huggingface.co/datasets/Salesforce/wikitext",
    },
    "fineweb_edu": {
        "path": "HuggingFaceFW/fineweb-edu",
        "name": "sample-10BT",
        "license": "odc-by",
        "url": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
    },
}


def _stream_rows(path: str, name: str | None, split: str, limit: int) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - Colab installs this
        raise RuntimeError("Install the `datasets` package before loading Hugging Face data.") from exc

    kwargs: dict[str, Any] = {"path": path, "split": split, "streaming": True}
    if name:
        kwargs["name"] = name
    dataset = load_dataset(**kwargs)
    for index, row in enumerate(dataset):
        if index >= limit:
            break
        yield row


def load_source_documents(source: str, max_docs: int, min_chars: int = 60) -> list[dict[str, Any]]:
    """Load one candidate source with cleaning and source provenance."""
    if source not in DATASET_CARDS:
        raise ValueError(f"Unknown source {source!r}. Choose from {sorted(DATASET_CARDS)}")
    card = DATASET_CARDS[source]
    rows = _stream_rows(card["path"], card["name"], "train", max_docs)
    documents: list[dict[str, Any]] = []
    seen = set()
    for row in rows:
        raw = row.get("text") or row.get("content") or ""
        cleaned = clean_document(str(raw), min_chars=min_chars)
        if not cleaned:
            continue
        key = _dedupe_key(cleaned)
        if key in seen:
            continue
        seen.add(key)
        documents.append({
            "text": cleaned,
            "source": source,
            "source_url": card["url"],
            "license": card["license"],
        })
        if len(documents) >= max_docs:
            break
    return documents


def build_dataset(config: dict[str, dict[str, Any]], seed: int = 1337) -> dict[str, Any]:
    """Build deduplicated train/val/test document splits from enabled sources."""
    all_docs: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    for source, settings in config.items():
        if not settings.get("enabled"):
            continue
        docs = load_source_documents(source, int(settings.get("max_docs", 1000)))
        source_counts[source] = len(docs)
        all_docs.extend(docs)

    all_docs = dedupe_documents(all_docs)
    train_docs, val_docs, test_docs = split_documents(all_docs, seed=seed)

    def stats(docs: Sequence[dict[str, Any]]) -> dict[str, Any]:
        per_source = Counter(doc["source"] for doc in docs)
        return {
            "documents": len(docs),
            "characters": sum(len(doc["text"]) for doc in docs),
            "paragraphs": sum(doc["text"].count("\n\n") + 1 for doc in docs),
            "by_source": dict(per_source),
        }

    return {
        "train": train_docs,
        "val": val_docs,
        "test": test_docs,
        "stats": {
            "requested_by_source": {k: v.get("max_docs") for k, v in config.items() if v.get("enabled")},
            "kept_by_source": source_counts,
            "after_dedupe": len(all_docs),
            "train": stats(train_docs),
            "val": stats(val_docs),
            "test": stats(test_docs),
        },
    }


# ---------------------------------------------------------------------------
# Byte-level BPE tokenizer (trained from scratch on training data only)
# ---------------------------------------------------------------------------


class BPETokenizer:
    """Small byte-level BPE with byte fallback and reserved special tokens.

    Tokens are stored as strings of Latin-1 characters, each representing one
    UTF-8 byte. Unknown Unicode is therefore never unrepresentable: it falls
    back to its UTF-8 byte tokens.
    """

    def __init__(
        self,
        vocab: Sequence[str],
        merges: Sequence[tuple[str, str]],
        config: dict[str, Any] | None = None,
    ) -> None:
        self.vocab = list(vocab)
        self.merges = [(a, b) for a, b in merges]
        self.rank = {pair: i for i, pair in enumerate(self.merges)}
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.config = dict(config or {})
        for special in SPECIAL_TOKENS:
            if special not in self.token_to_id:
                raise ValueError(f"Missing required special token {special!r} in vocabulary.")
        if len(set(self.vocab)) != len(self.vocab):
            raise ValueError("Tokenizer vocabulary contains duplicate tokens.")
        self._encode_chunk.cache_clear()

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int = 4096,
        min_frequency: int = 1,
        seed: int = 1337,
    ) -> "BPETokenizer":
        """Train merges on `text` only. `vocab_size` includes 4 specials + 256 bytes."""
        if vocab_size < 260:
            raise ValueError("vocab_size must be at least 260 (4 specials + 256 byte tokens).")
        seed_everything(seed)
        vocab = list(SPECIAL_TOKENS) + [chr(b) for b in range(256)]
        vocab_set = set(vocab)
        merges: list[tuple[str, str]] = []

        latin = text.encode("utf-8").decode("latin-1")
        chunks = re.findall(r"\s+|\S+", latin)
        word_counts = Counter(chunks)
        word_tokens: dict[str, list[str]] = {word: list(word) for word in word_counts}

        pair_counts: Counter[tuple[str, str]] = Counter()
        pair_words: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
        heap: list[tuple[int, tuple[str, str]]] = []

        def inc(pair: tuple[str, str], word: str) -> None:
            count = pair_counts.get(pair, 0) + 1
            pair_counts[pair] = count
            pair_words[pair].add(word)
            heapq.heappush(heap, (-count, pair))

        def dec(pair: tuple[str, str], word: str) -> None:
            count = pair_counts.get(pair, 0) - 1
            if count <= 0:
                pair_counts.pop(pair, None)
                pair_words.pop(pair, None)
            else:
                pair_counts[pair] = count
                if pair in pair_words:
                    pair_words[pair].discard(word)

        for word, count in word_counts.items():
            tokens = word_tokens[word]
            for pair in zip(tokens, tokens[1:]):
                pair_counts[pair] += count
                pair_words[pair].add(word)
        heap = [(-count, pair) for pair, count in pair_counts.items()]
        heapq.heapify(heap)

        while len(vocab) < vocab_size and heap:
            negative_count, pair = heapq.heappop(heap)
            if pair_counts.get(pair, 0) != -negative_count:
                continue  # stale heap entry
            if pair_counts[pair] < min_frequency:
                break
            left, right = pair
            new_token = left + right
            if new_token in vocab_set:
                pair_counts.pop(pair, None)
                pair_words.pop(pair, None)
                continue

            merges.append(pair)
            vocab.append(new_token)
            vocab_set.add(new_token)

            affected_words = list(pair_words.pop(pair, ()))
            pair_counts.pop(pair, None)
            for word in affected_words:
                old = word_tokens[word]
                new_tokens: list[str] = []
                i = 0
                while i < len(old):
                    if i + 1 < len(old) and (old[i], old[i + 1]) == pair:
                        if i > 0:
                            dec((old[i - 1], left), word)
                        if i + 2 < len(old):
                            dec((right, old[i + 2]), word)
                        new_tokens.append(new_token)
                        i += 2
                    else:
                        new_tokens.append(old[i])
                        i += 1
                word_tokens[word] = new_tokens
                for next_pair in zip(new_tokens, new_tokens[1:]):
                    inc(next_pair, word)

        config = {
            "type": "byte_level_bpe",
            "target_vocab_size": int(vocab_size),
            "actual_vocab_size": len(vocab),
            "special_tokens": list(SPECIAL_TOKENS),
            "byte_fallback": True,
            "trained_characters": len(text),
            "trained_chunks": len(chunks),
            "merge_count": len(merges),
        }
        return cls(vocab, merges, config)

    @lru_cache(maxsize=200_000)
    def _encode_chunk(self, chunk: str) -> tuple[str, ...]:
        tokens = list(chunk)
        while len(tokens) > 1:
            best_rank = None
            best_index = None
            for i in range(len(tokens) - 1):
                rank = self.rank.get((tokens[i], tokens[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_index = rank, i
            if best_index is None:
                break
            tokens[best_index:best_index + 2] = [tokens[best_index] + tokens[best_index + 1]]
        return tuple(tokens)

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        latin = text.encode("utf-8").decode("latin-1")
        ids: list[int] = []
        if add_special:
            ids.append(self.bos_id)
        for chunk in re.findall(r"\s+|\S+", latin):
            for token in self._encode_chunk(chunk):
                ids.append(self.token_to_id[token])
        if add_special:
            ids.append(self.eos_id)
        return ids

    def encode_prompt(self, text: str, max_context_tokens: int | None = None) -> list[int]:
        """Encode a generation prompt as `<bos>` + text, never with a trailing EOS.

        During training every document ends with `<eos>`. If generation starts
        after an EOS, the model behaves as if the document ended and decoding
        stops at that EOS, hiding the real continuation. Use this method for
        inference prompts instead of `encode(text, add_special=True)`.
        """
        body = self.encode(text, add_special=False)
        if max_context_tokens is not None:
            if max_context_tokens < 2:
                raise ValueError("max_context_tokens must be at least 2 (BOS + one token).")
            if len(body) + 1 > max_context_tokens:
                body = body[-(max_context_tokens - 1):]
        return [self.bos_id] + body

    def decode(self, ids: Sequence[int], skip_special: bool = True, stop_at_eos: bool = True) -> str:
        parts: list[str] = []
        for raw_id in ids:
            token_id = int(raw_id)
            if token_id < 0 or token_id >= len(self.vocab):
                continue
            token = self.vocab[token_id]
            if token == "<eos>" and stop_at_eos:
                break
            if token in SPECIAL_TOKENS and skip_special:
                continue
            parts.append(token)
        try:
            return "".join(parts).encode("latin-1").decode("utf-8", errors="replace")
        except UnicodeError:  # pragma: no cover - defensive
            return "".join(parts)

    def unknowns(self, text: str) -> list[str]:
        """Byte fallback means no character is ever unknown."""
        return []

    def roundtrip_test(self, sentences: Sequence[str]) -> list[dict[str, Any]]:
        results = []
        for sentence in sentences:
            ids = self.encode(sentence, add_special=True)
            decoded = self.decode(ids)
            results.append({
                "input": sentence,
                "token_ids": len(ids),
                "roundtrip_ok": decoded == sentence,
                "decoded": decoded,
            })
        return results

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "special_tokens": list(SPECIAL_TOKENS),
            "vocab": self.vocab,
            "merges": [list(pair) for pair in self.merges],
            "config": self.config,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != 1:
            raise ValueError(f"Unsupported tokenizer format: {payload.get('version')!r}")
        merges = [(pair[0], pair[1]) for pair in payload["merges"]]
        return cls(payload["vocab"], merges, payload.get("config"))


# ---------------------------------------------------------------------------
# Custom Transformer (same state dict as the original app, with extra decoding)
# ---------------------------------------------------------------------------


class Head(nn.Module):
    def __init__(self, head_size: int, n_embd: int, block_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, T, _ = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        weights = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        return self.dropout(F.softmax(weights, dim=-1)) @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, head_size: int, n_embd: int, block_size: int, dropout: float):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))


class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        return x + self.ffwd(self.ln2(x))


class MiniGPT(nn.Module):
    """Original nanoGPT-style model; state dict is unchanged from app.py."""

    def __init__(
        self,
        vocab_size: int,
        n_embd: int = 128,
        n_head: int = 4,
        n_layer: int = 4,
        block_size: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("Embedding dimension must be divisible by number of attention heads.")
        self.block_size = int(block_size)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.final_ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError("Input context exceeds block size.")
        x = self.token_embedding_table(idx) + self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )
        logits = self.lm_head(self.final_ln(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(B * T, -1), targets.reshape(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 300,
        temperature: float = 0.8,
        top_k: int = 0,
        top_p: float = 0.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        eos_id: int | None = None,
        stop_at_eos: bool = False,
    ) -> torch.Tensor:
        """Sample continuations. Original 4-argument signature remains compatible."""
        self.eval()
        idx = idx.clone()
        temperature = max(float(temperature), 1e-5)
        for _ in range(max(0, int(max_new_tokens))):
            logits, _ = self(idx[:, -self.block_size:])
            logits = logits[:, -1, :].float() / temperature

            if repetition_penalty and repetition_penalty != 1.0:
                penalty = float(repetition_penalty)
                for row in range(idx.size(0)):
                    previous = idx[row]
                    scores = logits[row]
                    scores[previous] = torch.where(
                        scores[previous] >= 0,
                        scores[previous] / penalty,
                        scores[previous] * penalty,
                    )

            if top_k and top_k > 0:
                k = min(int(top_k), logits.size(-1))
                values, _ = torch.topk(logits, k, dim=-1)
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))

            if top_p and 0.0 < float(top_p) < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cumulative - F.softmax(sorted_logits, dim=-1) > float(top_p)
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                logits = torch.zeros_like(logits).scatter_(-1, sorted_indices, sorted_logits)

            probs = F.softmax(logits, dim=-1)

            if no_repeat_ngram_size and int(no_repeat_ngram_size) > 1:
                n = int(no_repeat_ngram_size)
                probs = probs.clone()
                for row in range(idx.size(0)):
                    history = idx[row].tolist()
                    if len(history) < n - 1:
                        continue
                    prefix = tuple(history[-(n - 1):])
                    banned = {
                        history[i + n - 1]
                        for i in range(len(history) - n + 1)
                        if tuple(history[i:i + n - 1]) == prefix
                    }
                    if banned:
                        banned_tensor = torch.tensor(sorted(banned), device=probs.device)
                        probs[row, banned_tensor] = 0.0
                        total = probs[row].sum()
                        if total > 0:
                            probs[row] = probs[row] / total
                        else:
                            probs[row] = F.softmax(logits[row], dim=-1)

            next_token = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_token), dim=1)
            if stop_at_eos and eos_id is not None and bool((next_token == eos_id).all()):
                break
        return idx


# ---------------------------------------------------------------------------
# Model/checkpoint utilities
# ---------------------------------------------------------------------------

REQUIRED_CONFIG_KEYS = ("vocab_size", "n_embd", "n_head", "n_layer", "block_size", "dropout")


def build_model(config: dict[str, Any]) -> MiniGPT:
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f"config.json is missing required keys: {', '.join(missing)}")
    cfg = dict(config)
    if int(cfg["n_embd"]) % int(cfg["n_head"]) != 0:
        raise ValueError("n_embd must be divisible by n_head.")
    return MiniGPT(
        vocab_size=int(cfg["vocab_size"]),
        n_embd=int(cfg["n_embd"]),
        n_head=int(cfg["n_head"]),
        n_layer=int(cfg["n_layer"]),
        block_size=int(cfg["block_size"]),
        dropout=float(cfg["dropout"]),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def config_fingerprint() -> str:
    """Hash of this file: notebook and app should report the same value."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def save_checkpoint(
    output_dir: str | Path,
    model: MiniGPT,
    tokenizer: BPETokenizer,
    config: dict[str, Any],
    metadata: dict[str, Any],
    metrics_rows: Sequence[dict[str, Any]] | None = None,
    validation_text: str | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "config": config,
            "model_state_dict": model.state_dict(),
        },
        output_dir / "model.pt",
    )
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    tokenizer.save(output_dir / "tokenizer.json")
    if metrics_rows is not None:
        fieldnames = []
        for row in metrics_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with (output_dir / "training_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics_rows)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if validation_text is not None:
        (output_dir / "validation.txt").write_text(validation_text, encoding="utf-8")
    return output_dir


def load_colab_checkpoint(
    checkpoint_dir: str | Path, device: str | torch.device | None = None
) -> tuple[MiniGPT, BPETokenizer, dict[str, Any], dict[str, Any]]:
    """Validate and load model.pt/config.json/tokenizer.json with clear errors."""
    path = Path(checkpoint_dir).expanduser()
    required = ["model.pt", "config.json", "tokenizer.json"]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Checkpoint folder is incomplete. Missing: {', '.join(missing)}. "
            f"Expected folder: {path.resolve()}"
        )

    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    tokenizer = BPETokenizer.load(path / "tokenizer.json")
    if int(config.get("vocab_size", -1)) != tokenizer.vocab_size:
        raise ValueError(
            f"Vocabulary mismatch: config.json says {config.get('vocab_size')} but "
            f"tokenizer.json has {tokenizer.vocab_size} tokens."
        )

    model = build_model(config)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    try:
        payload = torch.load(path / "model.pt", map_location=device, weights_only=True)
    except TypeError:  # older torch
        payload = torch.load(path / "model.pt", map_location=device)
    state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state_dict, dict):
        raise ValueError("model.pt does not contain a state dictionary.")

    expected = model.state_dict()
    missing_keys = sorted(set(expected) - set(state_dict))
    unexpected_keys = sorted(set(state_dict) - set(expected))
    if missing_keys or unexpected_keys:
        detail = []
        if missing_keys:
            detail.append(f"missing={missing_keys[:5]}")
        if unexpected_keys:
            detail.append(f"unexpected={unexpected_keys[:5]}")
        raise ValueError("Checkpoint architecture does not match config.json: " + "; ".join(detail))

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    return model, tokenizer, config, metadata


def history_from_csv(path: str | Path) -> dict[str, list[float]]:
    path = Path(path)
    history: dict[str, list[float]] = {"step": [], "train": [], "val": []}
    if not path.is_file():
        return history
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                step = float(row.get("step", row.get("iteration", len(history["step"]))))
                train = float(row.get("train", row.get("train_loss", row.get("loss", "nan"))))
                val = float(row.get("val", row.get("val_loss", "nan")))
            except (TypeError, ValueError):
                continue
            history["step"].append(step)
            history["train"].append(train)
            history["val"].append(val)
    return history


@torch.no_grad()
def evaluate_token_stream(
    model: MiniGPT,
    data: torch.Tensor,
    block_size: int,
    device: str | torch.device = "cpu",
    max_windows: int = 512,
) -> dict[str, Any]:
    """Deterministic next-token loss/perplexity/accuracy over a token stream."""
    data = data.long()
    if len(data) <= block_size:
        return {
            "loss": float("nan"), "perplexity": float("nan"), "accuracy": 0.0,
            "y_true": np.array([], dtype=np.int64), "y_pred": np.array([], dtype=np.int64),
        }
    possible = list(range(0, len(data) - block_size, block_size))
    if max_windows and len(possible) > max_windows:
        indices = np.linspace(0, len(possible) - 1, max_windows, dtype=int)
        possible = [possible[int(i)] for i in indices]
    x = torch.stack([data[i:i + block_size] for i in possible]).to(device)
    y = torch.stack([data[i + 1:i + block_size + 1] for i in possible]).to(device)
    model.eval()
    logits, loss = model(x, y)
    pred = logits.argmax(-1).cpu().reshape(-1).numpy()
    true = y.cpu().reshape(-1).numpy()
    loss_value = float(loss.item())
    return {
        "loss": loss_value,
        "perplexity": float(math.exp(min(loss_value, 20))),
        "accuracy": float((pred == true).mean()),
        "y_true": true,
        "y_pred": pred,
    }


def generate_samples(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    prompts: Sequence[str],
    device: str | torch.device = "cpu",
    max_new_tokens: int = 160,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    no_repeat_ngram_size: int = 4,
) -> list[dict[str, Any]]:
    device = torch.device(device)
    results = []
    for prompt in prompts:
        ids = tokenizer.encode_prompt(prompt, max_context_tokens=model.block_size)
        context = torch.tensor([ids], dtype=torch.long, device=device)
        output = model.generate(
            context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            eos_id=tokenizer.eos_id,
            stop_at_eos=True,
        )[0].tolist()
        new_ids = output[len(ids):]
        results.append({
            "prompt": prompt,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "prompt_tokens": len(ids),
            "generated_tokens": len(new_ids),
            "eos_generated": tokenizer.eos_id in new_ids,
            "text": tokenizer.decode(new_ids),
        })
    return results
