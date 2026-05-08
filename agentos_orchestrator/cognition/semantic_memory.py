"""Semantic Dense Vector Episodic Memory.

Replaces character n-gram embeddings with TF-IDF + TruncatedSVD semantic
embeddings. Captures meaning, not just syntax.

Example: "Transfer funds in Bank A" and "Wire money in Bank B" will have
high cosine similarity because TF-IDF captures shared semantic terms.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from agentos_orchestrator.os_control.base import UiAction


class SemanticEmbedder:
    """Local semantic embedder using TF-IDF + SVD.

    Dimensions: 128 (captures semantic concepts like 'transfer' ≈ 'wire',
    'funds' ≈ 'money').
    """

    def __init__(self, n_components: int = 128, min_df: int = 1) -> None:
        self.n_components = n_components
        self.vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=0.95,
            ngram_range=(1, 2),  # Unigrams + bigrams capture phrases
            stop_words="english",
            lowercase=True,
            token_pattern=r"(?u)\b\w\w+\b",
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._is_fitted = False
        self._fallback_dim = n_components

    def _preprocess(self, text: str) -> str:
        """Clean and normalize text for embedding."""
        text = text.lower()
        # Normalize action types
        text = re.sub(r"\b(transfer|send|move|wire)\b", "transfer_action", text)
        text = re.sub(r"\b(funds|money|cash|amount)\b", "money_object", text)
        text = re.sub(r"\b(click|press|tap)\b", "click_action", text)
        text = re.sub(r"\b(type|write|enter|input)\b", "type_action", text)
        text = re.sub(r"\b(open|launch|start)\b", "open_action", text)
        text = re.sub(r"\b(close|quit|exit)\b", "close_action", text)
        text = re.sub(r"\b(save|store|export)\b", "save_action", text)
        return text

    def fit(self, texts: list[str]) -> None:
        """Fit the TF-IDF + SVD on a corpus of texts."""
        if not texts:
            return
        processed = [self._preprocess(t) for t in texts]
        if len(set(processed)) < 2:
            self._is_fitted = False
            return
        # Adjust max_df for tiny corpora to avoid ValueError
        n_docs = len(processed)
        if n_docs == 1:
            max_df = 1.0
        elif n_docs == 2:
            max_df = 1.0
        else:
            max_df = 0.95
        self.vectorizer = TfidfVectorizer(
            min_df=self.vectorizer.min_df,
            max_df=max_df,
            ngram_range=self.vectorizer.ngram_range,
            stop_words=self.vectorizer.stop_words,
            lowercase=self.vectorizer.lowercase,
            token_pattern=self.vectorizer.token_pattern,
        )
        tfidf_matrix = self.vectorizer.fit_transform(processed)
        n_features = tfidf_matrix.shape[1]
        if n_features < 2:
            self._is_fitted = False
            return
        n_components = min(n_features, self.n_components, n_docs)
        if n_components < 1:
            self._is_fitted = False
            return
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.svd.fit(tfidf_matrix)
        self._is_fitted = True

    def _pad_or_truncate(self, vec: np.ndarray) -> np.ndarray:
        """Ensure vector has exactly n_components dimensions."""
        if len(vec) < self.n_components:
            padded = np.zeros(self.n_components, dtype=np.float32)
            padded[: len(vec)] = vec
            return padded
        return vec[: self.n_components].astype(np.float32)

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text into a dense semantic vector."""
        if not self._is_fitted:
            # Return a simple bag-of-words vector as fallback
            return self._fallback_embed(text)
        processed = self._preprocess(text)
        tfidf = self.vectorizer.transform([processed])
        vec = self.svd.transform(tfidf)[0]
        vec = self._pad_or_truncate(vec)
        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts efficiently."""
        if not self._is_fitted:
            return np.array([self._fallback_embed(t) for t in texts])
        processed = [self._preprocess(t) for t in texts]
        tfidf = self.vectorizer.transform(processed)
        vecs = self.svd.transform(tfidf)
        # Ensure correct dimension
        vecs = np.array([self._pad_or_truncate(v) for v in vecs])
        # L2 normalize each row
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return (vecs / norms).astype(np.float32)

    def _fallback_embed(self, text: str) -> np.ndarray:
        """Simple fallback when model isn't fitted yet."""
        vec = np.zeros(self.n_components, dtype=np.float32)
        words = text.lower().split()
        for i, word in enumerate(words[: self.n_components]):
            # Hash-based embedding
            h = hash(word) % self.n_components
            vec[h] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two embeddings."""
        dot = float(np.dot(a, b))
        return max(-1.0, min(1.0, dot))


class SemanticEpisodicMemory:
    """Episodic memory with true semantic retrieval.

    Uses TF-IDF + SVD embeddings that capture meaning, not just character
    n-grams. Can recognize that "transfer funds" and "wire money" are
    semantically identical tasks.
    """

    def __init__(self, embedder: SemanticEmbedder | None = None) -> None:
        self.embedder = embedder or SemanticEmbedder(n_components=128)
        self._events: list[dict[str, Any]] = []
        self._embeddings: list[np.ndarray] = []

    @staticmethod
    def _event_text_payload(
        objective: str,
        action: UiAction,
        observation: str,
        outcome: str,
    ) -> str:
        return (
            f"{objective} {action.action_type} {action.selector} "
            f"{observation} {outcome}"
        )

    @classmethod
    def _stored_event_text(cls, event: dict[str, Any]) -> str:
        return cls._event_text_payload(
            str(event["objective"]),
            event["action"],
            str(event["observation"]),
            str(event["outcome"]),
        )

    def record(
        self,
        objective: str,
        action: UiAction,
        observation: str,
        outcome: str,
        reward: float,
        tags: list[str] | None = None,
    ) -> str:
        """Record an event with its semantic embedding."""
        event_id = f"sem_{len(self._events)}"
        text = self._event_text_payload(objective, action, observation, outcome)
        # Fit on the accumulated corpus once there is enough vocabulary/variance.
        if not self.embedder._is_fitted:
            existing_texts = [self._stored_event_text(event) for event in self._events]
            all_texts = [*existing_texts, text]
            self.embedder.fit(all_texts)
            if self.embedder._is_fitted and existing_texts:
                self._embeddings = [self.embedder.embed(item) for item in existing_texts]
        embedding = self.embedder.embed(text)
        self._events.append(
            {
                "event_id": event_id,
                "objective": objective,
                "action": action,
                "observation": observation,
                "outcome": outcome,
                "reward": reward,
                "tags": tags or [],
            }
        )
        self._embeddings.append(embedding)
        # Refit embedder periodically as vocabulary grows
        if len(self._events) % 20 == 0:
            all_texts = [self._stored_event_text(event) for event in self._events]
            self.embedder.fit(all_texts)
            # Re-embed all events with updated model
            self._embeddings = [self.embedder.embed(t) for t in all_texts]
        return event_id

    def retrieve_similar(
        self,
        objective: str,
        action_hint: UiAction | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve semantically similar past events."""
        if not self._events:
            return []
        query_text = objective
        if action_hint:
            query_text += f" {action_hint.action_type} {action_hint.selector}"
        query_emb = self.embedder.embed(query_text)
        similarities = [
            (self.embedder.cosine_similarity(query_emb, emb), event)
            for emb, event in zip(self._embeddings, self._events)
        ]
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [event for sim, event in similarities[:top_k] if sim > 0.1]

    def get_failure_patterns(
        self, objective: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Retrieve semantically similar failures."""
        all_similar = self.retrieve_similar(objective, top_k=len(self._events))
        failures = [e for e in all_similar if e["reward"] < 0]
        return failures[:top_k]

    def get_success_patterns(
        self, objective: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Retrieve semantically similar successes."""
        all_similar = self.retrieve_similar(objective, top_k=len(self._events))
        successes = [e for e in all_similar if e["reward"] > 0]
        return successes[:top_k]

    def semantic_search(
        self, query: str, top_k: int = 5
    ) -> list[tuple[float, dict[str, Any]]]:
        """Pure semantic search with similarity scores."""
        if not self._events:
            return []
        query_emb = self.embedder.embed(query)
        results = [
            (self.embedder.cosine_similarity(query_emb, emb), event)
            for emb, event in zip(self._embeddings, self._events)
        ]
        results.sort(key=lambda x: x[0], reverse=True)
        return [(sim, ev) for sim, ev in results[:top_k] if sim > 0.05]

    def transfer_learning_score(self, objective_a: str, objective_b: str) -> float:
        """Estimate how much knowledge from task A transfers to task B.

        High score means the agent should reuse strategies from A when doing B.
        """
        emb_a = self.embedder.embed(objective_a)
        emb_b = self.embedder.embed(objective_b)
        return self.embedder.cosine_similarity(emb_a, emb_b)
