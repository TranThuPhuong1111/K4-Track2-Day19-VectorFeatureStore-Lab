"""HybridMemoryAgent — episodic memory (vector store) + stable profile (feature store).

Bonus challenge, Lab 19. Design rationale lives in bonus/ARCHITECTURE.md; this
file only has to make those decisions runnable.

    remember(text, user_id)  chunk -> embed -> upsert into ONE Qdrant collection,
                             every point tagged with `user_id` (payload filter)
    recall(query, user_id)   Feast online lookup (profile + query velocity)
                             + in-process "push" buffer of recent queries
                             + hybrid search (BM25 + vector + profile boost, RRF k=60)
                             filtered by user_id -> assembled context string

No LLM is called: recall() returns the context an LLM would receive.
"""
from __future__ import annotations

import re
import sys
import time
import unicodedata
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent import RuleBasedPlanner  # noqa: E402  -- reuse topic detection from NB6
from app.embeddings import Embedder  # noqa: E402

COLLECTION = "bonus_memory"
RRF_K = 60                  # same constant as NB2 / app/search.py
# The profile ranker only knows "this memory is in the user's favourite topic";
# at full weight it would outvote an actual BM25/vector match, so it is a tie-breaker.
PROFILE_WEIGHT = 0.3
CHUNK_MAX_WORDS = 60        # ~80-100 tokens for Vietnamese with a BPE tokenizer
RECENT_WINDOW_S = 3600      # "queries in the last hour"

# Fallback when `feast apply` (NB4) has not been run: the agent still works,
# it just loses personalisation. Values mirror NB4's synthetic row for u_001.
DEFAULT_PROFILE = {"topic_affinity": None, "reading_speed_wpm": None,
                   "preferred_language": "vi", "queries_last_hour": 0}


def strip_diacritics(text: str) -> str:
    """'tự động mở rộng' -> 'tu dong mo rong'. 'đ' is not a combining mark, so map it by hand."""
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


# Vietnamese function words. A per-user memory set is tiny, so BM25's IDF is too
# weak to suppress them: "CHO tôi summary..." would match a note saying "... cho batch job".
VI_STOPWORDS = {"cho", "tôi", "của", "và", "là", "gì", "về", "có", "không", "được", "các",
                "những", "này", "đã", "đang", "với", "thì", "một", "khi", "nào", "trong", "để"}


def tokenize(text: str) -> list[str]:
    """Whitespace tokens minus VI stopwords, each emitted twice: with and without diacritics.

    Vietnamese users often type without dấu ("bao mat") or mix vi/en. Indexing
    both forms lets BM25 match either spelling without a VN word segmenter.
    """
    words = [w for w in re.findall(r"\w[\w\-/.]*", text.lower()) if w not in VI_STOPWORDS]
    return words + [strip_diacritics(w) for w in words if strip_diacritics(w) != w]


def chunk(text: str, max_words: int = CHUNK_MAX_WORDS) -> list[str]:
    """Greedy sentence packing: never split a sentence, cap a chunk at max_words."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?…])\s+|\n+", text) if s.strip()]
    chunks, cur = [], []
    for s in sentences:
        if cur and len(" ".join(cur + [s]).split()) > max_words:
            chunks.append(" ".join(cur))
            cur = []
        cur.append(s)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


@dataclass
class Memory:
    point_id: str
    user_id: str
    text: str
    topic: str | None
    ts: float


class HybridMemoryAgent:
    def __init__(self, feast_repo: Path | None = ROOT / "app" / "feast_repo",
                 top_k: int = 3) -> None:
        self.embedder = Embedder()
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(size=self.embedder.dim,
                                               distance=models.Distance.COSINE),
        )
        self.top_k = top_k
        # Per-user lexical side of the hybrid index (rebuilt lazily on write).
        self._memories: dict[str, list[Memory]] = defaultdict(list)
        self._bm25: dict[str, BM25Okapi | None] = {}
        # Streaming-style freshness: recall() pushes here immediately, the way a
        # Feast PushSource would; Feast's batch value catches up on materialize.
        self._recent: dict[str, deque[tuple[float, str]]] = defaultdict(deque)
        self.store = self._open_feast(feast_repo)

    # ── feature store ───────────────────────────────────────────────────
    @staticmethod
    def _open_feast(repo: Path | None):
        if repo is None or not (repo / "registry.db").exists():
            return None
        try:
            from feast import FeatureStore
            return FeatureStore(repo_path=str(repo))
        except Exception:  # noqa: BLE001 -- degrade, never crash recall()
            return None

    def profile(self, user_id: str) -> dict:
        prof = dict(DEFAULT_PROFILE)
        if self.store is None:
            return prof
        try:
            row = self.store.get_online_features(
                features=["user_profile_features:topic_affinity",
                          "user_profile_features:reading_speed_wpm",
                          "user_profile_features:preferred_language",
                          "query_velocity_features:queries_last_hour"],
                entity_rows=[{"user_id": user_id}],
            ).to_dict()
            prof.update({k: v[0] for k, v in row.items() if k != "user_id" and v[0] is not None})
        except Exception:  # noqa: BLE001
            pass
        return prof

    # ── write path ──────────────────────────────────────────────────────
    def remember(self, text: str, user_id: str = "u_001") -> None:
        """Add a new piece of episodic memory for this user."""
        pieces = chunk(text)
        if not pieces:
            return
        vectors = list(self.embedder.embed(pieces))
        now = time.time()
        points = []
        for piece, vec in zip(pieces, vectors):
            m = Memory(str(uuid.uuid4()), user_id, piece,
                       RuleBasedPlanner.detect_topic(piece), now)
            self._memories[user_id].append(m)
            points.append(models.PointStruct(
                id=m.point_id, vector=vec.tolist(),
                payload={"user_id": user_id, "text": piece, "topic": m.topic, "ts": now}))
        self.client.upsert(COLLECTION, points=points)
        self._bm25[user_id] = None          # invalidate; rebuilt on next recall

    # ── read path ───────────────────────────────────────────────────────
    def _keyword_rank(self, query: str, user_id: str, depth: int) -> list[str]:
        mems = self._memories[user_id]
        if not mems:
            return []
        if self._bm25.get(user_id) is None:
            self._bm25[user_id] = BM25Okapi([tokenize(m.text) for m in mems])
        scores = self._bm25[user_id].get_scores(tokenize(query))
        order = sorted(range(len(mems)), key=lambda i: -scores[i])[:depth]
        return [mems[i].point_id for i in order if scores[i] > 0]

    def _vector_rank(self, query: str, user_id: str, depth: int) -> list[str]:
        qv = next(self.embedder.embed([query])).tolist()
        hits = self.client.query_points(
            COLLECTION, query=qv, limit=depth,
            # Isolation lives INSIDE the query (filtered-ANN, NB5) — never post-filter.
            query_filter=models.Filter(must=[models.FieldCondition(
                key="user_id", match=models.MatchValue(value=user_id))]),
        ).points
        return [str(h.id) for h in hits]

    def search(self, query: str, user_id: str, affinity: str | None) -> list[Memory]:
        depth = max(self.top_k * 5, 20)
        rankers = [(self._keyword_rank(query, user_id, depth), 1.0),
                   (self._vector_rank(query, user_id, depth), 1.0)]
        by_id = {m.point_id: m for m in self._memories[user_id]}
        if affinity:   # 3rd ranker: profile boost = memories in the user's favourite topic, newest first
            fav = sorted((m for m in by_id.values() if m.topic == affinity), key=lambda m: -m.ts)
            rankers.append(([m.point_id for m in fav][:depth], PROFILE_WEIGHT))
        rrf: dict[str, float] = defaultdict(float)
        for ranking, weight in rankers:
            for rank, pid in enumerate(ranking, start=1):   # rank is 1-based
                rrf[pid] += weight / (RRF_K + rank)
        best = sorted(rrf, key=lambda pid: -rrf[pid])[: self.top_k]
        return [by_id[pid] for pid in best if pid in by_id]

    def recent_queries(self, user_id: str, now: float | None = None) -> list[str]:
        now = now or time.time()
        buf = self._recent[user_id]
        while buf and now - buf[0][0] > RECENT_WINDOW_S:
            buf.popleft()
        return [q for _, q in buf]

    def recall(self, query: str, user_id: str = "u_001") -> str:
        """Retrieve top-K memories + user profile features -> assembled context."""
        prof = self.profile(user_id)
        recent = self.recent_queries(user_id)          # BEFORE logging this query
        memories = self.search(query, user_id, prof.get("topic_affinity"))
        self._recent[user_id].append((time.time(), query))

        lines = [f"[user {user_id}] language={prof['preferred_language']}"
                 f" · reading_speed={prof['reading_speed_wpm'] or '?'}wpm"
                 f" · topic_affinity={prof['topic_affinity'] or 'unknown'}"]
        lines.append(f"Recent activity: {prof['queries_last_hour']} queries/hour (Feast, batch)"
                     f" · this session: {recent or '[]'} (push buffer)")
        lines.append("Top memories:")
        lines += [f"  {i}. ({m.topic or '-'}) {m.text}" for i, m in enumerate(memories, 1)] \
            or ["  (none)"]
        return "\n".join(lines)
