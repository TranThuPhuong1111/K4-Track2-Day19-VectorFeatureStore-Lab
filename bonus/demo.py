"""Five-query demo for HybridMemoryAgent.  Run:  python bonus/demo.py

Feast profile features are used if NB4 has been run (app/feast_repo/registry.db
exists); otherwise the agent degrades to "profile unknown" and still exits 0.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import HybridMemoryAgent  # noqa: E402

MEMORIES_U001 = [
    # đã đọc / ghi chú — đủ chủ đề để vector, BM25 và profile boost cạnh tranh nhau
    "Đã đọc: Kubernetes Pod lifecycle. Pod đi qua các pha Pending, Running, Succeeded, Failed. "
    "Liveness probe restart container khi treo, readiness probe gỡ Pod khỏi Service.",
    "Ghi chú: Horizontal Pod Autoscaler tăng số replica khi CPU vượt 70%. "
    "Kết hợp cluster autoscaler để hạ tầng co giãn theo lưu lượng người dùng.",
    "Đã đọc: tối ưu chi phí AWS bằng spot instance cho batch job, savings plan cho workload ổn định.",
    "Ghi chú họp: cloud security cần IAM least privilege, bật mã hoá S3 at rest bằng KMS, "
    "và audit CloudTrail hàng tuần. Không để access key trong code.",
    "Đã đọc: zero-trust network — mọi request đều phải xác thực, kể cả trong VPC nội bộ.",
    "Ghi chú: RAG với hybrid search BM25 + vector, RRF k=60, rank tính từ 1.",
    "Đã đọc: PostgreSQL index B-tree vs GIN, khi nào dùng partial index.",
]
MEMORY_U002 = "Bí mật của u_002: kế hoạch migrate Kubernetes sang ECS quý 4, ngân sách 2 tỷ."

QUERIES = [
    ("1 · vector hit",       "Tôi đã đọc gì về Kubernetes?"),
    ("2 · cần profile",      "Recommend đọc gì tiếp"),
    ("3 · cần fresh activity", "Tôi đang quan tâm gì gần đây?"),
    ("4 · paraphrase",       "Tài liệu về tự động mở rộng hạ tầng?"),
    ("5 · mixed + profile",  "Cho tôi summary cloud security"),
]


def main() -> int:
    agent = HybridMemoryAgent()
    print(f"Feast online store: {'connected' if agent.store else 'not applied (run NB4) — profile disabled'}")
    for text in MEMORIES_U001:
        agent.remember(text, user_id="u_001")
    agent.remember(MEMORY_U002, user_id="u_002")
    print(f"Remembered {len(MEMORIES_U001)} notes for u_001, 1 note for u_002\n")

    for label, q in QUERIES:
        print(f"=== Query {label}: {q!r}")
        ctx = agent.recall(q, user_id="u_001")
        print(ctx, "\n")
        # Privacy check: nothing u_002 stored may ever surface for u_001.
        assert "u_002" not in ctx and "ECS" not in ctx, "cross-user memory leak!"

    print("Isolation check passed — u_002's memory never appeared in u_001's context.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
