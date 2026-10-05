# Reflection — Lab 19

**Tên:** Trần Thu Phương
**Cohort:** A20-K4
**Path đã chạy:** lite (Windows 11, Python 3.13)

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

Precision@10 trung bình: hybrid 78,6% > BM25 77,8% > vector 73,2%.

- **exact** — BM25 96,7% = hybrid 96,7% > vector 88,7%: query chứa thuật ngữ verbatim ("PostgreSQL replication sharding") nên khớp từ khoá là đủ.
- **paraphrase** — với `bge-small-en` (model tiếng Anh) vector chỉ 24,0%, thua BM25 33,3%. Đổi sang `multilingual-e5-large` (NB2 §6), vector lên **75,3%** và thắng rõ: lỗi nằm ở model, không ở vector search.
- **mixed** — hybrid 100% > vector 98,5% > BM25 97,0%: hai retriever sai ở các doc khác nhau, RRF giữ doc cả hai cùng đồng ý.

**Khi không dùng hybrid:** tra cứu mã/ID chính xác (mã lỗi, SKU) → BM25 thuần, nhanh hơn ~6× (P99 3,4 so với 19,7 ms) mà chất lượng ngang nhau. Khi embedding đã mạnh và query chủ yếu là paraphrase → vector thuần: với e5-large, hybrid (87,0%) **thua** vector (92,2%) vì RRF cho BM25 yếu trọng số ngang bằng.

---

## Điều ngạc nhiên nhất khi làm lab này

Semantic cache ở ngưỡng 0,75 (con số AWS) trả lời **sai 36%** probe trên corpus này — phải lên 0,85 mới về 0%. Và post-filter rơi về recall 0,00 ở filter 3,8% mà không có lỗi hay log nào báo.

---

## Bonus challenge

- [x] Đã làm bonus (xem `bonus/`)
- [ ] Pair work với: _(làm một mình)_
