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
- **paraphrase** — BM25 33,3% > hybrid 32,0% > vector 24,0%: vector *không* thắng như lý thuyết, vì `bge-small-en` là model tiếng Anh, hiểu kém paraphrase tiếng Việt. Đây là vấn đề chọn model (cần `bge-m3`), không phải vấn đề fusion.
- **mixed** — hybrid 100% > vector 98,5% > BM25 97,0%: hai retriever sai ở các doc khác nhau, RRF giữ doc cả hai cùng đồng ý.

**Khi không dùng hybrid:** tra cứu mã/ID/thuật ngữ chính xác (mã lỗi, SKU, tên hàm) → BM25 thuần, nhanh hơn ~10× (P99 3,4 ms so với 19,7 ms) mà chất lượng ngang nhau. Corpus đa ngôn ngữ hoặc query toàn paraphrase với một embedding model mạnh → vector thuần, vì BM25 chỉ thêm nhiễu. Khi ngân sách latency rất chặt thì cũng bỏ hybrid, vì nó phải chạy cả hai retriever.

---

## Điều ngạc nhiên nhất khi làm lab này

Semantic cache ở ngưỡng 0,75 (con số AWS) trả lời **sai 36%** probe trên corpus này — phải lên 0,85 mới về 0%. Và post-filter rơi về recall 0,00 ở filter 3,8% mà không có lỗi hay log nào báo.

---

## Bonus challenge

- [x] Đã làm bonus (xem `bonus/`)
- [ ] Pair work với: _(làm một mình)_
