# Baselines (cùng frozen protocol với CoMemTree)

Mọi script chạy từ thư mục gốc repo (`python baseline/<file>.py --dataset <DS>`), tự đọc
`configs/datasets/<DS>.yaml`: 1000 eval user (`data/<DS>/protocol/eval_users_1000.json`),
20 candidate frozen, history = 20 TRAIN item (ML = 30). Kết quả ghi vào `results/baselines/<DS>/...`.
Không cần đặt `PYTHONPATH` (script tự thêm `src/`).

| File | Phương pháp | Output chính |
|---|---|---|
| `native_llm.py` | Native = **đúng mode `native` của CoMemTree** (cùng prompt, parser, Qwen) | `native/native_rankings.jsonl` |
| `i2agent.py --variant prior` | i2Agent rerank trên ranking Native (`base_recommender_ranking_prior`) | `i2agent_prior/` |
| `i2agent.py --variant direct` | i2Agent chấm thẳng 20 candidate, không có prior Native | `i2agent_direct/` |
| `amem_build_memory.py` | aMem4Rec: global memory epsilon-net + co-occurrence, dựng từ `precomputed/<DS>/train_shared.jsonl` (chính behaviour CoMemTree dùng để build Tree) | `amem/global_memory.json` |
| `amem_inference.py` | aMem4Rec retrieval + ranking (`--no_memory`: cùng prompt, không memory) | `amem/`, `amem_no_memory/` |
| `memrec_main.py` (+ `memrec/`) | MemRec: warm-up dựng memory trên TRAIN (Stage-R → ReRank → Stage-W), test với memory đóng băng | `memrec/` |
| `allmrec_main.py` (+ `allmrec/`) | A-LLMRec: SASRec → Stage 1 alignment → Stage 2 Qwen projection; **chỉ 1000 user, tối đa 20 interaction cho mọi giai đoạn** | `allmrec/` |
| `comemtree_baselines.py` | Random / Pop / LightGCN / SASRec (train trên 1000 user, full history) | `<tag>/` |

## Chi phí (mọi phương pháp cùng một định dạng `cost_summary.json`, schema `bt4rec_cost_v1`)

- `online`: phần chạy lúc test cho từng eval user — `llm_calls_per_user`, `input_tokens_per_user`,
  `output_tokens_per_user`, chi tiết `by_stage` (mỗi stage: calls, tokens, calls/user) và
  `shared_calls` (call không gắn user, vd. 1 call instruction-knowledge của i2Agent).
- `offline`: phần dựng memory một lần (aMem4Rec: Gemma evolution/merge), chia đều cho eval user.
- `upstream_reused`: phần dùng lại đã tính trước (trích xuất behaviour 1 call / cửa sổ 3 item,
  chung giữa CoMemTree và aMem4Rec).
- `memory`: số memory tạo ra (tổng, mỗi user, theo loại).
- 1 call = 1 prompt → 1 completion (batch nhiều user vào một `generate` không đổi số call).
  Token đếm chính xác theo từng mẫu: prompt không tính padding, output tính đến EOS.

| Phương pháp | LLM call / user (online) | Memory |
|---|---|---|
| Native | 1 | 0 |
| CoMemTree (mỗi mode) | 1 (+ trích xuất behaviour dùng chung, upstream) | behaviour/cửa sổ + Tree (states, nodes) |
| i2Agent direct | 4 + 1 call knowledge dùng chung | 4/user (probe, profile, dynamic interest, dynamic profile) + 1 knowledge |
| i2Agent prior | 4 + 1 Native (đọc từ file, token cộng vào nếu file có ghi) | như trên |
| aMem4Rec | 1 + 1 embedding query; offline: evolution/merge | global pool (số prototype cuối, merge, evolution, link) |
| MemRec | 2 (Stage-R + ReRank); offline warm-up ≈ 2–3/train user | user profile + item memory (Stage-W) |
| A-LLMRec | 1; offline: Stage-2 training pass (epochs × user) | 0 (tham số) |
| Random/Pop/LightGCN/SASRec | 0 | 0 (ghi số tham số) |

CoMemTree (`python -m bt4rec.cli.run`) giờ cũng ghi `cost_summary.json` trong run dir
(`by_mode`, `upstream_reused`, `memory`) và token chính xác theo từng user trong `rankings.jsonl`.

## Thay đổi so với các file cũ

- `native_llm_list_score_batch-4.py` dùng prompt riêng cho phim, history dạng JSON, 10 item,
  300 user, output list/score và fallback toàn bộ khi lỗi parse → **không so sánh được** với các
  mode CoMemTree. `native_llm.py` thay thế (prompt giống hệt mode `native`, đã kiểm tra trùng từng byte).
- i2Agent: gộp 2 file thành một (`--variant`); bỏ yêu cầu train/eval disjoint (protocol hiện tại
  train == eval, memory chỉ dùng TRAIN); history mặc định 20/30; đọc protocol trong `data/<DS>/protocol/`;
  candidate frozen là chuẩn (không bắt buộc trùng `test_neg`, giống CoMemTree); thêm cost + memory.
- aMem4Rec build: mặc định đọc `train_shared.jsonl` của CoMemTree, lọc theo `train_users_1000`,
  embedding chạy trên GPU nếu có; thêm cost (call Gemma gắn theo user kích hoạt, merge là shared).
- aMem4Rec inference: **sửa lỗi** query không được center trong khi memory đã center (builder mặc
  định center) → retrieval lệch; dùng đúng eval_users_1000 theo thứ tự (trước lấy theo thứ tự file
  sequences); không ghi đè/sinh lại file candidate; tính metric (trước không có); thêm cost.
