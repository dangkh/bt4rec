# BT4Rec / CoMemTree

Collaborative reverse behaviour Tree cho LLM sequential recommendation. LLM ranker nhận
**WHAT** (hành vi mới nhất của user) + **HOW** (trạng thái hành vi kế tiếp do Tree dự đoán).

## Cấu trúc

```
configs/            preset dataset (history, tree_dir, protocol) + siêu tham số Tree
scripts/            wrapper .sh mỏng, đánh số theo thứ tự pipeline
  01_prepare_protocol.sh   02_precompute_behavior.sh   03_build_tree.sh
  04_prepare_baselines.sh  05_run_ablation.sh           dev/verify_against_legacy.py
src/bt4rec/
  common/      io, text, seeding, vectors
  taxonomy.py  ★ chuẩn hoá mechanism/scope/direction + text embed (dùng chung builder & inference)
  data/        protocol, items, candidates, behavior_cache
  behavior/    windows, parsing, records, gemma.py, qwen.py   (backend trích xuất hành vi)
  tree/        memories, embeddings, clustering, assign ★, builder, vocab, query, stats
  llm/         encoder (Qwen3-Embedding), ranker (Qwen2.5 list ranker)
  ranking/     prompts, modes, parse
  guidance/    sources (latest/random/global/oracle), pmn, persistence (P(stay))
  eval/        metrics (+ paired bootstrap)
  inference/   inputs, audit, constants
  baselines/   pmn_summaries, oracle
  cli/         prepare_protocol, precompute, build_tree, prepare_pmn, prepare_oracle, run, config
tests/              unit test + dữ liệu giả lập (không cần GPU)
legacy/             bản gốc các script cũ (để tái lập số liệu cũ / đối chiếu)
```

## Cài đặt & chạy

```bash
pip install -e .            # hoặc: export PYTHONPATH=$PWD/src
pip install -e .[gpu]       # torch, transformers, sentence-transformers, unsloth

bash scripts/01_prepare_protocol.sh CDs
CUDA_VISIBLE_DEVICES=0 BACKEND=qwen bash scripts/02_precompute_behavior.sh CDs run
bash scripts/03_build_tree.sh CDs
CUDA_VISIBLE_DEVICES=0 bash scripts/04_prepare_baselines.sh CDs all      # chỉ cần cho pmn / oracle
MODES=latest_only,tree,tree_stay bash scripts/05_run_ablation.sh CDs main
```

Mọi CLI đều có `--help`, ví dụ `python -m bt4rec.cli.run --help`. Biến môi trường của từng
script ghi ở đầu file (`DRY_RUN=1` để chỉ in lệnh).

## Bảng đối chiếu file cũ → mới

| File cũ | Thay bằng |
|---|---|
| `prepare_data.py` | `bt4rec.cli.prepare_protocol` |
| `precompute_behavior.py` | `bt4rec.cli.precompute --backend gemma` |
| `precompute_train_test_shared_qwen.py` | `bt4rec.cli.precompute --backend qwen` |
| `build_reverse_behavior_tree.py` | `bt4rec.cli.build_tree` |
| `inference_bt4rec_ablation.py` | `bt4rec.cli.run` (tất cả mode, kể cả native / latest_only / recent3 / tree_original) |
| `inference_bt4rec.py`, `inference_bt4rec_reproduce.py` | giữ trong `legacy/` (prompt gốc, chỉ để tái lập) |
| `prepare_pmn_summaries.py` | `bt4rec.cli.prepare_pmn` |
| `prepare_target_user_oracle.py` | `bt4rec.cli.prepare_oracle` |
| `run_bt4rec_ablation.sh`, `run_comemtree_behavior_ablation_protocol.sh`, `run_recent3_tree_protocol.sh` | `scripts/05_run_ablation.sh` (chọn bằng `MODES=`) |
| `run_pmn_protocol.sh` | `scripts/04_prepare_baselines.sh … pmn` + `MODES=pmn` |
| module thiếu `inference_comemtree_qwen_shared` | `bt4rec.shared` (đã có sẵn) |
| module thiếu `oracle_probe_qwen` | `bt4rec.taxonomy` (fallback; dùng file gốc nếu có trên PYTHONPATH) |

## Thay đổi hành vi so với bản cũ

Code được **copy nguyên văn** từ script cũ; chỉ những điểm sau thay đổi:

1. **Chuẩn hoá nhãn thống nhất** (`taxonomy.py`): builder cũ giữ dấu `/` nên `same collection/series`
   được embed khác lúc train và test. Giờ cả hai dùng một hàm. Tree mới ghi `taxonomy_version`;
   inference cảnh báo nếu dùng Tree cũ → **nên build lại Tree**.
2. **Gán state dùng chung một quy tắc** (`tree/assign.py`): builder mặc định gán lại nhãn bằng
   cosine tới centroid tương thích gần nhất, giống hệt inference (`--label-assignment kmeans` = cũ).
3. **Signature của state = medoid** thay vì chuỗi xuất hiện nhiều nhất (`--state-signature mode` = cũ).
4. **`--collapse-consecutive-states`** được ghi vào metadata của Tree; inference gộp state liên tiếp
   y như lúc build, và P(stay) báo "không ước lượng được" thay vì một con số vô nghĩa.
5. **`random_how`** không còn crash khi `train_users == eval_users` (chỉ cần một user nguồn khác chính target).
6. **PMN**: `prepare_pmn` ghi `summary_version` v3 trong khi `run` chỉ nhận v4 (và dùng version trong
   `summary_key`), nên mọi profile bị bỏ qua. Giờ dùng chung `PMN_VERSION`.
7. **Summary** thêm `metrics_complete_paired` (cùng tập user cho mọi mode) và
   `paired_bootstrap_vs_latest_only` (CI 95% + p-value).

Tái lập Tree cũ: `LEGACY=1 bash scripts/03_build_tree.sh CDs`.

## Kiểm tra

```bash
python -m pytest tests                     # hoặc: python -m unittest discover -s tests
python scripts/dev/verify_against_legacy.py --legacy-dir legacy
```

`verify_against_legacy.py` chạy toàn bộ pipeline (protocol → precompute → tree → ranking) trên dữ liệu
giả lập bằng cả script cũ lẫn package mới ở chế độ tương thích, rồi so sánh từng file đầu ra.
