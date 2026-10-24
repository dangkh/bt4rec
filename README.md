# CoMemTree

Collaborative behaviour memory tree for LLM-based sequential recommendation.

An LLM ranker scores the next item of a user from two pieces of behaviour evidence:

- **WHAT**: what the user currently prefers. This is the semantic behaviour of the user's
  latest interaction window (`recommendation_behavior` + `semantic_focus`).
- **HOW**: how that preference is expected to evolve. This is the `mechanism`, `scope` and
  `trajectory_signature` of the next behaviour state predicted by a collaborative
  reverse suffix Tree, learned from the behaviour sequences of training users.

Behaviour memories are extracted once, offline (one LLM call per 3-item window). Inference
uses one ranking call per user.

## Installation

```bash
pip install -e .            # or: export PYTHONPATH=$PWD/src
pip install -e .[gpu]       # torch, transformers, sentence-transformers, unsloth
```

## Repository layout

```
configs/            dataset presets (configs/datasets/*.yaml) and Tree hyper-parameters (configs/tree.yaml)
scripts/            pipeline wrappers 01_ ... 05_ and analysis scripts
src/bt4rec/
  data/        frozen protocol, items, candidates, behaviour cache
  behavior/    window construction and behaviour extraction (gemma / qwen backends)
  tree/        memories, embeddings, clustering, state assignment, Tree builder and query
  llm/         Qwen3 sentence encoder, Qwen2.5 list ranker
  ranking/     prompts, ranking conditions (modes), output parser
  guidance/    HOW sources for the ablations (random, Tree root, PMN, oracle)
  eval/        metrics (+ paired bootstrap), cost accounting
  cli/         prepare_protocol, precompute, build_tree, prepare_pmn, prepare_oracle, run
baseline/           LLMRank (native), i2Agent, MemRec, aMem4Rec, A-LLMRec, SASRec / LightGCN
tests/              unit tests on synthetic data (no GPU)
```

## Pipeline

```bash
bash scripts/01_prepare_protocol.sh CDs                                  # frozen users / candidates
CUDA_VISIBLE_DEVICES=0 bash scripts/02_precompute_behavior.sh CDs run    # behaviour memories (offline)
CUDA_VISIBLE_DEVICES=0 bash scripts/03_build_tree.sh CDs                 # collaborative Tree
CUDA_VISIBLE_DEVICES=0 bash scripts/05_run_ablation.sh CDs main          # ranking
```

Datasets: `CDs`, `Video_Games`, `ML`, `Books`. Every CLI has `--help`, for example
`python -m bt4rec.cli.run --help`. Set `DRY_RUN=1` to print the commands without running them.

The equivalent direct call for the ranking step:

```bash
python -m bt4rec.cli.run --dataset CDs --modes native,latest_what_only,latest_how_only,tree,tree_v2
```

## Ranking conditions (`--modes`)

All conditions share the same history (20 train items; 30 for ML), the same 20 frozen
candidates, the same output format (a JSON list of candidate labels) and the same parser.
`native` and `latest_what_only` use a plain ranking prompt; the other conditions use the
CoMemTree prompt and differ only in section 2 (WHAT) and section 3 (HOW).

| Mode | WHAT | HOW | |
|---|---|---|---|
| `native` | – | – | plain ranking prompt: history and candidates, no WHAT/HOW framing (LLMRank) |
| `latest_what_only` | latest window | – | native prompt + a short summary of the latest window; reference condition in `summary.json` |
| `latest_how_only` | – | latest window | |
| `tree` | latest window | Tree prediction | **CoMemTree** |
| `tree_v2` | latest 3 windows | Tree prediction | **CoMemTree v2** |

Ablation of the HOW source:

| Mode | WHAT | HOW |
|---|---|---|
| `random_how_only` | – | random training user |
| `global_continuation` | latest window | most frequent continuation at the Tree root (same for every user) |
| `pmn` | latest window | preference-matched neighbour users (`scripts/04_prepare_baselines.sh CDs pmn`) |
| `oracle` | latest window | derived from the held-out test item: upper bound, diagnostic only (`... CDs oracle`) |

## Tree settings (`configs/tree.yaml`)

| Setting | Default | Ablation flag |
|---|---|---|
| State induction | global KMeans (K = 50) over the trajectory embeddings | `--cluster-mode constrained`: hard bucket by mechanism |
| Assignment | every memory goes to its nearest state | `--confidence-filter`: reject low similarity or low margin assignments |
| Transition counts | per-user normalised | `--count-mode occurrence` |
| Query context | full Tree (`--tree-max-query-order 0`) | `--tree-max-query-order 1` / `2` (set on `run`) |

Inference maps test behaviours with the same settings the Tree was built with, read from
the Tree metadata. For example:

```bash
TREE_DIR=trees/CDs_1000_20_mech EXTRA_ARGS="--cluster-mode constrained" bash scripts/03_build_tree.sh CDs
python -m bt4rec.cli.run --dataset CDs --tree-dir trees/CDs_1000_20_mech --modes tree,tree_v2
```

To build the Tree and evaluate on two disjoint groups of users:

```bash
python scripts/make_disjoint_split.py --dataset CDs --n 300
TREE_DIR=trees/CDs_disjoint300 EXTRA_ARGS="--include-users data/CDs/protocol/disjoint300_tree_users.json" \
  bash scripts/03_build_tree.sh CDs
python -m bt4rec.cli.run --dataset CDs --tree-dir trees/CDs_disjoint300 --require-user-disjoint-tree \
  --user-ids-file data/CDs/protocol/disjoint300_eval_users.json --max-users 300 --modes tree,tree_v2
```

## Outputs

Each run writes the following files to `results/comemtree/<run>/`:

- `rankings.jsonl`: one row per user, with the evidence shown and, for every mode, the
  ranking, `target_rank`, parse diagnostics and exact token counts
- `summary.json`: metrics (Hit@K, NDCG@K, MRR) per mode, the differences and paired
  bootstrap against `latest_what_only`, parse quality, and the configuration
- `cost_summary.json`: LLM calls and tokens per user, and the number of memories
- `prompt_example.txt`: the filled-in prompt of every mode for one user

Cost of the offline behaviour extraction (tokenizer only, no GPU):

```bash
python scripts/count_behavior_tokens.py --dataset CDs
```

## Baselines

See `baseline/README.md`. All baselines use the same frozen protocol and write the same
`cost_summary.json` format.

## Tests

```bash
python -m unittest discover -s tests
```
