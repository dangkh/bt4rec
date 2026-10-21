"""
Inference for the dual-view collaborative behavior Tree.

Core idea
---------
The dual-view precompute intentionally separates:

    WHAT the target user currently prefers
        = latest recommendation_behavior + semantic_focus

from:

    HOW that preference is expected to evolve next
        = collaborative Tree predicted abstract state
          (mechanism + scope + trajectory_signature)

The final ranking LLM receives BOTH directly. There is NO extra LLM behavior
composer, so the method uses only one ranking LLM call per condition.

Pipeline
--------
precomputed test behaviors
    -> trajectory-only mapping to frozen KMeans states
    -> query reverse suffix Tree
    -> latest semantic behavior (WHAT)
       + predicted abstract transition/state (HOW)
    -> candidate ranking

Controlled ranking conditions
-----------------------------
1) Native:
      history + candidates

2) Latest-only:
      history + latest observed semantic behavior + candidates

3) Recent-3 (additional baseline):
      history + last 3 observed semantic behaviors + candidates
      (no predicted Tree transition)

4) Tree (main):
      history + latest observed semantic behavior
      + Tree predicted next abstract state + candidates

All conditions use the SAME frozen candidates and list ranking/output protocol.
The original Native, Latest-only, and Tree prompts are unchanged.

Important implementation details
--------------------------------
- No behavior generation at inference time.
- No GT/test item is used to construct behavior evidence.
- Candidate order is loaded EXACTLY from --candidate-file and is never reshuffled
  at inference time. This keeps Tree / Latest-only / Native aligned with the same
  frozen candidate protocol used by the compared rankers.
- Test behavior -> state mapping uses the SAME structured trajectory text used
  by build_reverse_behavior_tree_dual.py.
- When the Tree was trained with mechanism-constrained KMeans, test mapping is
  restricted to states with the same mechanism whenever compatible states exist.
- Ambiguous/rejected test mappings BREAK the recent state suffix by default,
  matching the segmented transition construction used by the current Tree builder.
- Semantic fields from other users / cluster medoids are NEVER injected into the
  ranking prompt. The predicted state contributes abstract transition information only.
- Incomplete rankings are NOT retried. They are sanitized like the native baseline.
  Summary still reports parse-complete quality for audit.

Expected inputs
---------------
Dual test behavior cache:
    precompute_test_behaviors_gemma_dual_v2.py output

Tree directory:
    build_reverse_behavior_tree_dual.py output containing:
      behavior_states.json
      behavior_state_embeddings.npy
      behavior_tree.json

Example
-------
python inference_tree_latest_behavior_gemma.py   --items data/CDs/items.json   --sequences data/CDs/user_sequences_10_5000.json   --negatives data/CDs/user_negatives_10_5000.json   --candidate-file data/CDs/user_sequences_candidates_seed42.json   --precomputed-behaviors precomputed/CDs/test_user_behaviors_gemma_dual_v2_300.jsonl   --tree-dir behavior_tree_out_dual_v2_k50   --output results/tree_latest_behavior_cd_300.jsonl   --model Qwen/Qwen2.5-7B-Instruct   --max-users 300   --top-next 1   --tree-max-query-order 0   --history-size 10   --llm-batch-size 8   --run-native   --run-latest-only   --seed 42


Refactored entry point (formerly inference_bt4rec_ablation.py). Run: python -m bt4rec.cli.run --help
"""
from __future__ import annotations

import argparse
import gc
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
from tqdm.auto import tqdm

from bt4rec.common.io import (
    append_jsonl,
    dump_json,
    load_json,
    load_user_ids_file,
    processed_users_from_jsonl,
    read_jsonl,
)
from bt4rec.common.seeding import set_seed
from bt4rec.data.behavior_cache import (
    latest_behavior_view,
    load_precomputed_behaviors,
    recent3_behavior_views,
)
from bt4rec.data.candidates import get_candidates_for_user, load_candidate_file, validate_frozen_users
from bt4rec.data.items import as_str_list, get_item_info, normalize_items
from bt4rec.eval.metrics import (
    aggregate_complete_intersection,
    aggregate_mode,
    metric_gain,
    paired_bootstrap,
    target_rank,
)
from bt4rec.guidance.persistence import compute_order1_persistence
from bt4rec.guidance.pmn import load_pmn_profiles, select_pmn_neighbours
from bt4rec.guidance.sources import choose_random_how, how_only, load_oracle, read_train_how, tree_global_how
from bt4rec.inference.audit import (
    audit_profile_source_restrictions,
    manifest_checkpoint,
    validate_tree_provenance,
)
from bt4rec.inference.constants import RANDOM_SOURCE_MODES, RANK_PROMPT_VERSION, SCHEMA_VERSION, STAY_MODES, STAY_PROMPT_VERSIONS
from bt4rec.inference.inputs import resolve_inputs, resolve_source_files
from bt4rec.llm.encoder import QwenEncoder
from bt4rec.llm.ranker import LocalQwenRanker
from bt4rec.ranking.modes import mode_prompt, requested_modes
from bt4rec.tree.query import ReverseBehaviorTree, latest_accepted_state_suffix
from bt4rec.tree.vocab import ClusterStateVocabulary


def comemtree_cost(a, rows, modes, users, test_cache):
    """Per-mode cost in the shared bt4rec_cost_v1 layout (see bt4rec.eval.cost).

    Online = 1 ranking call per user per mode. Behaviour extraction (one LLM call
    per 3-item window) and Tree construction are precomputed once and reported as
    upstream / memory so they can be compared with the baselines' build cost.
    """
    from bt4rec.eval.cost import CostTracker, memory_block
    by_mode = {}
    for mode in modes:
        t = CostTracker(f'comemtree:{mode}')
        for r in rows:
            m = r.get('rankings', {}).get(mode, {})
            t.llm('ranking', r['user_id'], int(m.get('input_tokens', 0) or 0),
                  int(m.get('output_tokens', 0) or 0))
        by_mode[mode] = t.summary(users)['online']
        by_mode[mode]['token_counts_available'] = all('input_tokens' in r.get('rankings', {}).get(mode, {})
                                                      for r in rows)
    windows = {u: len(test_cache[u].get('generated_behaviors') or []) for u in users}
    states = nodes = None
    try:
        states = len(load_json(a.states_json).get('states', []))
        tj = load_json(a.tree_json)
        nodes = len(tj.get('nodes', {})) if isinstance(tj, dict) else None
    except Exception:
        pass
    train_windows = None
    try:
        tb = a.train_behaviors or str(Path(a.precomputed_behaviors).parent / 'train_shared.jsonl')
        with open(tb, encoding='utf8') as f:
            train_windows = sum(1 for line in f if line.strip())
    except Exception:
        pass
    memory = memory_block(
        total=sum(windows.values()), per_user=windows,
        kinds={'behavior_memories_eval_users': sum(windows.values()),
               'train_behavior_memories': train_windows,
               'tree_behavior_states': states, 'tree_nodes': nodes},
        scope='per_user behaviour memories + one global Tree',
        note='Each 3-item TRAIN window yields one behaviour memory (1 extraction LLM call). '
             'The Tree is built from train-user memories without extra LLM calls.')
    return {'schema_version': 'bt4rec_cost_v1', 'method': 'comemtree', 'num_eval_users': len(users),
            'by_mode': by_mode,
            'upstream_reused': {'behavior_extraction_llm_calls_eval_users': sum(windows.values()),
                                'behavior_extraction_calls_per_user': (sum(windows.values()) / max(1, len(users))),
                                'train_behavior_extraction_llm_calls': train_windows,
                                'note': 'precomputed once (bt4rec.cli.precompute); shared by CoMemTree '
                                        'modes and aMem4Rec. Tokens are printed by precompute.'},
            'online_embedding': {'texts_per_user': (sum(windows.values()) / max(1, len(users))),
                                 'note': 'each behaviour is embedded once to map it to a Tree state'},
            'memory': memory}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CoMemTree/Qwen LIST ranking, standardized frozen train-only protocol. "
                    "Minimal: --dataset ML (other presets: CDs, Video_Games)."
    )
    p.add_argument("--dataset", required=True,
                   help="CDs | Video_Games | ML | Books | Software")
    p.add_argument("--tree", "--tree-dir", dest="tree_dir", default=None,
                   help="Optional Tree directory override. Default auto-resolved from --dataset, "
                        "e.g. trees/ML_1000_20, trees/CDs_1000_20, trees/VGs_1000_20.")
    p.add_argument("--tree-root", default="trees",
                   help="Folder holding the standardized dataset Trees (default: trees)")
    p.add_argument("--data-root", default="data")
    p.add_argument("--precompute-root", default="precomputed")
    p.add_argument("--output-root", default="results/comemtree")
    p.add_argument("--run-dir", default=None,
                   help="Explicit run folder, chiefly for --resume; otherwise named automatically")
    p.add_argument("--behavior-model", choices=["qwen", "gemma"], default="qwen",
                   help="Only sets default cache path and expected Tree provenance; never converts caches")
    p.add_argument("--items", default=None)
    p.add_argument("--sequences", default=None)
    p.add_argument("--negatives", default=None)
    p.add_argument("--candidate-file", default=None)
    p.add_argument("--user-ids-file", default=None,
                   help="Optional frozen cohort override; by default data/<dataset>/protocol/eval_users_1000.json")
    p.add_argument("--precomputed-behaviors", default=None,
                   help="Default: precomputed/<dataset>/test_shared.jsonl; optional model subfolder fallback")
    p.add_argument("--states-json", default=None)
    p.add_argument("--state-embeddings", default=None)
    p.add_argument("--tree-json", default=None)
    p.add_argument("--allow-tree-source-mismatch", action="store_true",
                   help="Override strict source-path provenance check (not recommended)")
    p.add_argument("--require-eval-users-in-tree", action="store_true",
                   help="Optional restrictive audit: reject eval users absent from Tree training-user metadata. "
                        "Default OFF: an unseen user can query a collaborative Tree by mapping behavior states.")
    p.add_argument("--embedding-model", default=None)
    p.add_argument("--embedding-device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--embedding-batch-size", type=int, default=128)
    p.add_argument("--cluster-text-field", choices=["structured_combined", "trajectory_signature", "behavior_signature"], default=None)
    p.add_argument("--mapping-min-similarity", type=float, default=None)
    p.add_argument("--mapping-min-margin", type=float, default=None)
    p.add_argument("--respect-mapping-filter", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--max-seq-length", type=int, default=8192)
    p.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--rank-max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--llm-batch-size", type=int, default=8)
    p.add_argument("--history-size", type=int, default=None,
                   help="Default: 30 for ML, 20 for other datasets")
    p.add_argument("--max-train-interactions", type=int, default=None,
                   help="Diagnostic only: cache's own precompute history takes precedence")
    p.add_argument("--max-behaviors", type=int, default=0,
                   help="Recent HOW windows to map into Tree; 0=all. Latest WHAT always latest overall.")
    p.add_argument("--top-next", type=int, default=1)
    p.add_argument("--tree-max-query-order", type=int, default=0,
                   help="0=full Tree configured max_order, 1/2/...=order cap")
    p.add_argument("--run-native", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--run-latest-only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--run-recent3", action=argparse.BooleanOptionalAction, default=True,
                   help="Additional baseline: last 3 OBSERVED semantic behaviors, without Tree prediction")
    p.add_argument("--strict-cache-history", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--start-user", type=int, default=0)
    p.add_argument("--max-users", type=int, default=1000,
                   help="Select up to N users in frozen eval_users_1000.json order; 0=all. "
                        "Use --user-ids-file to choose a different frozen cohort.")
    p.add_argument("--candidate-size", type=int, default=20)
    p.add_argument("--prompt-example-user", default=None,
                   help="User ID for the saved filled-in example; defaults to first selected user")
    p.add_argument("--save-all-prompts", action="store_true",
                   help="Optionally save filled-in prompts for every evaluated user and mode")
    p.add_argument("--prompts-only", action="store_true",
                   help="Map behaviors and save filled-in prompts, but skip Qwen ranking")
    p.add_argument("--validate-only", action="store_true",
                   help="Validate all frozen inputs/cache without loading encoder or ranking LLM")
    p.add_argument("--resume", action="store_true",
                   help="Continue results in explicit --run-dir (requires same frozen run configuration)")
    p.add_argument("--seed", type=int, default=42)
    # Ablation-specific settings. Base prompt/Tree mapping/ranker are unchanged.
    p.add_argument("--stay-prompt", choices=["v2", "v1"], default="v2",
                   help="Section 4 wording for *_stay modes. v2 (default): latest WHAT + P(stay), "
                        "short. v1: legacy wording from inference_bt4rec_ablation.py.")
    p.add_argument("--modes", default="latest_only,latest_how,random_how,global_continuation,pmn,tree,oracle",
                   help="Comma-separated modes. Extra: recent3,recent3_tree,tree_original,native."
                        " All modes share identical frozen history/candidates/parser.")
    p.add_argument("--train-behaviors", default=None,
                   help="TRAIN-only source behavior memory for random HOW; default precomputed/<dataset>/train_shared.jsonl")
    p.add_argument("--source-user-ids-file", default=None,
                   help="Optional source cohort whitelist; auto-uses protocol/train_users_1000.json if present")
    p.add_argument("--exclude-all-eval-sources", action="store_true",
                   help="Exclude ALL 1000 frozen eval users from random/PMN source pool (strict disjoint)")
    p.add_argument("--require-user-disjoint-tree", action="store_true",
                   help="Fail if Tree metadata indicates any training/eval user overlap or lacks coverage metadata")
    p.add_argument("--pmn-summaries", default=None,
                   help="Gemma-generated target/source preference profiles from prepare_pmn_summaries.py")
    p.add_argument("--pmn-embedding-model", default="Qwen/Qwen3-Embedding-0.6B",
                   help="Fixed PMN profile encoder; independent of Tree states and centroids")
    p.add_argument("--oracle-behaviors", default=None,
                   help="Future-aware target behavior cache; used ONLY for oracle mode")
    p.add_argument("--allow-debug-pmn", action="store_true",
                   help="Smoke-test only: accept debug/extractive preference profiles")
    return p.parse_args()


def main():
    a = resolve_inputs(parse_args())
    modes = requested_modes(a.modes)
    set_seed(a.seed)
    resolve_source_files(a,modes)
    items = normalize_items(load_json(a.items))
    sequences = {str(k):v for k,v in load_json(a.sequences).items()}
    test_cache = load_precomputed_behaviors(a.precomputed_behaviors)
    frozen = load_candidate_file(a.candidate_file)
    requested = load_user_ids_file(a.user_ids_file)
    if not requested or len(requested)!=len(set(requested)):
        raise ValueError('Frozen eval users list missing or duplicated')
    users = requested[a.start_user:]
    if a.max_users>0:
        users=users[:a.max_users]
    if not users:
        raise ValueError('No selected users')
    for uid in users:
        if uid not in sequences or uid not in frozen or uid not in test_cache:
            raise ValueError(f'{uid} missing sequences/candidates/test behaviors')
    prov = validate_tree_provenance(a,users)
    if a.require_user_disjoint_tree and prov.get('user_disjoint_tree_cohort') is not True:
        raise ValueError(f'--require-user-disjoint-tree failed: {prov}')
    audit = validate_frozen_users(users,sequences,frozen,test_cache,items,
                                  a.candidate_size,a.strict_cache_history)
    train_allowed, forbidden_sources = audit_profile_source_restrictions(a,users,sequences,modes)
    need_random = bool(RANDOM_SOURCE_MODES & set(modes))
    train_how = (read_train_how(a.train_behaviors,train_allowed,forbidden_sources)
                 if need_random else None)
    # Every target needs at least one OTHER source user (self is always excluded).
    # The legacy check required a source outside the eval cohort, which always
    # failed under the frozen protocol where train_users == eval_users.
    if need_random:
        lacking=[u for u in users if not (set(train_how)-{u})]
        if lacking:
            raise ValueError(f'Random HOW lacks a distinct source user for {len(lacking)} targets, first={lacking[:5]}')
    oracles = load_oracle(a.oracle_behaviors,users,sequences) if 'oracle' in modes else {}
    pmn_targets,pmn_sources,pmn_audit = ({},[],{})
    if 'pmn' in modes:
        pmn_targets,pmn_sources,pmn_audit = load_pmn_profiles(
            a.pmn_summaries,a.dataset,users,sequences,a.history_size,
            train_allowed,forbidden_sources,a.allow_debug_pmn,train_how)

    info = {'dataset':a.dataset,'num_users':len(users),'candidate_size':a.candidate_size,
            'history_size':a.history_size,'modes':modes,'tree_provenance':prov,
            'frozen':audit,'pmn':pmn_audit,
            'num_random_training_users':len(train_how) if train_how else None,
            'oracle_cache_loaded':len(oracles),'all_eval_sources_excluded':a.exclude_all_eval_sources}
    print('[ABLATION VALIDATION]',json.dumps(info,ensure_ascii=False,indent=2),flush=True)
    if a.validate_only:
        return

    stamp=datetime.now(ZoneInfo('Australia/Sydney')).strftime('%Y%m%d_%H%M%S')
    order_tag = 'fulltree' if a.tree_max_query_order==0 else f'order{a.tree_max_query_order}'
    run_dir=Path(a.run_dir) if a.run_dir else Path(a.output_root)/(f'{a.dataset}_{len(users)}_{a.history_size}_{order_tag}_ablation_{stamp}')
    if a.resume and not a.run_dir:
        raise ValueError('--resume requires --run-dir')
    if run_dir.exists() and not a.resume:
        raise FileExistsError(f'Output run dir exists: {run_dir}')
    run_dir.mkdir(parents=True,exist_ok=True)
    manifest=run_dir/'run_manifest.json'
    checkpoints=manifest_checkpoint(a,modes,users)
    if a.resume:
        if not manifest.is_file() or load_json(manifest).get('checkpoints')!=checkpoints:
            raise ValueError('Cannot resume: configuration/file hashes changed')
    else:
        dump_json(manifest,{'checkpoints':checkpoints,'config':vars(a),'validation':info,
                            'created_at_sydney':stamp,'description':'controlled bt4rec behavior guidance ablation'})
    path=run_dir/'rankings.jsonl'
    done = processed_users_from_jsonl(path) if a.resume else set()
    pending=[uid for uid in users if uid not in done]
    if not pending:
        # The previous summary is left in place instead of overwritten.
        print('[RESUME] No new users; existing outputs preserved',flush=True)
        return
    print(f'[RUN] {run_dir} | users={len(users)} pending={len(pending)} modes={modes}',flush=True)

    behaviors_by_user = {}
    for uid in pending:
        all_b=test_cache[uid]['generated_behaviors']
        behaviors_by_user[uid]=(all_b[-a.max_behaviors:] if a.max_behaviors>0 else all_b)
    vocab=ClusterStateVocabulary(states_json=a.states_json,state_embeddings_npy=a.state_embeddings,
            embedding_model=a.embedding_model,embedding_device=a.embedding_device,
            embedding_batch_size=a.embedding_batch_size,cluster_text_field=a.cluster_text_field,
            min_similarity=a.mapping_min_similarity,min_margin=a.mapping_min_margin,
            respect_mapping_filter=a.respect_mapping_filter)
    try:
        mapped=vocab.map_users(behaviors_by_user)
        tree=ReverseBehaviorTree(a.tree_json,vocab)
        global_how,global_audit=tree_global_how(tree)
        neighbours={}
        if 'pmn' in modes:
            # PMN embeddings use a fixed pretrained model, never Tree centroids.
            # Reuse the already-loaded instance ONLY when the model name matches.
            if vocab.encoder.model_name == a.pmn_embedding_model:
                neighbours=select_pmn_neighbours(vocab.encoder,pmn_targets,pmn_sources,pending)
            else:
                pmn_encoder=QwenEncoder(a.pmn_embedding_model,a.embedding_device,a.embedding_batch_size)
                try:
                    neighbours=select_pmn_neighbours(pmn_encoder,pmn_targets,pmn_sources,pending)
                finally:
                    pmn_encoder.release()
        prepared=[]
        for uid in tqdm(pending,desc='Prepare controlled guidance',unit='user'):
            seq=sequences[uid]
            latest=latest_behavior_view(test_cache[uid]['generated_behaviors'])
            recent3=recent3_behavior_views(test_cache[uid]['generated_behaviors'])
            suffix=latest_accepted_state_suffix(mapped.get(uid,[]),
                                                collapse_consecutive=tree.collapse_consecutive_states)
            tr=tree.query(suffix,top_next=a.top_next,max_query_order=a.tree_max_query_order)
            predicted=tr.get('predicted_next_states',[])
            if not predicted and any(m in modes for m in ('tree','tree_original','recent3_tree',
                                                     'tree_stay','tree_original_stay','recent3_tree_stay')):
                raise ValueError(f'{uid}: Tree query yielded no continuation')
            cand_ids,targets=get_candidates_for_user(uid,seq,{},frozen,a.seed)
            if len(targets)!=1:
                raise ValueError(f'{uid}: frozen protocol expects one held-out test target')
            hist=[get_item_info(x,items) for x in as_str_list(seq.get('train'))[-a.history_size:]]
            cmeta=[get_item_info(x,items) for x in cand_ids]
            hows={}
            metadata={}
            if 'latest_how' in modes:
                hows['latest_how']=how_only(test_cache[uid]['generated_behaviors'][-1])
            if need_random:
                # random_only reuses this exact sample (same seed/uid) without latest WHAT.
                hows['random_how'],metadata['random_how']=choose_random_how(train_how,uid,a.seed)
            if 'global_continuation' in modes:
                hows['global_continuation']=global_how
                metadata['global_continuation']=global_audit
            if any(m in modes for m in ('tree','recent3_tree','tree_stay','recent3_tree_stay')):
                hows['tree']=how_only(predicted[0])
                metadata['tree']={'state_id':predicted[0]['state_id'],
                                  'probability':predicted[0].get('probability'),
                                  'support_user_count':predicted[0].get('support_user_count')}
            if 'pmn' in modes:
                hows['pmn']=neighbours[uid]['how']
                metadata['pmn']={k:v for k,v in neighbours[uid].items() if k!='how'}
            if 'oracle' in modes:
                hows['oracle']=oracles[uid]
                metadata['oracle']={'future_aware':True,'diagnostic_only':True}
            stay = compute_order1_persistence(tree,mapped.get(uid,[])) if set(modes) & STAY_MODES else None
            if stay is not None:
                stay['prompt_version'] = STAY_PROMPT_VERSIONS[a.stay_prompt]
            # HOW of the user's own latest observed window (shown in the v2 stay block).
            latest_how = None
            if set(modes) & STAY_MODES:
                try:
                    latest_how = how_only(test_cache[uid]['generated_behaviors'][-1])
                except ValueError:
                    latest_how = None
            if stay is not None and a.stay_prompt == 'v2':
                stay['latest_observed_how'] = latest_how
            rec={'user_id':uid,'history_items':hist,'candidate_items':cmeta,
                 'candidate_ids':cand_ids,'targets':targets,
                 'latest_behavior':latest,'recent3_behaviors':recent3,'hows':hows,
                 'guidance_metadata':metadata,'tree_result':tr,
                 'state_mappings':mapped.get(uid,[]),'tree_query_state_suffix':suffix,
                 'predicted_transition':predicted, 'last_state_persistence':stay,
                 'stay_prompt':a.stay_prompt, 'latest_how':latest_how}
            rec['prompts']={m:mode_prompt(m,rec) for m in modes}
            prepared.append(rec)
        # All prompts generated BEFORE any ranker inference. Their modalities are inspectable.
        selected=next((x for x in prepared if x['user_id']==a.prompt_example_user),prepared[0])
        text=['Dataset: '+a.dataset,'User: '+selected['user_id'],
              'System prompt: You are a recommendation ranking system. Return only the requested candidate-label ranking.']
        for mode,p in selected['prompts'].items():
            text.extend(['','='*18+f' {mode} '+'='*18,p])
        (run_dir/'prompt_example.txt').write_text('\n'.join(text)+'\n',encoding='utf8')
        dump_json(run_dir/'prompt_example.json',{'dataset':a.dataset,'user_id':selected['user_id'],
               'candidate_label_map':{f'C{i:02d}':x for i,x in enumerate(selected['candidate_ids'],1)},
               'latest_behavior':selected['latest_behavior'],
               'guidance':selected['hows'],'last_state_persistence':selected['last_state_persistence'],
               'prompts':selected['prompts']})
        if a.save_all_prompts:
            with (run_dir/'prompts_all.jsonl').open('w',encoding='utf8') as fo:
                for rec in prepared:
                    fo.write(json.dumps({'user_id':rec['user_id'],'prompts':rec['prompts']},ensure_ascii=False)+'\n')
        if 'pmn' in modes:
            with (run_dir/'pmn_neighbors.jsonl').open('w',encoding='utf8') as fo:
                for rec in prepared:
                    fo.write(json.dumps({'user_id':rec['user_id'],
                           **rec['guidance_metadata']['pmn']},ensure_ascii=False)+'\n')
    finally:
        vocab.release_encoder()
    if a.prompts_only:
        print(f'[PROMPTS-ONLY] {run_dir}/prompt_example.txt; LLM reranking skipped')
        return
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ranker=LocalQwenRanker(a.model,a.max_seq_length,a.load_in_4bit,
                           a.rank_max_new_tokens,a.temperature,a.seed)
    ranked_by_mode={}
    for mode in modes:
        print(f'[RANK] {mode}: {len(prepared)} users; batch={a.llm_batch_size}',flush=True)
        jobs=[{'prebuilt_prompt':rec['prompts'][mode],'candidate_items':rec['candidate_items']}
              for rec in prepared]
        ranked_by_mode[mode]=ranker.rank_jobs(jobs,batch_size=a.llm_batch_size,
                                              call_type=mode,max_retries=0)
    for i,rec in enumerate(tqdm(prepared,desc='Save ablation rankings',unit='user')):
        ranks={}
        for mode in modes:
            ordered,meta=ranked_by_mode[mode][i]
            ranks[mode]={'ranked_item_ids':ordered,'target_rank':target_rank(ordered,rec['targets'][0]),**meta}
        row={'schema_version':SCHEMA_VERSION,'rank_prompt_version':RANK_PROMPT_VERSION,
             'run_dir':str(run_dir),'user_id':rec['user_id'],
             'ground_truth_item_ids':rec['targets'],'candidate_item_ids':rec['candidate_ids'],
             'candidate_label_map':{f'C{k:02d}':iid for k,iid in enumerate(rec['candidate_ids'],1)},
             'history_items':rec['history_items'],
             'latest_observed_behavior':rec['latest_behavior'],
             'recent3_observed_behaviors':rec['recent3_behaviors'],
             'guidance':rec['hows'],'guidance_audit':rec['guidance_metadata'],
             'tree_result':rec['tree_result'],'state_mappings':rec['state_mappings'],
             'tree_query_state_suffix':rec['tree_query_state_suffix'],
             'last_state_persistence':rec['last_state_persistence'],'rankings':ranks}
        append_jsonl(path,row)

    rows=[r for r in read_jsonl(path) if r.get('schema_version')==SCHEMA_VERSION
          and r.get('rank_prompt_version')==RANK_PROMPT_VERSION]
    counts=Counter(str(r.get('user_id')) for r in rows)
    if any(v!=1 for v in counts.values()) or set(counts)!=set(users):
        raise ValueError(f'Unexpected completed cohort/multiple rows; users={len(counts)}, expected={len(users)}')
    metric_all={mode:aggregate_mode(rows,mode,False) for mode in modes}
    metric_complete={mode:aggregate_mode(rows,mode,True) for mode in modes}
    # Paired view: users whose ranking was parse-complete in EVERY mode, so all
    # modes are compared on the same subset (metrics_complete_only is per mode).
    metric_complete_paired=aggregate_complete_intersection(rows,modes)
    significance=({mode:paired_bootstrap(rows,mode,'latest_only',seed=a.seed)
                   for mode in modes if mode!='latest_only'} if 'latest_only' in modes else {})
    parse_quality={}
    for mode in modes:
        recs=[r.get('rankings',{}).get(mode) for r in rows]
        good=[r for r in recs if isinstance(r,dict)]
        parse_quality[mode]={'rows':len(good),'parse_complete_count':sum(bool(r.get('parse_complete')) for r in good),
                             'parse_complete_rate':sum(bool(r.get('parse_complete')) for r in good)/len(good) if good else None}
    deltas={mode:metric_gain(metric_all[mode],metric_all.get('latest_only'))
            for mode in modes if mode!='latest_only'} if 'latest_only' in modes else {}
    if 'pmn' in modes and 'tree' in modes:
        deltas['tree_minus_pmn']=metric_gain(metric_all['tree'],metric_all['pmn'])
    if set(modes) & STAY_MODES and tree.collapse_consecutive_states:
        print('[WARN] Tree was built with --collapse-consecutive-states: P(stay) is not estimable; '
              'stay modes only add the "Not estimable" block.',flush=True)
    if 'tree_stay' in modes and 'tree' in modes:
        deltas['tree_stay_minus_tree']=metric_gain(metric_all['tree_stay'],metric_all['tree'])
    if 'tree_original_stay' in modes and 'tree_original' in modes:
        deltas['tree_original_stay_minus_tree_original']=metric_gain(metric_all['tree_original_stay'],metric_all['tree_original'])
    if 'recent3_tree_stay' in modes and 'recent3_tree' in modes:
        deltas['recent3_tree_stay_minus_recent3_tree']=metric_gain(metric_all['recent3_tree_stay'],metric_all['recent3_tree'])
    if 'recent3_tree' in modes and 'recent3' in modes:
        deltas['recent3_tree_minus_recent3']=metric_gain(metric_all['recent3_tree'],metric_all['recent3'])
    stay_rows=[r.get('last_state_persistence') for r in rows]
    stay_values=[float(x['probability_next_same_behavior_state']) for x in stay_rows
                 if isinstance(x,dict) and x.get('available')]
    stay_audit=({'order':1,'prompt_version':STAY_PROMPT_VERSIONS[a.stay_prompt],
                 'available_count':len(stay_values),
                 'unavailable_count':len(rows)-len(stay_values),
                 'mean_probability':float(np.mean(stay_values)) if stay_values else None,
                 'median_probability':float(np.median(stay_values)) if stay_values else None,
                 'unavailable_reasons':dict(Counter(x.get('reason') for x in stay_rows
                     if isinstance(x,dict) and not x.get('available')))}
                if set(modes) & STAY_MODES else None)
    cost=comemtree_cost(a,rows,modes,users,test_cache)
    dump_json(run_dir/'cost_summary.json',cost)
    summary={'schema_version':SCHEMA_VERSION,'rank_prompt_version':RANK_PROMPT_VERSION,
             'dataset':a.dataset,'modes':modes,'num_users':len(users),
             'metrics':metric_all,'metrics_complete_only':metric_complete,
             'metrics_complete_paired':metric_complete_paired,
             'paired_bootstrap_vs_latest_only':significance,
             'differences':deltas,'parse_quality':parse_quality,
             'tree_context_order_counts':dict(Counter(str(r.get('tree_result',{}).get('matched_order')) for r in rows)),
             'tree_root_fallback_count':sum(r.get('tree_result',{}).get('matched_order')==0 for r in rows),
             'ranking_llm':ranker.stats(), 'cost_by_mode':cost['by_mode'],
             'frozen_protocol':audit,'tree_provenance':prov,
             'last_state_persistence_audit':stay_audit,
             'pmnc':pmn_audit,'run_dir':str(run_dir),'config':vars(a),
             'warning':'ORACLE uses held-out test information; diagnostic only, not deployable. '
                       'Tree_original prompt differs from common controlled template; report separately.'}
    dump_json(run_dir/'summary.json',summary)
    print('[RESULTS]',json.dumps({'metrics':metric_all,'differences':deltas,
                                   'parse_quality':parse_quality},ensure_ascii=False,indent=2))
    print(f'Rankings: {path}\nSummary: {run_dir/"summary.json"}')

if __name__ == "__main__":
    main()
