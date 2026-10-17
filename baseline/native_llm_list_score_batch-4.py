#!/usr/bin/env python3
"""MovieLens Native ranking: list vs score, batched Unsloth Qwen/Gemma.
No memory, no few-shot. Frozen candidates are read-only and authoritative.
Standard sequences schema: {uid: {train: [...], test: [...]}}.
"""
import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
import time


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def ids(value):
    return [] if value is None else [str(x) for x in (value if isinstance(value, list) else [value])]


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def item_info(iid, metadata):
    m = metadata.get(str(iid), {})
    category = m.get('main_cat') or m.get('category') or m.get('categories') or m.get('genres') or 'Unknown'
    if isinstance(category, list):
        category = ' > '.join(map(str, category[:3]))
    return {'item_id': str(iid), 'title': str(m.get('title') or f'Unknown Item {iid}'), 'category': str(category)}


def build_prompt(history, candidates, output_mode):
    width = max(2, len(str(len(candidates))))
    labels = [f"C{i:0{width}d}" for i in range(1, len(candidates)+1)]
    profile = [{k: x[k] for k in ('title', 'category')} for x in history]
    entries = [dict(label=label, title=x['title'], category=x['category'])
               for label, x in zip(labels, candidates)]
    common = f"""You are a movie recommendation system.
Recommend the user's NEXT movie using their recent viewing history.

RECENT HISTORY (oldest to newest):
{json.dumps(profile, ensure_ascii=False, indent=2)}

CANDIDATE MOVIES:
{json.dumps(entries, ensure_ascii=False, indent=2)}

Use the provided movie titles and categories to assess compatibility with the user's preferences.
Candidate input order has no relevance meaning.
Use ONLY these candidate labels: {', '.join(labels)}.
Include all {len(labels)} labels exactly once. Do not invent, omit, or duplicate labels.
Return ONLY valid JSON. No rationale, explanation, or markdown fences.
"""
    if output_mode == 'list':
        return common + '\nRank all candidate labels from most relevant to least relevant.\n' + \
            'JSON format example only (your response must include ALL labels):\n' + \
            json.dumps({'ranked_labels': list(reversed(labels[:2]))})
    return common + '\nAssign each candidate a relevance score between 0 and 1; higher means more relevant.\n' + \
        'Scores do not need to sum to 1.\n' + \
        'JSON format example only (your response must include ALL labels):\n' + \
        json.dumps({'scores': [{'label': x, 'score': s} for x, s in zip(labels[:2], [0.82, 0.37])]})


def parse_output(raw, candidates, output_mode):
    valid_ids = [x['item_id'] for x in candidates]
    width = max(2, len(str(len(candidates))))
    labels = [f"C{i:0{width}d}" for i in range(1, len(candidates)+1)]
    mapping = dict(zip(labels, valid_ids))
    error, recovered, returned, scores = None, False, [], {}
    try:
        s, e = raw.find('{'), raw.rfind('}')
        if s < 0 or e < s:
            raise ValueError('No complete JSON object')
        text = raw[s:e+1]
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            if not exc.msg.startswith('Invalid control character'):
                raise
            obj = json.loads(text, strict=False)
            recovered = True
        if output_mode == 'list':
            value = obj.get('ranked_labels')
            if not isinstance(value, list):
                raise ValueError('ranked_labels must be a list')
            returned = [str(x).strip() for x in value]
        else:
            value = obj.get('scores')
            if not isinstance(value, list):
                raise ValueError('scores must be a list')
            for entry in value:
                if not isinstance(entry, dict):
                    raise ValueError('Every score entry must be an object')
                label = str(entry.get('label', '')).strip()
                returned.append(label)
                score = entry.get('score')
                if isinstance(score, bool) or not isinstance(score, (int, float)):
                    raise ValueError(f'Non-numeric score for {label}: {score!r}')
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError(f'Out-of-range score for {label}: {score!r}')
                scores[label] = float(score)
    except (ValueError, TypeError, AttributeError) as exc:
        error = str(exc)
    counts = Counter(returned)
    missing = [x for x in labels if x not in counts]
    unknown = [x for x in counts if x not in mapping]
    duplicate = [x for x, count in counts.items() if count > 1]
    ok = error is None and not missing and not unknown and not duplicate and len(returned) == len(labels)
    ranked_labels = returned if output_mode == 'list' else sorted(labels, key=lambda x: -scores.get(x, -1))
    ranking = [mapping[x] for x in ranked_labels] if ok else valid_ids[:]
    return ranking, dict(parse_ok=ok, fallback_used=not ok, parse_error=error,
                         control_char_recovered=recovered, missing_labels=missing,
                         unknown_labels=unknown, duplicate_labels=duplicate,
                         returned_count=len(returned), raw_returned_labels=returned,
                         scores_by_label=scores, label_to_item_id=mapping,
                         score_tie_count=(len(scores)-len(set(scores.values()))) if scores else 0)


def metrics(rows):
    result = {}
    for k in (1, 3, 5, 10):
        hits, recalls, ndcgs = [], [], []
        for row in rows:
            gt = set(row['ground_truth_item_ids'])
            top = row['reranked_item_ids'][:k]
            h = sum(x in gt for x in top)
            hits.append(float(h > 0)); recalls.append(h / len(gt))
            dcg = sum(1 / math.log2(i+2) for i, x in enumerate(top) if x in gt)
            ideal = sum(1 / math.log2(i+2) for i in range(min(k, len(gt))))
            ndcgs.append(dcg / ideal)
        for name, values in [('Hit', hits), ('Recall', recalls), ('NDCG', ndcgs)]:
            result[f'{name}@{k}'] = sum(values) / len(values) if values else None
    return result


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('items', 'sequences', 'candidate_file', 'output'):
        p.add_argument('--'+arg, required=True)
    p.add_argument('--eval_users', help='Optional JSON list of user IDs; defaults to candidate-file order')
    p.add_argument('--model', choices=['qwen', 'gemma'], required=True)
    p.add_argument('--model_name', help='Optional checkpoint override within the selected model family')
    p.add_argument('--output_mode', choices=['list', 'score'], required=True)
    p.add_argument('--history_size', type=int, default=10)
    p.add_argument('--number_of_users', type=int, default=300, help='0 = all selected users; otherwise require exactly N users')
    p.add_argument('--llm_batch_size', type=int, default=4)
    p.add_argument('--max_seq_length', type=int, default=8192)
    p.add_argument('--max_new_tokens', type=int, default=1024)
    p.add_argument('--load_in_4bit', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--debug_first_10', action='store_true', help='Run only first 10 selected users and stop')
    p.add_argument('--debug_n', type=int, default=10, help='Save full prompt before/after truncation for first N users')
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()


def main():
    a = arguments()
    if a.history_size <= 0 or a.llm_batch_size <= 0 or a.number_of_users < 0:
        raise ValueError('history_size/batch_size must be positive; number_of_users must be >=0')
    budget = a.max_seq_length - a.max_new_tokens
    if budget <= 0 or a.max_new_tokens <= 0:
        raise ValueError('Require max_seq_length > max_new_tokens > 0')
    output = Path(a.output)
    debug_file = output.with_name(output.stem + '_prompt_debug.jsonl')
    stats_file = output.with_name(output.stem + '_stats.json')
    if not a.overwrite and any(p.exists() for p in (output, debug_file, stats_file)):
        raise FileExistsError('Output exists: choose a new --output or explicitly use --overwrite. No implicit resume.')
    metadata = {str(k): v for k, v in read(a.items).items()}
    sequences = {str(k): v for k, v in read(a.sequences).items()}
    candidates = read(a.candidate_file)
    if isinstance(candidates, list):
        candidates = {str(x['user_id']): x for x in candidates}
    else:
        candidates = {str(k): v for k, v in candidates.items()}
    users = ids(read(a.eval_users)) if a.eval_users else list(candidates)
    if a.debug_first_10:
        users = users[:10]
        a.debug_n = max(a.debug_n, 10)
    elif a.number_of_users:
        
        if len(users) < a.number_of_users:
            raise ValueError(f'Requested {a.number_of_users} users but only {len(users)} available')
        users = users[:a.number_of_users]
    if not users or len(set(users)) != len(users):
        raise ValueError('Selected users must be nonempty and unique')
    jobs = []
    for uid in users:
        row, seq = candidates[uid], sequences[uid]
        cids = ids(row.get('candidates', row.get('candidate_item_ids')))
        gt = ids(seq.get('test'))
        cached_gt = ids(row.get('target', row.get('ground_truth', row.get('ground_truth_item_ids'))))
        if not cids or len(set(cids)) != len(cids) or not gt or not set(gt).issubset(cids):
            raise ValueError(f'user={uid}: invalid candidates or missing test target')
        if cached_gt and set(cached_gt) != set(gt):
            raise ValueError(f'user={uid}: frozen target differs from sequence test')
        history_ids = ids(seq.get('train'))[-a.history_size:]
        history = [item_info(x, metadata) for x in history_ids]
        ci = [item_info(x, metadata) for x in cids]
        jobs.append(dict(user_id=uid, history=history, candidate_items=ci,
                         candidate_item_ids=cids, ground_truth_item_ids=gt))
    print(f'[CONFIG] model={a.model} output_mode={a.output_mode} users={len(jobs)} batch={a.llm_batch_size}', flush=True)
    # Import Unsloth before torch/transformers, while keeping --help and parser tests GPU-free.
    import unsloth
    from unsloth import FastLanguageModel, FastModel
    from unsloth.chat_templates import get_chat_template
    import torch
    import numpy as np
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    name = a.model_name or ('Qwen/Qwen2.5-7B-Instruct' if a.model == 'qwen' else 'unsloth/gemma-3-4b-it')
    loader = FastLanguageModel if a.model == 'qwen' else FastModel
    model, tokenizer = loader.from_pretrained(model_name=name, max_seq_length=a.max_seq_length,
                                              load_in_4bit=a.load_in_4bit, full_finetuning=False)
    tokenizer = get_chat_template(tokenizer, chat_template='qwen-2.5' if a.model == 'qwen' else 'gemma3')
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    loader.for_inference(model)
    model.eval()
    device = next(model.parameters()).device
    output.parent.mkdir(parents=True, exist_ok=True)
    debug_file.write_text('', encoding='utf-8')
    results = []
    started = time.perf_counter()
    for start in range(0, len(jobs), a.llm_batch_size):
        chunk = jobs[start:start+a.llm_batch_size]
        prompts = [build_prompt(j['history'], j['candidate_items'], a.output_mode) for j in chunk]
        system = 'You are a movie recommendation ranking system. Return only valid JSON.'
        texts = [tokenizer.apply_chat_template([{'role':'system','content':system}, {'role':'user','content':p}],
                                                tokenize=False, add_generation_prompt=True) for p in prompts]
        # Chat template already contains special tokens. Use identical settings for both counts.
        full = [int(tokenizer(t, add_special_tokens=False, truncation=False, return_tensors='pt')['attention_mask'].sum().item()) for t in texts]
        inputs = tokenizer(texts, add_special_tokens=False, return_tensors='pt', padding=True,
                           truncation=True, max_length=budget).to(device)
        kept = inputs['attention_mask'].sum(dim=1).tolist()
        width = inputs['input_ids'].shape[1]
        torch.cuda.synchronize(); tick = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False,
                                       use_cache=True, pad_token_id=tokenizer.pad_token_id,
                                       eos_token_id=tokenizer.eos_token_id)[:, width:]
        torch.cuda.synchronize(); elapsed = time.perf_counter()-tick
        raw_outputs = tokenizer.batch_decode(generated, skip_special_tokens=True)
        eos = tokenizer.eos_token_id
        for i, (job, raw) in enumerate(zip(chunk, raw_outputs)):
            ranking, meta = parse_output(raw, job['candidate_items'], a.output_mode)
            tokens = generated[i].tolist()
            stop = tokens.index(eos) if eos in tokens else None
            out_n = stop+1 if stop is not None else len(tokens)
            record = dict(**job, reranked_item_ids=ranking, raw_output=raw, **meta,
                          full_input_tokens=full[i], input_tokens=int(kept[i]),
                          input_truncated=full[i]>kept[i], output_tokens=out_n,
                          output_has_eos=stop is not None,
                          output_limit_reached=stop is None and out_n>=a.max_new_tokens,
                          batch_time_sec=elapsed)
            results.append(record)
            print(f"[RANK] {len(results)}/{len(jobs)} user={job['user_id']} input={int(kept[i])}/{full[i]} truncated={record['input_truncated']} output={out_n}/{a.max_new_tokens} valid={meta['parse_ok']} missing={meta['missing_labels']} unknown={meta['unknown_labels']} duplicates={meta['duplicate_labels']}", flush=True)
            if len(results) <= a.debug_n:
                sent = tokenizer.decode(inputs['input_ids'][i][inputs['attention_mask'][i].bool()], skip_special_tokens=False)
                with debug_file.open('a', encoding='utf-8') as f:
                    f.write(json.dumps(dict(**record, prompt_before_truncation=texts[i], prompt_sent_to_model=sent), ensure_ascii=False)+'\n')
                print('[RAW OUTPUT] '+raw, flush=True)
        save(output, results)
    count = len(results)
    stats = dict(config=vars(a), resolved_model_name=name, selected_user_ids=users, num_results=count,
                 parse_complete_ratio=sum(r['parse_ok'] for r in results)/count,
                 fallback_count=sum(r['fallback_used'] for r in results),
                 users_with_score_ties=sum(r['score_tie_count'] > 0 for r in results),
                 truncated_count=sum(r['input_truncated'] for r in results),
                 output_limit_count=sum(r['output_limit_reached'] for r in results),
                 metrics_all_users=metrics(results), elapsed_sec=time.perf_counter()-started)
    save(stats_file, stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f'Results: {output}\nStats: {stats_file}\nDebug prompts: {debug_file}')


if __name__ == '__main__':
    main()
