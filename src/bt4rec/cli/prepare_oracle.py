"""Generate an explicitly test-aware (non-deployable) next-behavior oracle.

The prompt sees the LAST observed training events and the held-out test item.
Its output is an abstract HOW (mechanism/scope/trajectory_signature) with no
candidate labels or target title in the ranking prompt.  The oracle must be
restricted to diagnostic upper-bound analysis, NEVER deployed as a baseline.


Run: python -m bt4rec.cli.prepare_oracle --help
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm

from bt4rec import shared
from bt4rec.baselines.oracle import prompt_for


def parser():
    p = argparse.ArgumentParser(description='Build test-aware target-user next-behavior oracle')
    p.add_argument('--dataset', required=True)
    p.add_argument('--items', default=None)
    p.add_argument('--sequences', default=None)
    p.add_argument('--candidate-file', required=True)
    p.add_argument('--user-ids-file', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--max-users', type=int, default=1000)
    p.add_argument('--start-user', type=int, default=0)
    p.add_argument('--model', default='unsloth/gemma-3-4b-it-unsloth-bnb-4bit')
    p.add_argument('--max-seq-length', type=int, default=4096)
    p.add_argument('--max-new-tokens', type=int, default=192)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--window-size', type=int, default=3,
                   help='Oracle extractor local window: last window_size-1 train items + test item')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--validate-only', action='store_true', help='Input paths only; no model')
    return p


def main(a):
    shared.set_seed(a.seed)
    if a.window_size < 2:
        raise ValueError('--window-size must be >=2')
    ds = shared._canonical_dataset(a.dataset)
    items_file = a.items or f'data/{ds}/items.json'
    seq_file = a.sequences or f'data/{ds}/user_sequences.json'
    items = shared.normalize_items(shared.load_json(items_file))
    sequences = {str(u): v for u,v in shared.load_json(seq_file).items()}
    frozen = shared.load_candidate_file(a.candidate_file)
    users = shared.load_user_ids_file(a.user_ids_file)
    if not users or len(users) != len(set(users)):
        raise ValueError('Eval file empty or duplicated')
    users = users[a.start_user:]
    if a.max_users:
        users = users[:a.max_users]
    tasks=[]
    for uid in users:
        if uid not in sequences:
            raise ValueError(f'Missing user {uid} in sequences')
        seq=sequences[uid]
        cid,gt=shared.get_candidates_for_user(uid,seq,{},frozen,a.seed)
        if len(gt)!=1 or len(cid)!=20 or gt!=shared.as_str_list(seq.get('test')):
            raise ValueError(f'Bad frozen candidate/test protocol for {uid}')
        hist=shared.as_str_list(seq.get('train'))
        if not hist:
            raise ValueError(f'No train history for {uid}')
        hist_items=[shared.get_item_info(x, items) for x in hist[-(a.window_size-1):]]
        fut=shared.get_item_info(gt[0], items)
        tasks.append((uid,gt[0],prompt_for(hist_items,fut)))
    print(f'[VALID] {len(tasks)} users, 20 candidates, target held-out, no model loaded', flush=True)
    if a.validate_only:
        return
    out=Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not a.resume:
        raise FileExistsError('Oracle output exists; set --resume to continue')
    processed={str(row['user_id']) for row in shared.read_jsonl(out) if row.get('future_aware') is True and row.get('user_id')}
    tasks=[t for t in tasks if t[0] not in processed]
    if not tasks:
        print('[DONE] All target-user oracle records present', flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError('Oracle Gemma/Qwen generation requires CUDA; validate-only works on CPU')
    import unsloth  # noqa
    from unsloth import FastLanguageModel
    model, tokenizer=FastLanguageModel.from_pretrained(
        model_name=a.model,max_seq_length=a.max_seq_length,load_in_4bit=True)
    FastLanguageModel.for_inference(model)
    model.eval()
    device=next(model.parameters()).device
    with out.open('a',encoding='utf8') as f:
        for uid,gt,prompt in tqdm(tasks,desc='future-aware oracle',unit='user'):
            # Gemma chat templates may reject a separate system role.
            messages=[{'role':'user','content':prompt}]
            try:
                formatted=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            except Exception as e:
                raise RuntimeError('Tokenizer missing usable chat template. Choose a supported Gemma/Qwen instruct model.') from e
            batch=tokenizer(formatted,return_tensors='pt',add_special_tokens=False).to(device)
            if batch['input_ids'].shape[-1]+a.max_new_tokens > a.max_seq_length:
                raise ValueError('Oracle input exceeds available context; do not truncate silently')
            with torch.inference_mode():
                result=model.generate(**batch,do_sample=False,max_new_tokens=a.max_new_tokens,
                                      pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                                      eos_token_id=tokenizer.eos_token_id)
            raw=tokenizer.decode(result[0,batch['input_ids'].shape[-1]:],skip_special_tokens=True)
            try:
                record=shared.extract_json_value(raw)
                if not isinstance(record,dict):
                    raise ValueError('Expected JSON object')
                obj={k:record[k] for k in ('mechanism','scope','trajectory_signature')}
                if not all(isinstance(v,str) and v.strip() for v in obj.values()):
                    raise ValueError('Missing oracle abstract fields')
            except Exception as e:
                raise ValueError(f'user {uid}: invalid future oracle output {raw[:500]!r}') from e
            item={'user_id':uid,'target_item_id':gt,'future_aware':True,
                  'source':'gemma_target_aware_heldout_next_item',
                  'oracle_behavior':obj, 'raw_oracle_response':raw}
            f.write(json.dumps(item,ensure_ascii=False)+'\n')
            f.flush()
    print(f'[DONE] {out}',flush=True)

if __name__ == '__main__':
    main(parser().parse_args())
