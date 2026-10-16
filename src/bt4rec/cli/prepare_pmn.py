"""PMN stage 1: leakage-controlled user preference summaries (Gemma 3).

Target summary: last H observed TRAIN items of target user.
Source summary: last H items of source user's TRAIN prefix, excluding its
terminal TRAIN item. Its terminal local-behavior window provides the observed
continuation. No Tree, candidates, test item or source continuation are shown
to the summary generator. The exact same summary prompt is used for both roles.

Imports shared utilities from inference_comemtree_qwen_shared.py (the LIST
ranking version used for CoMemTree reproducibility).


Refactored entry point (formerly prepare_pmn_summaries.py). Run: python -m bt4rec.cli.prepare_pmn --help
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from bt4rec import shared
from bt4rec.baselines.pmn_summaries import (
    SUMMARY_VERSION,
    decode_summary,
    make_cases,
    summary_prompt,
)


def salvage_truncated_summary(raw: str, min_words: int = 20) -> str:
    """Last resort for an output cut off by the token budget.

    Keeps the text after "preference_summary": up to the last complete sentence.
    Raises if fewer than ``min_words`` words survive. Rows are flagged by the caller.
    """
    m = re.search(r'"preference_summary"\s*:\s*"(.*)', str(raw or ''), flags=re.S)
    if not m:
        raise ValueError('No preference_summary field to salvage')
    text = m.group(1).split('"}')[0]
    text = text.replace('\\"', '"').replace('\\n', ' ')
    cut = text.rfind('.')
    if cut > 0:
        text = text[:cut + 1]
    text = shared.clean_text(text)
    if len(text.split()) < min_words:
        raise ValueError(f'Salvaged summary too short ({len(text.split())} words)')
    return text


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description='PMN preference profiles from frozen TRAIN prefix')
    p.add_argument('--dataset',required=True)
    p.add_argument('--data-root',default='data')
    p.add_argument('--train-behaviors',required=True)
    p.add_argument('--precomputed-behaviors',default=None, help='Optional target cache for strict history validation')
    p.add_argument('--train-user-ids',default=None,help='Optional frozen training-user cohort')
    p.add_argument('--history-size',type=int,default=None)
    p.add_argument('--window-size',type=int,default=3)
    p.add_argument('--window-index-offset',default='auto',
                   help='auto aligns windows using cached provenance; or set nonnegative 0/1/2/...')
    p.add_argument('--max-target-users',type=int,default=1000)
    p.add_argument('--max-source-users',type=int,default=0)
    p.add_argument('--model',default='unsloth/gemma-3-4b-it-unsloth-bnb-4bit')
    p.add_argument('--max-seq-length',type=int,default=4096)
    p.add_argument('--max-new-tokens',type=int,default=320,
                   help='Gemma output budget (legacy 192 truncated some summaries mid-JSON)')
    p.add_argument('--retry-max-new-tokens',type=int,default=768,
                   help='Larger budget for a single retry of any summary whose JSON is truncated/invalid')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output',default=None)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('--debug-extractive-summary',action='store_true',
                   help='TEST ONLY: generate deterministic non-LLM summaries; not for paper experiments')
    return p.parse_args()


def main() -> None:
    a=parse_args()
    shared.set_seed(a.seed)
    ds=shared._canonical_dataset(a.dataset)
    h=a.history_size or (30 if ds=='ML' else 20)
    if min(a.window_size,a.batch_size,a.max_new_tokens,h) <= 0:
        raise ValueError('Window size, history size, batch size, max-new-tokens must be positive')
    out=Path(a.output or f'precomputed/{ds}/pmn_preference_summaries_gemma_v4.jsonl')
    cases,audit=make_cases(ds,a.data_root,a.train_behaviors,h,a.window_size,
                           a.window_index_offset,a.max_target_users,a.max_source_users,
                           a.precomputed_behaviors,a.train_user_ids)
    print('[PMN-VALIDATE]',json.dumps(audit,ensure_ascii=False),flush=True)
    if a.validate_only:
        return
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists() and not a.resume:
        raise FileExistsError(f'{out} already exists; use --resume or another --output')
    seen={}
    if out.exists():
        for old in shared.read_jsonl(out):
            k=(old.get('role'),str(old.get('user_id')))
            if k in seen and seen[k].get('summary_key') != old.get('summary_key'):
                raise ValueError(f'Conflicting stored profile records {k}; delete old output')
            seen[k]=old
    pending=[]
    for c in cases:
        old=seen.get((c['role'],c['user_id']))
        if old:
            if old.get('summary_key') != c['summary_key'] or old.get('summary_version') != SUMMARY_VERSION:
                raise ValueError(f"Stale summary {c['role']}:{c['user_id']}; source prefix or protocol changed. Use NEW output path.")
            if not shared.clean_text(old.get('preference_summary')):
                raise ValueError('Stored preference_summary empty')
        else:
            pending.append(c)
    print(f'[PMN] total={len(cases)} reused={len(cases)-len(pending)} pending={len(pending)} output={out}',flush=True)
    if not pending:
        return
    if a.debug_extractive_summary:
        print('[WARNING] DEBUG extractive profiles; DO NOT REPORT AS Gemma PMN',flush=True)
        def debug_summary(c):
            return 'Preferences reflected by observed categories: '+', '.join(
                list(dict.fromkeys(i['category'] for i in c['history_items']))[-8:])
        with out.open('a',encoding='utf8') as f:
            for c in pending:
                row={'schema_version':'pmn_summary_v1','summary_version':SUMMARY_VERSION,
                     'summary_backend':'debug_extractive','dataset':ds,
                     'role':c['role'],'user_id':c['user_id'],
                     'summary_key':c['summary_key'], 'history_item_ids':c['history_ids'],
                     'history_size':h,'preference_summary':debug_summary(c),
                     'source_how':c['source_how'],'terminal_window_index':c['terminal_window_index'],
                     'terminal_train_item_id':c.get('terminal_train_item_id')}
                f.write(json.dumps(row,ensure_ascii=False)+'\n')
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Gemma summary generation requires CUDA. --validate-only works without CUDA.')
    import unsloth  # noqa: F401
    from tqdm.auto import tqdm
    from unsloth import FastLanguageModel
    model,tokenizer=FastLanguageModel.from_pretrained(
        model_name=a.model,max_seq_length=a.max_seq_length,load_in_4bit=True)
    FastLanguageModel.for_inference(model)
    model.eval()
    tokenizer.padding_side='left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id=tokenizer.eos_token_id
    device=next(model.parameters()).device
    retried=0
    salvaged=0
    with out.open('a',encoding='utf8') as f:
        for start in tqdm(range(0,len(pending),a.batch_size),desc='PMN Gemma summaries',unit='batch'):
            chunk=pending[start:start+a.batch_size]
            prompt_texts=[tokenizer.apply_chat_template(
                [{'role':'user','content':summary_prompt(c['history_items'])}],
                tokenize=False,add_generation_prompt=True) for c in chunk]
            inputs=tokenizer(prompt_texts,return_tensors='pt',padding=True,add_special_tokens=False).to(device)
            width=int(inputs['input_ids'].shape[-1])
            if width+a.max_new_tokens>a.max_seq_length:
                raise ValueError(f'Gemma input length {width} exceeds context {a.max_seq_length} with output {a.max_new_tokens}')
            with torch.inference_mode():
                generated=model.generate(**inputs,max_new_tokens=a.max_new_tokens,do_sample=False,
                                         pad_token_id=tokenizer.pad_token_id,
                                         eos_token_id=tokenizer.eos_token_id)
            raws=tokenizer.batch_decode(generated[:,width:],skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
            for c,prompt_text,raw in zip(chunk,prompt_texts,raws):
                salvaged_now=False
                try:
                    summary=decode_summary(raw)
                except Exception:
                    # Most failures are outputs cut off by the token budget: retry this one
                    # profile alone (greedy, same prompt) with a larger budget.
                    single=tokenizer([prompt_text],return_tensors='pt',add_special_tokens=False).to(device)
                    w1=int(single['input_ids'].shape[-1])
                    with torch.inference_mode():
                        regen=model.generate(**single,max_new_tokens=a.retry_max_new_tokens,do_sample=False,
                                             pad_token_id=tokenizer.pad_token_id,
                                             eos_token_id=tokenizer.eos_token_id)
                    raw=tokenizer.decode(regen[0,w1:],skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)
                    try:
                        summary=decode_summary(raw)
                        retried+=1
                    except Exception as e:
                        try:
                            summary=salvage_truncated_summary(raw)
                            salvaged+=1
                            salvaged_now=True
                            print(f"[PMN-SALVAGE] {c['role']}:{c['user_id']} kept text up to last full sentence",
                                  flush=True)
                        except Exception:
                            raise ValueError(f"Summary parse failed for {c['role']}:{c['user_id']} even with "
                                             f"{a.retry_max_new_tokens} tokens; raw={raw[:400]!r}; "
                                             'completed records can be reused with --resume') from e
                row={'schema_version':'pmn_summary_v1','summary_version':SUMMARY_VERSION,
                     'summary_backend':'gemma','summary_model':a.model,'dataset':ds,
                     'role':c['role'],'user_id':c['user_id'],
                     'summary_key':c['summary_key'],'history_item_ids':c['history_ids'],
                     'history_size':h,'preference_summary':summary,
                     'summary_salvaged_from_truncation':salvaged_now,
                     'source_how':c['source_how'],'terminal_window_index':c['terminal_window_index'],
                     'terminal_train_item_id':c.get('terminal_train_item_id')}
                f.write(json.dumps(row,ensure_ascii=False)+'\n')
            f.flush()
    print('[DONE]',out,f'retried_with_larger_budget={retried}', f'salvaged={salvaged}',flush=True)

if __name__ == "__main__":
    main()
