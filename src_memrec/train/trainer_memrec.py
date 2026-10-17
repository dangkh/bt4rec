"""
MemRec Trainer v2 (evaluation only, no training)
Supports eval_feedback modes: gt, random, none
Supports parallel evaluation for speedup
"""
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
from tqdm import tqdm
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from src.models import LLMClient, MemRecAgent
from src.data import RecDataset

# Thread-safe random number generator
_thread_local = threading.local()


class MemRecTrainer:
    """MemRec Trainer (evaluation only, reranking)"""
    
    def __init__(
        self,
        model,  # None for MemRec
        dataset: RecDataset,
        config: Dict,
        device: torch.device
    ):
        """
        Initialize trainer
        
        Args:
            model: None (MemRec doesn't need it)
            dataset: RecDataset (needs item_metadata loaded)
            config: Configuration dictionary
            device: Device (for API consistency)
        """
        if getattr(dataset, "frozen_protocol", False) and config.get("load_memory_path"):
            raise ValueError("Legacy memory imports are disabled: rebuild from TRAIN with title/category")
        if getattr(dataset, "frozen_protocol", False):
            m = config.get('memrec', {})
            if m.get('reranker_mode') != 'llm' or m.get('reranker_output') != 'list':
                raise ValueError('Frozen MemRec requires llm reranker and C-label list output')
            if config.get('item_text_mode', 'category') != 'category':
                raise ValueError('Descriptions are forbidden in frozen MemRec')
        self.dataset = dataset
        self.config = config
        self.device = device
        self.save_conversations = config.get('save_llm_conversations', False)
        self.conversation_file = config.get('conversation_file', None)
        
        # instruction_mode controls what (if anything) is passed as the
        # reranker's "instruction" context:
        #   'file'    - the dataset's .instruction file (LLM-synthesized persona/intent), default
        #   'history' - a plain list of the user's recent train item titles (no .instruction file needed)
        #   'none'    - no instruction context at all
        # disable_instructions=True is kept as a shorthand for instruction_mode='none'.
        self.instruction_mode = config.get('instruction_mode', 'none' if config.get('disable_instructions', False) else 'file')

        if self.instruction_mode == 'file':
            if dataset.instructions is None and hasattr(dataset, 'load_instructions'):
                try:
                    dataset.load_instructions()
                    if dataset.instructions:
                        print(f"  ✓ Loaded instructions for {len(dataset.instructions)} users")
                except Exception as e:
                    print(f"  ⚠ Could not load instructions: {e}")
        else:
            print(f"  ⓘ instruction_mode='{self.instruction_mode}' - not loading .instruction file")

        # Extract configuration
        self.n_eval_candidates = config.get('n_eval_candidates', 10)
        self.n_eval_users = config.get('n_eval_users', None)
        self.eval_user_list = config.get('eval_user_list', None)  # Path to JSON file with fixed user list

        # Fixed eval candidates (optional): {user_id: {"val_neg": [...], "test_neg": [...]}}.
        # When set, evaluation uses these pre-sampled negatives instead of
        # MemRec's own random sampling, so results stay comparable with other
        # models evaluated on the same fixed candidate sets.
        self.fixed_negatives = None
        fixed_negatives_path = config.get('fixed_negatives_path')
        if fixed_negatives_path:
            fixed_negatives_path = Path(fixed_negatives_path)
            if not fixed_negatives_path.is_absolute():
                fixed_negatives_path = Path(__file__).parent.parent.parent / fixed_negatives_path
            with open(fixed_negatives_path, 'r') as f:
                raw = json.load(f)
            self.fixed_negatives = {int(uid): v for uid, v in raw.items()}
            print(f"  ✓ Loaded fixed eval negatives for {len(self.fixed_negatives)} users from {fixed_negatives_path}")
        self.topk = config.get('topk', [1, 3, 5])
        self.metrics = config.get('metrics', ['Hit', 'NDCG'])
        
        # MemRec configuration
        memrec_config = config.get('memrec', {})
        self.k = memrec_config.get('k', 16)
        self.tau = memrec_config.get('tau_tokens', memrec_config.get('tau', 1800))
        self.n_facets = memrec_config.get('n_facets', 7)
        self.temperature = memrec_config.get('temperature', 0.0)
        self.max_tokens = memrec_config.get('max_tokens', 1000)
        self.mix_min_users = memrec_config.get('mix_min_users', 4)
        self.mix_min_items = memrec_config.get('mix_min_items', 6)
        
        # Ablation study controls
        self.enable_stage_r = memrec_config.get('enable_stage_r', True)  # Enable Stage-R (memory retrieval)
        self.enable_stage_w = memrec_config.get('enable_stage_w', True)  # Enable Stage-W (memory writing)
        
        # Reranker configuration (v2 new)
        self.reranker_mode = memrec_config.get('reranker_mode', 'vector')  # 'vector' or 'llm'
        # LLM reranker output: 'score' (original, per-item 0-1 scores) or 'list' (ranked id list,
        # same output format as AgenticRec_CFmemory's llm_ranking)
        self.reranker_output = memrec_config.get('reranker_output', 'score')
        # Candidates listed in the Stage-R prompt: 'first10' (original), 'all', or 'none'
        self.stage_r_candidates = memrec_config.get('stage_r_candidates', 'first10')
        
        # Pruner configuration (v2 new)
        pruner_config = memrec_config.get('pruner', {})
        self.pruner_mode = pruner_config.get('mode', 'hybrid_rule')  # 'hybrid_rule' or 'learned_mlp'
        self.pruner_checkpoint = pruner_config.get('checkpoint', None)
        
        # Write configuration
        write_config = memrec_config.get('write', {})
        self.fanout_cap = write_config.get('fanout_cap', 8)
        
        # eval_feedback mode
        if config.get('eval_feedback', 'none') != 'none':
            raise ValueError('Evaluation feedback is forbidden; use eval_feedback: none')
        self.eval_feedback = 'none'  # gt, random, none
        
        # Warm-up configuration (v2 new)
        warmup_config = config.get('warmup', {})
        self.warmup_enabled = warmup_config.get('enabled', False)
        self.warmup_rounds = warmup_config.get('rounds', 1)
        
        # Debug settings
        self.debug = config.get('debug', False)
        self.debug_log_file = config.get('debug_log_file', None)
        
        # LLM configuration
        provider_config = config.get('provider', {})
        provider_name = provider_config.get('name', 'azure_openai')  # Default to azure_openai
        self.llm_model = provider_config.get('model', config.get('llm_model', 'gpt-4o-mini'))
        self.api_endpoint = provider_config.get('endpoint', config.get('api_endpoint'))
        self.api_key = provider_config.get('api_key', config.get('api_key'))
        # Allow provider to override api_version
        self.api_version = provider_config.get('api_version', config.get('api_version', '2024-08-01-preview'))
        
        # Initialize LLM client (for Stage-R and Stage-W)
        self.llm_client = LLMClient(
            api_endpoint=self.api_endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
            model=self.llm_model,
            provider_name=provider_name,
            save_conversations=self.save_conversations,
            conversation_log_path=self.conversation_file
        )
        
        # If reranker_mode is 'llm', create separate LLMClient for Stage-ReRank.
        # Reuses the main `provider` block by default (so e.g. a fully
        # open-source local_hf run doesn't silently fall back to Azure);
        # an optional `reranker_provider` config block can override it.
        if self.reranker_mode == "llm":
            reranker_provider_config = config.get('reranker_provider', provider_config)
            reranker_llm_model = reranker_provider_config.get('model', config.get('llm_model', 'gpt-4o-mini'))
            reranker_provider_name = reranker_provider_config.get('name', provider_name)
            reranker_endpoint = reranker_provider_config.get('endpoint', config.get('api_endpoint'))
            reranker_api_key = reranker_provider_config.get('api_key', config.get('api_key'))

            if reranker_provider_name == provider_name and reranker_llm_model == self.llm_model:
                # Same provider+model as Stage-R/W - reuse the client instead of
                # loading a second copy (matters most for local_hf: avoids
                # double GPU memory for the same open-source model).
                self.reranker_llm_client = self.llm_client
            else:
                self.reranker_llm_client = LLMClient(
                    api_endpoint=reranker_endpoint,
                    api_key=reranker_api_key,
                    api_version=self.api_version,
                    model=reranker_llm_model,
                    provider_name=reranker_provider_name,
                    save_conversations=self.save_conversations,
                    conversation_log_path=self.conversation_file
                )

            print(f"\nLLM Client Configuration:")
            print(f"  Stage-R & Stage-W: {self.llm_model} @ {self.api_endpoint}")
            print(f"  Stage-ReRank: {reranker_llm_model} @ {reranker_endpoint}")
        else:
            self.reranker_llm_client = None
        
        # Load item metadata
        print("\nLoading item metadata for MemRec...")
        self.dataset.load_item_metadata(item_text_mode='category')
        
        # Initialize MemRec Agent (v2: three stages)
        print("\nInitializing MemRec Agent v2...")
        self.agent = MemRecAgent(
            dataset=self.dataset,
            llm_client=self.llm_client,
            k=self.k,
            tau=self.tau,
            n_facets=self.n_facets,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            mix_min_users=self.mix_min_users,
            mix_min_items=self.mix_min_items,
            fanout_cap=self.fanout_cap,
            reranker_mode=self.reranker_mode,
            reranker_output=self.reranker_output,
            stage_r_candidates=self.stage_r_candidates,
            pruner_mode=self.pruner_mode,
            pruner_checkpoint=self.pruner_checkpoint,
            enable_stage_r=self.enable_stage_r,  # Ablation: control Stage-R
            reranker_llm_client=self.reranker_llm_client,  # Separate LLMClient for reranker
            debug=self.debug
        )

        # Reuse memory saved by a previous run's test() (results/.../memory.jsonl)
        # instead of rebuilding it from scratch via the warm-up phase.
        load_memory_path = config.get('load_memory_path')
        if load_memory_path:
            self.agent.storage.load_from_jsonl(load_memory_path)

        print(f"\nInitialized MemRec Trainer v2")
        print(f"  LLM Model: {self.llm_model}")
        print(f"  Pruner: mode={self.pruner_mode}, k={self.k} (mix: ≥{self.mix_min_users} users, ≥{self.mix_min_items} items)")
        if self.pruner_checkpoint:
            print(f"    Checkpoint: {self.pruner_checkpoint}")
        print(f"  Reranker: {self.reranker_mode}")
        print(f"  Token budget τ: {self.tau}")
        print(f"  N facets: {self.n_facets}")
        print(f"  Evaluation candidates: {self.n_eval_candidates}")
        print(f"  Eval feedback mode: {self.eval_feedback}")
        if self.warmup_enabled:
            print(f"  🔥 Training: ENABLED")
        if self.n_eval_users:
            print(f"  Eval users (sampled): {self.n_eval_users}")
        if self.debug:
            print(f"  🔍 Debug mode: ENABLED")
    
    def _init_debug_logger(self, save_dir):
        """Initialize debug log file"""
        if not self.debug:
            return
        
        if self.debug_log_file:
            log_file = Path(self.debug_log_file)
        else:
            log_file = Path(save_dir) / "debug.txt" if save_dir else Path("debug_memrec.txt")
        
        log_file.parent.mkdir(parents=True, exist_ok=True)
        self.debug_log_file = str(log_file)
        
        # Clear existing file
        with open(self.debug_log_file, 'w', encoding='utf-8') as f:
            f.write(f"MemRec Debug Log\n")
            f.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*100 + "\n\n")
        
        print(f"  🔍 Debug log: {self.debug_log_file}")
    
    def _log_debug(self, message, level="INFO"):
        """Write to debug log"""
        if not self.debug or not self.debug_log_file:
            return
        
        timestamp = time.strftime('%H:%M:%S')
        log_line = f"[{timestamp}] [{level}] {message}\n"
        
        with open(self.debug_log_file, 'a', encoding='utf-8') as f:
            f.write(log_line)
    
    def _log_user_detail(self, user_id, target_item, candidates, details, ranked_items, pos):
        """Log detailed information for a single user"""
        if not self.debug:
            return
        
        self._log_debug("\n" + "="*100)
        self._log_debug(f"USER {user_id}")
        self._log_debug("="*100)
        
        # Basic information
        self._log_debug(f"\nTarget item: {target_item}")
        if target_item in self.dataset.item_metadata:
            meta = self.dataset.item_metadata[target_item]
            self._log_debug(f"  Title: {meta.get('title', 'N/A')[:100]}")
        
        self._log_debug(f"\nCandidates ({len(candidates)} items): {candidates}")
        
        # Pruned subgraph
        if 'pruned_subgraph' in details:
            ps = details['pruned_subgraph']
            self._log_debug(f"\nPruned neighbors: {len(ps['neighbors'])} ({ps['n_items']} items, {ps['n_users']} users)")
            self._log_debug("Selected neighbors:")
            for i, n in enumerate(ps['neighbors'], 1):
                self._log_debug(f"  [{i}] {n['type'].upper()}-{n['id']}: score={n['score']:.4f}")
        
        # Packed context
        if 'packed_context' in details:
            cb = details['packed_context']
            self._log_debug(f"\nContext: {cb['n_neighbors']} neighbors, ~{cb['estimated_tokens']} tokens")
        
        # Facets
        self._log_debug(f"\nFacets ({len(details['facets'])} generated):")
        for i, facet in enumerate(details['facets'], 1):
            self._log_debug(f"  [{i}] {facet.get('facet', facet.get('text', 'N/A'))}")
            self._log_debug(f"      Confidence: {facet.get('confidence', 0):.2f}")
        
        # Evidence
        evidence = details.get('evidence', [])
        if evidence:
            self._log_debug(f"\nEvidence ({len(evidence)} entries):")
            for i, ev in enumerate(evidence[:10], 1):  # Only show first 10
                self._log_debug(f"  [{i}] Facet {ev['facet_idx']} ← {ev['neighbor_id']} (weight={ev['weight']:.2f})")
        
        # Rerank scores (v2: replaces candidate_notes)
        rerank_scores = details.get('rerank_scores', [])
        if rerank_scores:
            self._log_debug(f"\nCandidate scoring ({len(rerank_scores)} items):")
            for i, score_entry in enumerate(rerank_scores, 1):
                cid = score_entry['item_id']
                score = score_entry['score']
                rationale = score_entry.get('rationale', 'N/A')
                marker = " ⭐ TARGET" if cid == target_item else ""
                self._log_debug(f"  [{i}] Item {cid}: score={score:.3f}{marker}")
                self._log_debug(f"      {rationale[:100]}")
        
        # Ranking result
        self._log_debug(f"\nRanking:")
        self._log_debug(f"  Original: {candidates}")
        self._log_debug(f"  Reranked: {ranked_items}")
        self._log_debug(f"  Target position: {pos+1}/{len(candidates)}")
        
        if pos < 1:
            self._log_debug("  ✓ Hit@1!")
        elif pos < 3:
            self._log_debug("  ✓ Hit@3!")
        elif pos < 5:
            self._log_debug("  ✓ Hit@5!")
    
    def _build_eval_candidates(self, user_id: int, target_item: int, split: str, rng=None) -> List[int]:
        """
        Build the evaluation candidate set (target + negatives) for a user.

        Uses self.fixed_negatives (target + pre-sampled negatives, shared
        across models/experiments for a fair comparison) when available for
        this user/split; otherwise falls back to MemRec's own uniform-random
        sampling from all non-interacted items.

        Returns None if no candidates could be built (not enough negatives).
        """
        if self.fixed_negatives is not None and user_id in self.fixed_negatives:
            negatives = self.fixed_negatives[user_id].get(f"{split}_neg", [])
            if negatives:
                candidates = [target_item] + list(negatives)
                if rng is not None:
                    rng.shuffle(candidates)
                else:
                    random.shuffle(candidates)
                return candidates

        all_items = set(range(self.dataset.n_items))
        user_history = set(self.dataset.get_user_train_items(user_id))
        negative_pool = list(all_items - user_history - {target_item})

        if len(negative_pool) < self.n_eval_candidates - 1:
            return None

        if rng is not None:
            negative_items = rng.choice(negative_pool, size=self.n_eval_candidates - 1, replace=False).tolist()
        else:
            negative_items = random.sample(negative_pool, self.n_eval_candidates - 1)

        candidates = [target_item] + negative_items
        if rng is not None:
            rng.shuffle(candidates)
        else:
            random.shuffle(candidates)
        return candidates

    def _get_user_instruction(self, user_id: int) -> str:
        """
        Build the "instruction" text passed to the reranker, per
        self.instruction_mode (see __init__ for the mode descriptions).
        """
        if self.instruction_mode == 'file':
            if self.dataset.instructions and user_id in self.dataset.instructions:
                return self.dataset.instructions[user_id].get('instruction')
            return None

        if self.instruction_mode == 'history':
            item_ids = self.dataset.get_user_train_items(user_id)[-10:]
            if not item_ids or not self.dataset.item_metadata:
                return None
            titles = [
                self.dataset.item_metadata[i]['title']
                for i in item_ids
                if i in self.dataset.item_metadata and self.dataset.item_metadata[i].get('title')
            ]
            if not titles:
                return None
            return "Recently interacted items (oldest to newest): " + "; ".join(titles)

        return None

    def _evaluate_single_user(
        self,
        user_id: int,
        target_item: int,
        target_data: Dict,
        lock: threading.Lock = None,
        split: str = 'test'
    ) -> Tuple[int, Dict]:
        """
        Evaluate single user (for parallel processing)

        Returns:
            (position, timings_dict)
        """
        # Get thread-local random number generator
        if not hasattr(_thread_local, 'rng'):
            _thread_local.rng = np.random.RandomState(seed=hash(threading.current_thread().ident) % (2**32))

        # Build candidate set (fixed negatives if configured, else random)
        candidates = self._build_eval_candidates(user_id, target_item, split, rng=_thread_local.rng)
        if candidates is None:
            return (-1, {})  # Skip

        instruction = self._get_user_instruction(user_id)

        # Rerank
        try:
            ranked_items, details = self.agent.rerank(
                user_id=user_id,
                candidates=candidates,
                instruction=instruction,  # Pass instruction to reranker
                return_details=True,
                debug_logger=None  # No debug output in parallel mode
            )
            
            # Extract timings
            timings = details.get('timings', {})
            stage_r_time = timings.get('stage_r', 0.0)
            stage_rr_time = timings.get('stage_rr', 0.0)
            
            # Stage-W (only if enabled)
            stage_w_time = 0.0
            patches_applied = 0
            if self.enable_stage_w and self.eval_feedback != 'none':
                feedback = self._generate_feedback(
                    user_id=user_id,
                    target_item=target_item,
                    ranked_items=ranked_items,
                    mode=self.eval_feedback
                )
                
                if feedback:
                    t0 = time.time()
                    write_result = self.agent.write(
                        user_id=user_id,
                        feedback=feedback,
                        recent_facets=details.get('facets', []),
                        pruned_subgraph=details.get('pruned_subgraph', None),
                        debug_logger=None
                    )
                    stage_w_time = time.time() - t0
                    patches_applied = write_result.get('stats', {}).get('total_applied', 0)
            
            # Calculate ranking position
            try:
                position = ranked_items.index(target_item)
            except ValueError:
                position = len(ranked_items)
            
            return (position, {
                'stage_r_time': stage_r_time,
                'stage_rr_time': stage_rr_time,
                'stage_w_time': stage_w_time,
                'patches_applied': patches_applied
            })
            
        except Exception as e:
            if lock:
                with lock:
                    print(f"\nError evaluating user {user_id}: {e}")
            else:
                print(f"\nError evaluating user {user_id}: {e}")
            return (-1, {})
    
    def evaluate(self, split: str = 'test', save_dir: str = None, parallel: bool = False, n_workers: int = 16) -> Dict:
        """
        Evaluate MemRec reranking performance
        
        Args:
            split: 'test' or 'valid'
            save_dir: Save directory (optional)
            parallel: Whether to use parallel evaluation
            n_workers: Number of parallel workers
            
        Returns:
            Metrics dictionary
        """
        if getattr(self.dataset, 'frozen_protocol', False):
            if split != 'test' or parallel:
                raise ValueError('Frozen protocol supports sequential test evaluation only')
            if save_dir:
                self.save_dir = save_dir
            return self._test_frozen()
        # Initialize debug logger (disabled in parallel mode)
        if self.debug and save_dir and not parallel:
            self._init_debug_logger(save_dir)
        # Get evaluation data
        if split == 'test':
            target_data = self.dataset.test_data
        else:
            target_data = self.dataset.valid_data
        
        # Sample users (if specified)
        # Priority: eval_user_list > n_eval_users > all users
        if self.eval_user_list:
            # Load fixed user list from JSON file
            from pathlib import Path
            user_list_path = Path(self.eval_user_list)
            if not user_list_path.is_absolute():
                # Relative to project root
                user_list_path = Path(__file__).parent.parent.parent / user_list_path
            
            print(f"\n📋 Loading fixed user list from: {user_list_path}")
            with open(user_list_path, 'r') as f:
                user_data = json.load(f)
            
            loaded_user_ids = user_data['user_ids']
            # Filter to only include users in target_data
            eval_user_ids = [uid for uid in loaded_user_ids if uid in target_data]
            print(f"  ✓ Loaded {len(loaded_user_ids)} user IDs, {len(eval_user_ids)} are in {split} set")
            
            if len(eval_user_ids) < len(loaded_user_ids):
                print(f"  ⚠ {len(loaded_user_ids) - len(eval_user_ids)} users from list not found in {split} set")
        elif self.n_eval_users and self.n_eval_users < len(target_data):
            all_user_ids = list(target_data.keys())
            random.seed(self.config.get('seed', 42))
            eval_user_ids = random.sample(all_user_ids, self.n_eval_users)
            print(f"\nSampled {self.n_eval_users} users for evaluation")
        else:
            eval_user_ids = list(target_data.keys())
        
        # ========== Training Phase ==========
        # Simulate memory construction and propagation before testing
        self._run_training(eval_user_ids, target_data, split, parallel=parallel, n_workers=n_workers)
        
        # Evaluation loop
        ranking_positions = []  # Store target item ranking positions
        n_success = 0
        
        # Stage statistics (v2: three stages)
        stage_r_times = []
        stage_rr_times = []  # Rerank
        stage_w_times = []
        stage_w_applied = []
        
        # Token usage tracking per stage
        # Record initial token statistics at evaluation start
        initial_llm_stats = self.llm_client.get_token_stats()
        initial_rr_llm_stats = self.reranker_llm_client.get_token_stats() if self.reranker_llm_client else None
        
        # For accumulating token usage per stage
        stage_r_input_tokens = []
        stage_r_output_tokens = []
        stage_rr_input_tokens = []
        stage_rr_output_tokens = []
        stage_w_input_tokens = []
        stage_w_output_tokens = []
        
        print(f"\nEvaluating MemRec on {split} set ({len(eval_user_ids)} users)...")
        print(f"Eval feedback mode: {self.eval_feedback}")
        if parallel:
            print(f"🚀 Parallel mode: {n_workers} workers")
        
        # ========== Parallel Mode ==========
        if parallel:
            lock = threading.Lock()
            futures = []
            
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                # Submit all tasks
                for user_id in eval_user_ids:
                    target_item = target_data[user_id]
                    future = executor.submit(
                        self._evaluate_single_user,
                        user_id, target_item, target_data, lock, split
                    )
                    futures.append(future)
                
                # Collect results
                with tqdm(total=len(futures), desc=f"MemRec {split}") as pbar:
                    for future in as_completed(futures):
                        pos, timings = future.result()
                        if pos >= 0:  # Valid result
                            ranking_positions.append(pos)
                            n_success += 1
                            stage_r_times.append(timings.get('stage_r_time', 0.0))
                            stage_rr_times.append(timings.get('stage_rr_time', 0.0))
                            stage_w_times.append(timings.get('stage_w_time', 0.0))
                            stage_w_applied.append(timings.get('patches_applied', 0))
                        pbar.update(1)
            
            # In parallel mode, cannot record token statistics per user separately
            # Calculate overall token usage after evaluation ends, then divide by number of users to get average
            final_llm_stats = self.llm_client.get_token_stats()
            final_rr_llm_stats = self.reranker_llm_client.get_token_stats() if self.reranker_llm_client else None
            
            # Calculate overall token usage increment
            total_stage_r_input = final_llm_stats['total_input_tokens'] - initial_llm_stats['total_input_tokens']
            total_stage_r_output = final_llm_stats['total_output_tokens'] - initial_llm_stats['total_output_tokens']
            
            if self.reranker_mode == "llm" and final_rr_llm_stats:
                total_stage_rr_input = final_rr_llm_stats['total_input_tokens'] - initial_rr_llm_stats['total_input_tokens']
                total_stage_rr_output = final_rr_llm_stats['total_output_tokens'] - initial_rr_llm_stats['total_output_tokens']
            else:
                total_stage_rr_input = 0
                total_stage_rr_output = 0
            
            # Calculate averages (Note: assumes all users executed Stage-R and Stage-ReRank)
            n_users_with_rerank = len(stage_r_times)
            if n_users_with_rerank > 0:
                # Note: In parallel mode, Stage-R and Stage-ReRank token usage may include other operations
                # So averages may not be precise, but can serve as reference
                avg_stage_r_input = total_stage_r_input / n_users_with_rerank
                avg_stage_r_output = total_stage_r_output / n_users_with_rerank
                avg_stage_rr_input = total_stage_rr_input / n_users_with_rerank if self.reranker_mode == "llm" else 0
                avg_stage_rr_output = total_stage_rr_output / n_users_with_rerank if self.reranker_mode == "llm" else 0
                
                stage_r_input_tokens = [avg_stage_r_input] * n_users_with_rerank
                stage_r_output_tokens = [avg_stage_r_output] * n_users_with_rerank
                stage_rr_input_tokens = [avg_stage_rr_input] * n_users_with_rerank
                stage_rr_output_tokens = [avg_stage_rr_output] * n_users_with_rerank
            
            # Stage-W token usage needs separate calculation (not all users execute Stage-W)
            if len(stage_w_times) > 0:
                # In parallel mode, cannot precisely calculate Stage-W token usage
                # Use approximate value: assume Stage-W token usage similar to Stage-R
                # Or can set to 0 to indicate cannot precisely count
                stage_w_input_tokens = [0] * len(stage_w_times)  # Cannot precisely count
                stage_w_output_tokens = [0] * len(stage_w_times)  # Cannot precisely count
        
        # ========== Serial Mode (original logic, debug support retained) ==========
        else:
            for user_id in tqdm(eval_user_ids, desc=f"MemRec {split}"):
                target_item = target_data[user_id]

                # Build candidate set (fixed negatives if configured, else random)
                candidates = self._build_eval_candidates(user_id, target_item, split)
                if candidates is None:
                    print(f"Warning: Not enough negative samples for user {user_id}")
                    continue

                # Stage-R: Reranking
                try:
                    if self.debug:
                        self._log_debug(f"\n{'='*100}")
                        self._log_debug(f"Processing user {user_id}")
                        self._log_debug(f"{'='*100}")
                    
                    instruction = self._get_user_instruction(user_id)
                    if self.debug and instruction:
                        self._log_debug(f"\n💬 User Instruction: {instruction[:100]}...")
                    
                    # Record token statistics before rerank
                    before_rerank_llm_stats = self.llm_client.get_token_stats()
                    before_rerank_rr_llm_stats = self.reranker_llm_client.get_token_stats() if self.reranker_llm_client else None
                    
                    t0 = time.time()
                    ranked_items, details = self.agent.rerank(
                        user_id=user_id,
                        candidates=candidates,
                        instruction=instruction,  # Pass instruction to reranker
                        return_details=True,  # Need details for Stage-W
                        debug_logger=self._log_debug if self.debug else None
                    )
                    total_time = time.time() - t0
                    
                    # Record token statistics after rerank
                    after_rerank_llm_stats = self.llm_client.get_token_stats()
                    after_rerank_rr_llm_stats = self.reranker_llm_client.get_token_stats() if self.reranker_llm_client else None
                    
                    # Calculate token usage
                    # Note: Stage-R uses self.llm_client, Stage-ReRank uses self.reranker_llm_client (if reranker_mode is 'llm')
                    if self.reranker_mode == "llm" and self.reranker_llm_client:
                        # Stage-R and Stage-ReRank use different clients, can distinguish
                        # Stage-R uses llm_client
                        stage_r_input = after_rerank_llm_stats['total_input_tokens'] - before_rerank_llm_stats['total_input_tokens']
                        stage_r_output = after_rerank_llm_stats['total_output_tokens'] - before_rerank_llm_stats['total_output_tokens']
                        # Stage-ReRank uses reranker_llm_client
                        stage_rr_input = after_rerank_rr_llm_stats['total_input_tokens'] - before_rerank_rr_llm_stats['total_input_tokens']
                        stage_rr_output = after_rerank_rr_llm_stats['total_output_tokens'] - before_rerank_rr_llm_stats['total_output_tokens']
                    else:
                        # Vector mode: only Stage-R uses llm_client, Stage-ReRank doesn't use LLM
                        stage_r_input = after_rerank_llm_stats['total_input_tokens'] - before_rerank_llm_stats['total_input_tokens']
                        stage_r_output = after_rerank_llm_stats['total_output_tokens'] - before_rerank_llm_stats['total_output_tokens']
                        stage_rr_input = 0
                        stage_rr_output = 0
                    
                    # v2: Extract stage-by-stage timings
                    timings = details.get('timings', {})
                    stage_r_time = timings.get('stage_r', 0.0)
                    stage_rr_time = timings.get('stage_rr', 0.0)
                    stage_r_times.append(stage_r_time)
                    stage_rr_times.append(stage_rr_time)
                    
                    # Record token usage
                    stage_r_input_tokens.append(stage_r_input)
                    stage_r_output_tokens.append(stage_r_output)
                    stage_rr_input_tokens.append(stage_rr_input)
                    stage_rr_output_tokens.append(stage_rr_output)
                    
                    if self.debug:
                        self._log_debug(f"\nStage-R: {stage_r_time*1000:.1f}ms")
                        self._log_debug(f"Stage-ReRank ({self.reranker_mode}): {stage_rr_time*1000:.1f}ms")
                        self._log_debug(f"Total rerank time: {total_time*1000:.1f}ms")
                    
                    # Find target's ranking position (0-indexed)
                    if target_item in ranked_items:
                        pos = ranked_items.index(target_item)
                        ranking_positions.append(pos)
                        n_success += 1
                        
                        # Debug log
                        if self.debug:
                            self._log_user_detail(user_id, target_item, candidates, details, ranked_items, pos)
                    else:
                        # Should not happen
                        ranking_positions.append(len(candidates))
                        
                        if self.debug:
                            self._log_debug(f"WARNING: Target item {target_item} not in ranked_items for user {user_id}", level="WARNING")
                    
                    # Stage-W: Write (based on eval_feedback mode)
                    try:
                        if self.eval_feedback != 'none':
                            feedback = self._generate_feedback(
                                user_id=user_id,
                                target_item=target_item,
                                ranked_items=ranked_items,
                                mode=self.eval_feedback
                            )
                            
                            if feedback:
                                if self.debug:
                                    self._log_debug(f"\nStage-W: feedback={feedback}")
                                
                                # Record token statistics before Stage-W
                                before_stage_w_stats = self.llm_client.get_token_stats()
                                
                                t0 = time.time()
                                write_result = self.agent.write(
                                    user_id=user_id,
                                    feedback=feedback,
                                    recent_facets=details.get('facets', []),
                                    pruned_subgraph=details.get('pruned_subgraph', None),
                                    debug_logger=self._log_debug if self.debug else None
                                )
                                stage_w_time = time.time() - t0
                                
                                # Record token statistics after Stage-W
                                after_stage_w_stats = self.llm_client.get_token_stats()
                                
                                # Calculate Stage-W token usage
                                stage_w_input = after_stage_w_stats['total_input_tokens'] - before_stage_w_stats['total_input_tokens']
                                stage_w_output = after_stage_w_stats['total_output_tokens'] - before_stage_w_stats['total_output_tokens']
                                
                                stage_w_times.append(stage_w_time)
                                stage_w_applied.append(write_result['stats']['total_applied'])
                                stage_w_input_tokens.append(stage_w_input)
                                stage_w_output_tokens.append(stage_w_output)
                                
                                if self.debug:
                                    self._log_debug(f"Stage-W completed in {stage_w_time*1000:.1f}ms")
                                    self._log_debug(f"Applied patches: {write_result['stats']}")
                    except Exception as e_write:
                        # Stage-W errors don't affect evaluation results
                        if self.debug:
                            self._log_debug(f"WARNING: Stage-W failed for user {user_id}: {e_write}", level="WARNING")
                    
                except Exception as e:
                    # Entire evaluation flow errors
                    print(f"\nError evaluating user {user_id}: {e}")
                    ranking_positions.append(len(candidates))
                    
                    if self.debug:
                        import traceback
                        self._log_debug(f"ERROR for user {user_id}: {e}", level="ERROR")
                        self._log_debug(traceback.format_exc(), level="ERROR")
                
                continue
        
        # Calculate metrics
        metrics = self._compute_metrics(ranking_positions, len(eval_user_ids))
        
        # Add Stage statistics (v2: three stages)
        metrics['n_stage_r_calls'] = len(stage_r_times)
        metrics['avg_stage_r_time_ms'] = np.mean(stage_r_times) * 1000 if stage_r_times else 0.0
        metrics['n_stage_rr_calls'] = len(stage_rr_times)
        metrics['avg_stage_rr_time_ms'] = np.mean(stage_rr_times) * 1000 if stage_rr_times else 0.0
        metrics['reranker_mode'] = self.reranker_mode
        metrics['pruner_mode'] = self.pruner_mode
        metrics['n_stage_w_calls'] = len(stage_w_times)
        metrics['avg_stage_w_time_ms'] = np.mean(stage_w_times) * 1000 if stage_w_times else 0.0
        metrics['avg_patches_applied'] = np.mean(stage_w_applied) if stage_w_applied else 0.0
        
        # Print results
        print(f"\n{'='*80}")
        print(f"MemRec v2 Evaluation Results ({split} set)")
        print(f"{'='*80}")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        
        print(f"\nStage timings (avg):")
        print(f"  Stage-R: {metrics['avg_stage_r_time_ms']:.1f} ms")
        print(f"  Stage-ReRank ({self.reranker_mode}): {metrics['avg_stage_rr_time_ms']:.1f} ms")
        if metrics['n_stage_w_calls'] > 0:
            print(f"  Stage-W: {metrics['avg_stage_w_time_ms']:.1f} ms")
        
        # Stage Token Usage statistics
        if len(stage_r_input_tokens) > 0:
            avg_stage_r_input = np.mean(stage_r_input_tokens)
            avg_stage_r_output = np.mean(stage_r_output_tokens)
            avg_stage_rr_input = np.mean(stage_rr_input_tokens) if len(stage_rr_input_tokens) > 0 else 0
            avg_stage_rr_output = np.mean(stage_rr_output_tokens) if len(stage_rr_output_tokens) > 0 else 0
            avg_stage_w_input = np.mean(stage_w_input_tokens) if len(stage_w_input_tokens) > 0 else 0
            avg_stage_w_output = np.mean(stage_w_output_tokens) if len(stage_w_output_tokens) > 0 else 0
            
            print(f"\nStage Token Usage:")
            print(f"  Stage-R:  Input ~{avg_stage_r_input:.0f} | Output ~{avg_stage_r_output:.0f}")
            if self.reranker_mode == "llm":
                print(f"  Stage-RR: Input ~{avg_stage_rr_input:.0f} | Output ~{avg_stage_rr_output:.0f}")
            else:
                print(f"  Stage-RR: Input ~0 | Output ~0 (vector mode)")
            if metrics['n_stage_w_calls'] > 0:
                print(f"  Stage-W:  Input ~{avg_stage_w_input:.0f} | Output ~{avg_stage_w_output:.0f}")
        
        # Token usage statistics (overall)
        token_stats = self.llm_client.get_token_stats()
        if token_stats['total_requests'] > 0:
            metrics['llm_token_stats'] = token_stats
            print(f"\nLLM Token Usage Statistics:")
            print(f"  Total requests: {token_stats['total_requests']}")
            print(f"  Total input tokens: {token_stats['total_input_tokens']}")
            print(f"  Total output tokens: {token_stats['total_output_tokens']}")
            print(f"  Total tokens: {token_stats['total_tokens']}")
            print(f"  Avg input tokens per request: {token_stats['avg_input_tokens']:.1f}")
            print(f"  Avg output tokens per request: {token_stats['avg_output_tokens']:.1f}")
        
        # If reranker_llm_client is used, also show its statistics
        if self.reranker_llm_client:
            rr_token_stats = self.reranker_llm_client.get_token_stats()
            if rr_token_stats['total_requests'] > 0:
                print(f"\nReranker LLM Token Usage Statistics:")
                print(f"  Total requests: {rr_token_stats['total_requests']}")
                print(f"  Total input tokens: {rr_token_stats['total_input_tokens']}")
                print(f"  Total output tokens: {rr_token_stats['total_output_tokens']}")
                print(f"  Total tokens: {rr_token_stats['total_tokens']}")
                print(f"  Avg input tokens per request: {rr_token_stats['avg_input_tokens']:.1f}")
                print(f"  Avg output tokens per request: {rr_token_stats['avg_output_tokens']:.1f}")
        
        # Storage statistics
        storage_stats = self.agent.get_stats()
        print(f"\nStorage statistics:")
        for k, v in storage_stats['storage'].items():
            print(f"  {k}: {v}")
        
        # Debug summary
        if self.debug and self.debug_log_file:
            self._log_debug("\n" + "="*100)
            self._log_debug("EVALUATION SUMMARY")
            self._log_debug("="*100)
            for k, v in metrics.items():
                self._log_debug(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
            self._log_debug(f"Storage: {storage_stats['storage']}")
            if token_stats['total_requests'] > 0:
                self._log_debug(f"Token Usage: {token_stats}")
            print(f"\n✓ Debug log saved to: {self.debug_log_file}")
        
        return metrics
    
    def _construct_candidates(
        self,
        user_id: int,
        target_item: int,
        history: list[int]
    ) -> list[int]:
        """
        Construct candidate set: target + negatives
        
        Args:
            user_id: User ID
            target_item: Target item
            history: User interaction history
            
        Returns:
            Candidate item list (shuffled)
        """
        # Get negative sample pool (exclude history from pre-computed negatives)
        user_positives = set(history + [target_item])
        negative_pool = [item for item in self.dataset.user_negatives.get(user_id, []) 
                         if item not in user_positives]
        
        if len(negative_pool) < self.n_eval_candidates - 1:
            # Not enough negative samples, use all available negatives
            negative_items = negative_pool
        else:
            # Random sampling
            negative_items = random.sample(negative_pool, self.n_eval_candidates - 1)
        
        # Construct candidate set and shuffle
        candidates = [target_item] + negative_items
        random.shuffle(candidates)
        
        return candidates
    
    def _warmup_single_user(self, user_id, warmup_round, target_offset, split, lock=None):
        """
        Execute training for a single user (for parallel processing)
        
        Returns:
            bool: Whether memory was successfully updated
        """
        # Get user's complete interaction history
        all_interactions = self.dataset.get_user_history(user_id, split)
        
        # Check if history is long enough
        if len(all_interactions) <= target_offset:
            return False
        
        # Calculate history and target
        target_idx = -(target_offset + 1)
        history = all_interactions[:target_idx]
        target_item = all_interactions[target_idx]
        
        instruction = self._get_user_instruction(user_id)

        # Construct candidate set
        candidates = self._construct_candidates(
            user_id=user_id,
            target_item=target_item,
            history=history
        )
        
        try:
            # Three-stage recommendation
            ranked_items, details = self.agent.rerank(
                user_id=user_id,
                candidates=candidates,
                instruction=instruction,
                debug_logger=None  # Parallel mode doesn't support debug
            )
            
            # Check target position
            if target_item not in ranked_items:
                return False
            
            target_pos = ranked_items.index(target_item)
            
            # Generate feedback and update memory
            if target_pos < 10:  # top-10 triggers Stage-W
                feedback = {
                    'action': 'CLICK',
                    'item_id': target_item,
                    'position': target_pos
                }
                
                # Stage-W: Write memory
                write_result = self.agent.write(
                    user_id=user_id,
                    feedback=feedback,
                    recent_facets=details.get('facets', []),
                    pruned_subgraph=details.get('pruned_subgraph', None),
                    debug_logger=None  # Parallel mode doesn't support debug
                )
                
                return True
            
            return False
            
        except Exception as e:
            return False
    
    def _run_training(
        self,
        eval_user_ids: list[int],
        target_data: dict,
        split: str = 'test',
        parallel: bool = False,
        n_workers: int = 16
    ):
        """
        Execute Training phase: simulate memory construction using sliding window
        
        Args:
            eval_user_ids: List of evaluation user IDs
            target_data: Test data (user_id -> target_item)
            split: Dataset split ('test' or 'valid')
            parallel: Whether to use parallel mode
            n_workers: Number of parallel workers
        """
        if not self.warmup_enabled:
            return
        
        print(f"\n{'='*80}")
        print(f"🔥 Training MemRec")
        print(f"{'='*80}")
        if parallel:
            print(f"🚀 Parallel mode: {n_workers} workers")
        
        # Execute training for each user
        for warmup_round in range(self.warmup_rounds):
            # Calculate target position for current round
            target_offset = self.warmup_rounds - warmup_round
            
            n_updates = 0  # Count how many users updated memory in this round
            
            # ========== Parallel Mode ==========
            if parallel:
                lock = threading.Lock()
                futures = []
                
                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    # Submit all tasks
                    for user_id in eval_user_ids:
                        future = executor.submit(
                            self._warmup_single_user,
                            user_id, warmup_round, target_offset, split, lock
                        )
                        futures.append(future)
                    
                    # Collect results
                    with tqdm(total=len(futures), desc=f"MemRec Training") as pbar:
                        for future in as_completed(futures):
                            success = future.result()
                            if success:
                                n_updates += 1
                            pbar.update(1)
            
            # ========== Serial Mode (debug support retained) ==========
            else:
                for user_id in tqdm(eval_user_ids, desc=f"MemRec Training"):
                    # Get user's complete interaction history
                    all_interactions = self.dataset.get_user_history(user_id, split)
                    
                    # Check if history is long enough
                    # target_offset=1 requires at least 2 items (T-1 + history)
                    if len(all_interactions) <= target_offset:
                        continue  # Not enough history, skip this user
                    
                    # Calculate history and target
                    # target_offset=1 => T-1 (2nd from end) => idx=-2
                    # target_offset=2 => T-2 (3rd from end) => idx=-3
                    target_idx = -(target_offset + 1)
                    history = all_interactions[:target_idx]
                    target_item = all_interactions[target_idx]
                    
                    instruction = self._get_user_instruction(user_id)

                    # Construct candidate set (target + negatives)
                    candidates = self._construct_candidates(
                        user_id=user_id,
                        target_item=target_item,
                        history=history
                    )
                    
                    # Debug: Training round details
                    if self.debug:
                        self._log_debug(f"\n{'='*80}")
                        self._log_debug(f"🔥 TRAINING Round {warmup_round+1}/{self.warmup_rounds} - User {user_id}")
                        self._log_debug(f"{'='*80}")
                        self._log_debug(f"Target: T-{target_offset} (item {target_item})")
                        self._log_debug(f"History length: {len(history)}")
                        self._log_debug(f"Candidates: {len(candidates)} items")
                    
                    # Three-stage recommendation
                    try:
                        ranked_items, details = self.agent.rerank(
                            user_id=user_id,
                            candidates=candidates,
                            instruction=instruction,
                            debug_logger=self._log_debug if self.debug else None  # Pass debug_logger
                        )
                        
                        # Check target position
                        if target_item not in ranked_items:
                            if self.debug:
                                msg = f"  [Training] User {user_id}: Target {target_item} not in ranked_items!"
                                print(msg)
                                self._log_debug(msg)
                            continue
                        
                        target_pos = ranked_items.index(target_item)
                        
                        if self.debug:
                            msg = f"  [Training] User {user_id}: Target {target_item} ranked at position {target_pos+1}/{len(ranked_items)}"
                            print(msg)
                            self._log_debug(msg)
                            self._log_debug(f"\nRanking: {ranked_items[:5]}... (showing top-5)")
                        
                        # Generate feedback and update memory
                        if target_pos < 10:  # top-10 triggers Stage-W (relaxed condition to increase training success rate)
                            feedback = {
                                'action': 'CLICK',
                                'item_id': target_item,
                                'position': target_pos
                            }
                            
                            if self.debug:
                                self._log_debug(f"\n🔄 Stage-W: Updating memories...")
                            
                            # Stage-W: Write memory (extract correct parameters)
                            write_result = self.agent.write(
                                user_id=user_id,
                                feedback=feedback,
                                recent_facets=details.get('facets', []),
                                pruned_subgraph=details.get('pruned_subgraph', None),
                                debug_logger=self._log_debug if self.debug else None  # Pass debug_logger
                            )
                            
                            n_patches = write_result.get('stats', {}).get('total_applied', 0)
                            stats = write_result.get('stats', {})
                            
                            if self.debug:
                                msg = f"  [Training] User {user_id}: Stage-W applied {n_patches} patches"
                                print(msg)
                                self._log_debug(msg)
                                self._log_debug(f"  User updated: {stats.get('user_applied', 0)}")
                                self._log_debug(f"  Item updated: {stats.get('item_applied', 0)}")
                                self._log_debug(f"  Neighbors updated: {stats.get('neighbor_applied', 0)}")
                            
                            n_updates += 1
                        else:
                            if self.debug:
                                print(f"  [Training] User {user_id}: Target not in top-5, skipping Stage-W")
                    
                    except Exception as e:
                        if self.debug:
                            print(f"  [Training] Error for user {user_id}: {e}")
                            import traceback
                            traceback.print_exc()
                        continue
            
            print(f"  ✓ {n_updates}/{len(eval_user_ids)} users updated memories")
        
        print(f"\n{'='*80}")
        print(f"🔥 MemRec Training Complete!")
        print(f"{'='*80}\n")
    
    def _generate_feedback(
        self,
        user_id: int,
        target_item: int,
        ranked_items: list[int],
        mode: str
    ) -> Dict:
        """
        Generate feedback information
        
        Args:
            user_id: User ID
            target_item: Target item
            ranked_items: Sorted item list
            mode: 'gt' or 'random'
            
        Returns:
            Feedback dictionary
        """
        if mode == 'gt':
            # Ground-truth: if target item is in top-5, CLICK; otherwise skip
            if target_item in ranked_items[:5]:
                return {
                    'action': 'CLICK',
                    'item_id': target_item,
                    'position': ranked_items.index(target_item)
                }
            else:
                # Optional: record as SKIP
                return None
        
        elif mode == 'random':
            # Randomly select a candidate and randomly generate feedback
            selected_item = random.choice(ranked_items[:5])  # Only select from top-5
            action = random.choice(['CLICK', 'SKIP'])
            return {
                'action': action,
                'item_id': selected_item,
                'position': ranked_items.index(selected_item)
            }
        
        return None
    
    def _compute_metrics(self, ranking_positions: list, n_users: int) -> Dict:
        """
        Calculate evaluation metrics
        
        Args:
            ranking_positions: List of ranking positions (0-indexed)
            n_users: Number of users
            
        Returns:
            Metrics dictionary
        """
        metrics = {}
        
        if len(ranking_positions) == 0:
            # No valid results
            for k in self.topk:
                for metric in self.metrics:
                    metrics[f'{metric}@{k}'] = 0.0
            # metrics['MRR'] = 0.0  # MRR disabled for consistency
            return metrics
        
        positions = np.array(ranking_positions)
        
        # Hit@K and NDCG@K
        for k in self.topk:
            if 'Hit' in self.metrics:
                hit_at_k = np.mean(positions < k)
                metrics[f'Hit@{k}'] = float(hit_at_k)
            
            if 'NDCG' in self.metrics:
                # NDCG: 1/log2(pos+2) if pos < k, else 0
                ndcg_scores = np.where(
                    positions < k,
                    1.0 / np.log2(positions + 2),
                    0.0
                )
                metrics[f'NDCG@{k}'] = float(np.mean(ndcg_scores))
        
        # Note: MRR metric is disabled for consistency with baseline models
        # # MRR (always compute)
        # mrr = np.mean(1.0 / (positions + 1))
        # metrics['MRR'] = float(mrr)
        
        return metrics
    
    def train(self, save_dir: str = None) -> Dict:
        """
        Train (MemRec doesn't need training, directly return empty results)
        
        Returns:
            Empty dictionary (MemRec has no training phase)
        """
        # MemRec doesn't need traditional training (already builds memory through training phase)
        
        # Save save_dir for use by test()
        self.save_dir = save_dir
        
        # Return empty results, actual evaluation happens in test()
        return {
            'valid': {},
            'metrics': {}
        }
    
    def test(self, parallel: bool = False, n_workers: int = 16) -> Dict:
        """Test (standard interface)"""
        print("\n" + "="*80)
        print("Testing MemRec on test set")
        print("="*80)
        
        if getattr(self.dataset, "frozen_protocol", False):
            return self._test_frozen()

        # Evaluation
        save_dir = getattr(self, 'save_dir', None)
        test_metrics = self.evaluate(split='test', save_dir=save_dir, parallel=parallel, n_workers=n_workers)
        
        # Save memory storage (optional)
        if save_dir:
            memory_path = Path(save_dir) / "memory.jsonl"
            self.agent.storage.save_to_jsonl(str(memory_path), dataset=self.dataset)
            
            # Output save information
            stats = self.agent.get_stats()
            print(f"Memory saved to {memory_path} ({stats['storage']['n_updates']} entries: {stats['storage']['n_users']} users, {stats['storage']['n_items']} items)")
            
            # If LLM conversations saving is enabled, output save information
            if self.save_conversations and self.conversation_file:
                conversation_path = Path(self.conversation_file)
                if conversation_path.exists():
                    # Count conversations
                    try:
                        with open(conversation_path, 'r') as f:
                            n_conversations = sum(1 for _ in f)
                        print(f"LLM conversations saved to {conversation_path} ({n_conversations} conversations)")
                    except:
                        print(f"LLM conversations saved to {conversation_path}")
        
        return test_metrics

    def _test_frozen(self):
        """Train-only memory construction followed by strictly read-only evaluation.

        Sequential by design: Stage-W is stateful and ordering must be reproducible.
        This path does not use legacy warm-up, random eval sampling or eval feedback.
        """
        import hashlib
        import math
        from src.memory.graph import UserItemGraph
        ds = self.dataset
        save_dir = Path(getattr(self, 'save_dir', None) or self.config.get('output_dir', 'results/memrec_frozen'))
        save_dir.mkdir(parents=True, exist_ok=True)
        if self.reranker_mode != 'llm' or self.reranker_output != 'list':
            raise ValueError('Frozen protocol requires reranker_mode=llm and reranker_output=list (C-label)')
        if getattr(self, '_frozen_completed', False):
            raise RuntimeError('Create a new trainer for a new frozen evaluation')
        self._frozen_completed = True
        original_train = ds.train_data
        warmup_records = []
        # Each round reveals one further TRAIN interaction. All train-user histories
        # are clipped to that round before graph construction, including neighbors.
        try:
            if self.warmup_enabled and self.enable_stage_w:
                rounds = max(0, int(self.warmup_rounds))
                for round_idx in range(rounds):
                    offset = rounds-round_idx
                    prefixes, targets = {}, {}
                    for u in ds.train_user_ids:
                        full = ds.full_train_data[u]
                        cut = len(full)-offset
                        prefixes[u] = full[:max(0, cut)][-ds.history_size:]
                        if cut >= 1:
                            targets[u] = full[cut]
                    ds.train_data = prefixes
                    self.agent.graph = UserItemGraph(ds)
                    for u in tqdm(ds.train_user_ids, desc=f'TRAIN memory {round_idx+1}/{rounds}', unit='user'):
                        if u not in targets:
                            continue
                        target = targets[u]
                        pool = [i for i in ds.training_negatives[u]
                                if i not in set(ds.full_train_data[u]) and i != target]
                        count = int(self.config.get('warmup', {}).get('n_candidates', self.n_eval_candidates))
                        if count < 2 or len(pool) < count-1:
                            raise ValueError(f'Train user={ds.raw_user_ids[u]}: insufficient declared negatives for {count} candidates')
                        rng = random.Random(f'{ds.seed}:{ds.raw_user_ids[u]}:{round_idx}')
                        candidates = [target] + rng.sample(pool, count-1)
                        rng.shuffle(candidates)
                        ranked, details = self.agent.rerank(
                            u, candidates, instruction=ds.history_text(u), return_details=True)
                        if len(ranked) != len(candidates) or set(ranked) != set(candidates):
                            raise ValueError('Warm-up ranking is not a candidate permutation')
                        # Preserve original MemRec warm-up trigger: target in top 10.
                        wrote = ranked.index(target) < 10
                        if wrote:
                            self.agent.write(u, {'action': 'CLICK', 'item_id': target,
                                                 'position': ranked.index(target)},
                                             details.get('facets', []),
                                             pruned_subgraph=details.get('pruned_subgraph'))
                        warmup_records.append({'user_id': ds.raw_user_ids[u], 'round': round_idx+1,
                                               'target': ds.raw_item_ids[target], 'stage_w': wrote})
        finally:
            ds.train_data = original_train
            self.agent.graph = UserItemGraph(ds)
        self.agent.evaluation_only = True
        self.agent.storage.read_only = True
        def snapshot():
            storage = self.agent.storage
            payload = {'users': storage.user_profiles, 'items': storage.item_descriptions,
                       'updates': storage.n_updates}
            return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        before = snapshot()
        calls_before = self.agent.n_stage_w_calls
        positions = []
        mapping = {'user_id_map': ds.user_id_map, 'item_id_map': ds.item_id_map}
        (save_dir/'id_maps.json').write_text(json.dumps(mapping, ensure_ascii=False, indent=2))
        (save_dir/'warmup.json').write_text(json.dumps(warmup_records, ensure_ascii=False, indent=2))
        with (save_dir/'rankings.jsonl').open('w', encoding='utf-8') as output:
            for u in tqdm(ds.eval_user_ids, desc='MemRec frozen eval', unit='user'):
                candidates = list(ds.frozen_candidates[u])
                ranked, details = self.agent.rerank(
                    u, candidates, instruction=ds.history_text(u), return_details=True)
                if len(ranked) != len(candidates) or set(ranked) != set(candidates):
                    raise ValueError('Evaluation ranking is not a candidate permutation')
                target = ds.test_data[u]  # Used only after ranking for metrics/output.
                positions.append(ranked.index(target))
                output.write(json.dumps({'user_id': ds.raw_user_ids[u],
                    'candidate_item_ids': [ds.raw_item_ids[i] for i in candidates],
                    'ranked_item_ids': [ds.raw_item_ids[i] for i in ranked],
                    'target': ds.raw_item_ids[target],
                    'stage_r_error': details.get('retrieval_bundle', {}).get('_error'),
                    'stage_r_diagnostics': details.get('retrieval_bundle', {}).get('_diagnostics', {}),
                    'diagnostics': getattr(self.agent.reranker, 'last_diagnostics', {})},
                    ensure_ascii=False)+'\n')
                output.flush()
        after = snapshot()
        if before != after or calls_before != self.agent.n_stage_w_calls:
            raise RuntimeError('Memory changed during evaluation')
        metrics = {}
        for k in self.topk:
            metrics[f'Hit@{k}'] = sum(p < k for p in positions)/len(positions)
            metrics[f'Recall@{k}'] = metrics[f'Hit@{k}']
            metrics[f'NDCG@{k}'] = sum(1/math.log2(p+2) if p < k else 0 for p in positions)/len(positions)
        metrics.update(n_eval_users=len(positions), n_stage_w_calls=0,
                       warmup_stage_w_calls=calls_before, memory_unchanged=True,
                       memory_sha256=after, train_eval_overlap=ds.overlap_count)
        (save_dir/'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        # Complete snapshot; intentionally not accepted as legacy warm-start input.
        (save_dir/'memory_frozen.json').write_text(json.dumps({
            'schema': 'memrec_frozen_title_category_train_only_v1',
            'user_profiles': self.agent.storage.user_profiles,
            'item_memories': self.agent.storage.item_descriptions,
            'memory_sha256': after}, ensure_ascii=False), encoding='utf-8')
        print(json.dumps(metrics, indent=2))
        return metrics
