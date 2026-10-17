"""
LLM-based Reranker for Stage-ReRank
Scores candidate items based on facets/vector_profile output from Stage-R
"""
import json
from typing import Dict, List


class LLMReranker:
    """Use LLM to rerank candidate items"""
    
    def __init__(self, llm_client, output_mode: str = "score"):
        """
        Initialize LLM Reranker

        Args:
            llm_client: LLMClient instance
            output_mode: "score" (original: a 0-1 score per candidate, sorted afterwards) or
                         "list" (LLM outputs the ranked id list directly, same output format as
                         AgenticRec_CFmemory's llm_ranking; converted to descending scores)
        """
        if output_mode not in ("score", "list"):
            raise ValueError(f"Unknown reranker output_mode: {output_mode}")
        self.llm = llm_client
        self.output_mode = output_mode

    @staticmethod
    def _list_task(n: int) -> str:
        # Same instruction/output as AgenticRec_CFmemory's _build_ranking_prompt
        return f"""
**Your Task:**
Rank candidate items by relevance to the user (most relevant first).
Rank ALL {n} items. Use ONLY the exact item ids from the candidate items above — include every id exactly once, no new ids, no duplicates.

**Expected Output Format:**
Return ONLY valid JSON:
{{"ranked_item_ids": [id1, ..., id{n}], "reasoning": "1 sentence on top-3 ranking logic"}}
"""
    
    def build_rerank_prompt(
        self,
        user_id: int,
        facets: List[Dict],
        candidates: List[Dict],  # [{'id': int, 'title': str, 'tags': [...]}]
        item_mems: Dict[int, Dict] = None,  # {item_id: ItemMem}
        instruction: str = None,  # User instruction (like iAgent)
        vanilla_mode: bool = False  # Vanilla mode: no memory, only item descriptions
    ) -> List[Dict[str, str]]:
        """
        Build reranking prompt
        
        Args:
            user_id: User ID
            facets: Facets from Stage-R output (empty in vanilla mode)
            candidates: List of candidate items (with metadata)
            item_mems: Item memories (optional, empty in vanilla mode)
            instruction: User instruction/intent (optional, from dataset)
            vanilla_mode: Whether vanilla mode (no memory)
            
        Returns:
            List of messages
        """
        # Build paper-friendly single prompt
        if vanilla_mode:
            # Vanilla mode: no memory, only user instruction and item descriptions
            prompt_parts = [
                f"You are an intelligent recommendation scoring system. Your task is to evaluate how well each candidate item matches the target user's preferences.",
                f"\n**Target User:** User {user_id}"
            ]
            
            # Add user instruction/persona if available
            if instruction:
                prompt_parts.append(f"\n**User Profile:**\n{instruction}")
            else:
                prompt_parts.append(f"\n**User Profile:**\nNo specific user profile provided.")
            
            # Format candidate items with descriptions
            prompt_parts.append("\n**Candidate Items:**")
            for c in candidates:
                cid = c['id']
                title = c.get('title', f'Item {cid}')
                description = c.get('category', '')
                tags = c.get('tags', [])
                if description:
                    # Truncate long descriptions
                    if len(description) > 200:
                        description = description[:200] + "..."
                    prompt_parts.append(f"  • Item {cid}: {title}. {description}")
                elif tags:
                    tags_str = ", ".join(tags[:5])  # Limit to 5 tags
                    prompt_parts.append(f"  • Item {cid}: {title} (Tags: {tags_str})")
                else:
                    prompt_parts.append(f"  • Item {cid}: {title}")
            
            # Task description for vanilla mode
            prompt_parts.append("""
**Your Task:**
For each of the candidate items listed above, provide a relevance score between 0 and 1 that indicates how well the item matches the user's profile:
  • 1.0 = Excellent match, highly aligned with user's preferences
  • 0.5 = Moderate match, partially relevant
  • 0.0 = Poor match, not aligned with user's interests

For each item, provide a brief rationale explaining your scoring decision based on the user's profile and item characteristics.

**Expected Output Format:**
Your response should be a JSON object with a single field:
- "scores": An array of scoring objects, each containing:
  * "item_id": The item's ID (integer)
  * "score": Your relevance score between 0 and 1 (number)
  * "rationale": A brief explanation of your scoring (string)
""")
        else:
            # MemRec mode: with memory and facets
            prompt_parts = [
                f"You are an intelligent recommendation scoring system. Your task is to evaluate how well each candidate item matches the target user's preferences based on their personal memory and collaborative signals.",
                f"\n**Target User:** User {user_id}"
            ]
            
            # Add user instruction if available
            if instruction:
                prompt_parts.append(f"\n**User's Current Request:**\n{instruction}")
            
            # Format preference patterns (extracted from collaborative memories)
            prompt_parts.append("\n**User Preferences (Extracted from Collaborative Memories):**")
            prompt_parts.append("Based on collaborative signals from neighboring users and items, we have identified the following preference patterns:")
            if facets:
                for i, f in enumerate(facets[:10], 1):
                    facet_text = f.get('facet', f.get('text', 'N/A'))
                    conf = f.get('confidence', 0)
                    prompt_parts.append(f"  {i}. {facet_text} (confidence: {conf:.2f})")
            else:
                prompt_parts.append("  (No facets extracted)")
            
            # Format Item memories
            prompt_parts.append("\n**Candidate Item Memories:**")
            for c in candidates:
                cid = c['id']
                title = c.get('title', f'Item {cid}')
                memory = ""
                if item_mems and cid in item_mems:
                    memory = item_mems[cid]
                    if len(memory) > 150:
                        memory = memory[:150] + "..."
                else:
                    memory = "(No memory recorded)"
                prompt_parts.append(f"  • Item {cid} ({title}): {memory}")
            
            # Task description for MemRec mode
            prompt_parts.append("""
**Your Task:**
For each of the candidate items listed above, provide a relevance score between 0 and 1 that indicates how well the item matches the user's preferences:
  • 1.0 = Excellent match, highly aligned with user's facets and memory
  • 0.5 = Moderate match, partially relevant
  • 0.0 = Poor match, not aligned with user's interests

For each item, provide a brief rationale explaining your scoring decision based on the user's preference facets and personal memory.

**Expected Output Format:**
Your response should be a JSON object with a single field:
- "scores": An array of scoring objects, each containing:
  * "item_id": The item's ID (integer)
  * "score": Your relevance score between 0 and 1 (number)
  * "rationale": A brief explanation of your scoring (string)
""")
        
        if self.output_mode == "list":
            # Keep all MemRec context (profile/facets/item memories); swap only the scoring
            # framing in the intro and the task/output block (always the last part).
            prompt_parts[0] = prompt_parts[0].replace(
                "an intelligent recommendation scoring system. Your task is to evaluate how well each candidate item matches",
                "an intelligent recommendation ranking system. Your task is to rank the candidate items by how well they match")
            prompt_parts[-1] = self._list_task(len(candidates))

        prompt = "".join(prompt_parts)

        # Single message format
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        return messages
    
    @staticmethod
    def _list_to_scores(response: Dict, candidates: List[Dict]) -> List[Dict]:
        """
        Turn a ranked id list into [{'item_id', 'score', 'rationale'}] with strictly descending
        scores, so MemRecAgent's sort-by-score reproduces the list order. Same cleanup as
        AgenticRec_CFmemory's llm_ranking: drop hallucinated/duplicate ids, append missing
        candidates at the end in candidate order.
        """
        cand_ids = [c['id'] for c in candidates]
        by_str = {str(cid): cid for cid in cand_ids}
        seen, ranked = set(), []
        for rid in response.get('ranked_item_ids', []) or []:
            cid = by_str.get(str(rid).strip())
            if cid is not None and cid not in seen:
                seen.add(cid)
                ranked.append(cid)
        ranked += [cid for cid in cand_ids if cid not in seen]
        reasoning = str(response.get('reasoning', ''))
        n = len(ranked)
        return [{'item_id': cid, 'score': float(n - i), 'rationale': reasoning if i == 0 else ''}
                for i, cid in enumerate(ranked)]

    def get_rerank_schema(self) -> Dict:
        """Get reranking JSON schema"""
        return {
            "scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "item_id": {"type": "integer"},
                        "score": {"type": "number"},
                        "rationale": {"type": "string"}
                    },
                    "required": ["item_id", "score", "rationale"],
                    "additionalProperties": False
                }
            }
        }
    
    def rerank(
        self,
        user_id: int,
        retrieval_bundle: Dict,  # Stage-R output (empty in vanilla mode)
        candidates: List[Dict],  # [{'id', 'title', 'tags', ...}]
        item_mems: Dict[int, Dict] = None,
        instruction: str = None,  # User instruction (like iAgent)
        temperature: float = 0.0,
        max_tokens: int = 4000,
        debug_logger = None,
        vanilla_mode: bool = False  # Vanilla mode: no memory
    ) -> List[Dict]:
        """
        Rerank candidate items
        
        Args:
            user_id: User ID
            retrieval_bundle: Stage-R output {'facets', 'vector_profile', 'support_edges'} (empty in vanilla mode)
            candidates: List of candidate items
            item_mems: Item memories (optional, empty in vanilla mode)
            instruction: User instruction/intent (optional)
            temperature: LLM temperature
            max_tokens: Maximum tokens
            debug_logger: Debug logger
            vanilla_mode: Whether vanilla mode (no memory)
            
        Returns:
            [{'item_id': int, 'score': float, 'rationale': str}, ...]
        """
        if self.output_mode == "list":
            return self._rerank_clabel(user_id, retrieval_bundle, candidates, item_mems,
                                       instruction, temperature, max_tokens, debug_logger, vanilla_mode)
        # Build prompt
        messages = self.build_rerank_prompt(
            user_id=user_id,
            facets=retrieval_bundle.get('facets', []) if not vanilla_mode else [],
            candidates=candidates,
            item_mems=item_mems if not vanilla_mode else {},
            instruction=instruction,
            vanilla_mode=vanilla_mode
        )
        
        # Get schema
        properties = self.get_rerank_schema()
        if self.output_mode == "list":
            properties = {
                "ranked_item_ids": {"type": "array", "items": {"type": "integer"}},
                "reasoning": {"type": "string"}
            }

        # Call LLM (use stageRR cache)
        try:
            response = self.llm.generate_json(
                messages=messages,
                properties=properties,
                temperature=temperature,
                max_tokens=max_tokens,
                debug_logger=debug_logger
            )

            if self.output_mode == "list":
                scores = self._list_to_scores(response, candidates)
            else:
                scores = response.get('scores', [])
            
            # Debug log: record scoring results
            if debug_logger and scores:
                debug_logger(f"\n📊 Rerank Scores (Stage-ReRank):")
                for i, score_entry in enumerate(scores[:10], 1):
                    item_id = score_entry.get('item_id')
                    score = score_entry.get('score', 0)
                    rationale = score_entry.get('rationale', 'N/A')[:80]
                    debug_logger(f"  [{i}] Item {item_id}: {score:.3f} - {rationale}")
                debug_logger("")  # Empty line separator
            
            return scores
        except Exception as e:
            print(f"Error in LLM Reranker for user {user_id}: {e}")
            # Return default scores
            return [{'item_id': c['id'], 'score': 0.5, 'rationale': 'Error'} for c in candidates]


    def _rerank_clabel(self, user_id, retrieval_bundle, candidates, item_mems,
                       instruction, temperature, max_tokens, debug_logger, vanilla_mode):
        labels = {f"C{i:02d}": c['id'] for i, c in enumerate(candidates, 1)}
        rows = [{"label": label, "title": c.get("title", ""),
                 "category": c.get("category", ""),
                 "memory": "" if vanilla_mode else (item_mems or {}).get(c['id'], "")}
                for label, c in zip(labels, candidates)]
        prompt = (
            "Rank ALL candidates for the user's next interaction, most relevant first. "
            "Use only candidate labels C01..Cn, each exactly once. Never output item IDs.\n"
            "Observed TRAIN history (oldest to newest):\n" + (instruction or "") +
            "\nRetrieved preferences:\n" + json.dumps(
                [] if vanilla_mode else retrieval_bundle.get("facets", []), ensure_ascii=False) +
            "\nCandidates:\n" + json.dumps(rows, ensure_ascii=False) +
            f"\nReturn exactly {len(labels)} distinct labels in ranking. "
            "Reorder the full label set below by relevance; do not copy its input order. "
            "Do not return only the top two.\nAllowed label set: " + json.dumps(list(labels)) +
            '\nReturn one JSON object with fields "ranking" (the full ordered label array) '
            'and "reasoning" (one short sentence, at most 25 words). '
            "Use valid JSON escapes; apostrophes do not need escaping."

        )
        error = None
        try:
            response = self.llm.generate_json(
                messages=[{"role": "user", "content": prompt}],
                properties={"ranking": {"type": "array", "items": {"type": "string"}},
                            "reasoning": {"type": "string"}},
                temperature=temperature, max_tokens=max_tokens, debug_logger=debug_logger)
            raw = response.get("ranking", [])
            if not isinstance(raw, list):
                raise ValueError("ranking must be a list")
            reasoning = str(response.get("reasoning", ""))
        except Exception as exc:
            raw, reasoning, error = [], "", str(exc)
        ranked, seen, unknown, duplicate = [], set(), [], []
        for label in raw:
            if not isinstance(label, str) or label not in labels:
                unknown.append(label)
            elif label in seen:
                duplicate.append(label)
            else:
                seen.add(label)
                ranked.append(label)
        missing = [label for label in labels if label not in seen]
        self.last_diagnostics = {"raw_labels": raw, "missing": missing, "unknown": unknown,
                                 "duplicates": duplicate, "error": error, "reasoning": reasoning,
                                 "returned_count": len(raw), "expected_count": len(labels),
                                 "repair_used": bool(missing or unknown or duplicate or error),
                                 "json_diagnostics": getattr(self.llm, 'last_json_diagnostics', {})}
        ranked.extend(missing)
        return [{"item_id": labels[label], "score": float(len(ranked)-i),
                 "rationale": reasoning if i == 0 else ""} for i, label in enumerate(ranked)]
