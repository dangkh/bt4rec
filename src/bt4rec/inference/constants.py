SCHEMA_VERSION = "bt4rec_controlled_behavior_source_ablation_v1"


RECENT3_PROMPT_VERSION = "recent3_semantic_qwen_clabel_json_list_v1"


RANK_PROMPT_VERSION = "bt4rec_controlled_additional_how_list_v1"


AB_MODES = (
    'latest_only', 'latest_how', 'latest_how_only', 'random_how', 'random_only', 'global_continuation',
    'pmn', 'tree', 'tree_how_only', 'oracle', 'recent3', 'recent3_tree',
    'tree_original', 'native',
    'stay_only', 'tree_stay', 'tree_original_stay', 'recent3_tree_stay',
)


PMN_VERSION = 'pmn_preference_summary_train_prefix_gemma_v4_train_cache_metadata'


STAY_PROMPT_VERSIONS = {
    'v2': 'last_behavior_persistence_v2_what_how',  # default: latest WHAT + latest HOW + P(stay)
    'v1': 'last_state_order1_persistence_v1',      # legacy wording (inference_bt4rec_ablation.py)
}
STAY_PROMPT_VERSION = STAY_PROMPT_VERSIONS['v2']


STAY_MODES = frozenset(('stay_only', 'tree_stay', 'tree_original_stay', 'recent3_tree_stay'))


# Modes that need a random TRAIN-source HOW.  random_only = random_how WITHOUT the
# target user's latest WHAT (section 2 'Not provided.'); both use the SAME sampled
# HOW per (seed, user), so the two runs are directly paired.
RANDOM_SOURCE_MODES = frozenset(('random_how', 'random_only'))


# Latest-behavior decomposition of the target user's last window:
#   latest_only     = WHAT only  (section 2 = latest WHAT,  section 3 'Not provided.')
#   latest_how_only = HOW only   (section 2 'Not provided.', section 3 = latest HOW)
#   latest_how      = WHAT + HOW (both sections filled)
LATEST_HOW_MODES = frozenset(('latest_how', 'latest_how_only'))
