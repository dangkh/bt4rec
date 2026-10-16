SCHEMA_VERSION = "bt4rec_controlled_behavior_source_ablation_v1"


RECENT3_PROMPT_VERSION = "recent3_semantic_qwen_clabel_json_list_v1"


RANK_PROMPT_VERSION = "bt4rec_controlled_additional_how_list_v1"


AB_MODES = (
    'latest_only', 'latest_how', 'random_how', 'global_continuation',
    'pmn', 'tree', 'oracle', 'recent3', 'recent3_tree',
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
