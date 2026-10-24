SCHEMA_VERSION = "bt4rec_controlled_behavior_source_ablation_v1"


RECENT3_PROMPT_VERSION = "recent3_semantic_qwen_clabel_json_list_v1"


RANK_PROMPT_VERSION = "comemtree_list_v2_plain_native_latest"


# Ranking conditions.  Section 2 of the prompt = WHAT (observed semantic behavior),
# section 3 = HOW (mechanism / scope / trajectory).  All conditions share the same
# history, frozen candidates, output format and parser.
#
#   Main comparison
#     native             plain ranking prompt: history + candidates, no WHAT/HOW framing
#     latest_what_only   native prompt + short summary (WHAT) of the latest window
#     latest_how_only    no WHAT,                              HOW of the latest window
#     tree               WHAT of the latest window           + Tree-predicted next HOW   (CoMemTree)
#     tree_v2            WHAT of the latest 3 windows        + Tree-predicted next HOW   (CoMemTree v2)
#
#   Ablation of the HOW source (section 3 replaced, section 2 as noted)
#     random_how_only    no WHAT, HOW of a random TRAIN source user
#     global_continuation latest WHAT + most frequent continuation at the Tree root (same for all users)
#     pmn                latest WHAT + HOW of preference-matched neighbour users
#     oracle             latest WHAT + HOW derived from the held-out test item (upper bound, diagnostic)
MAIN_MODES = ('native', 'latest_what_only', 'latest_how_only', 'tree', 'tree_v2')
ABLATION_MODES = ('random_how_only', 'global_continuation', 'pmn', 'oracle')
AB_MODES = MAIN_MODES + ABLATION_MODES

# Modes whose prompt contains the Tree-predicted HOW.
TREE_MODES = frozenset(('tree', 'tree_v2'))

# Reference condition for `differences` / paired bootstrap in summary.json.
REFERENCE_MODE = 'latest_what_only'


PMN_VERSION = 'pmn_preference_summary_train_prefix_gemma_v4_train_cache_metadata'


# Modes that need a random TRAIN-source HOW.
RANDOM_SOURCE_MODES = frozenset(('random_how_only',))
