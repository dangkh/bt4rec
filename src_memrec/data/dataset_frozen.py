"""MemRec-only JSON adapter. No re-split, descriptions, reviews, or ID coercion."""
import json
from pathlib import Path


def read_json(path):
    with open(path, encoding='utf-8') as stream:
        return json.load(stream)


def ids(value):
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    return [str(x) for x in value]


def user_list(path):
    value = read_json(path)
    if isinstance(value, dict):
        value = value.get('users', value.get('user_ids', list(value)))
    if not isinstance(value, list):
        raise ValueError(f'{path}: expected a user ID list')
    value = ids(value)
    if not value or len(value) != len(set(value)):
        raise ValueError(f'{path}: empty or duplicate user IDs')
    return value


def text(value):
    if isinstance(value, list):
        return ' | '.join(text(x) for x in value)
    if isinstance(value, dict):
        return ' | '.join(f'{k}: {text(v)}' for k, v in value.items())
    return '' if value is None else str(value)


class FrozenRecDataset:
    frozen_protocol = True

    def __init__(self, data_dir, config):
        self.data_path = Path(data_dir)
        self.name = config.get('dataset', self.data_path.name)
        self.seed = int(config.get('seed', 42))
        self.history_size = int(config.get('history_size', 10))
        if self.history_size < 1:
            raise ValueError('history_size must be positive')
        def path(key, default):
            # Explicit paths are relative to the invocation directory.
            return Path(config[key]) if config.get(key) else self.data_path / default
        seq = read_json(path('sequences_file', 'user_sequences.json'))
        neg = read_json(path('negatives_file', 'user_negatives.json'))
        metadata = read_json(path('items_file', 'items.json'))
        if not isinstance(seq, dict) or not isinstance(neg, dict):
            raise ValueError('sequences and negatives must be user-keyed JSON objects')
        if isinstance(metadata, list):
            catalog = {}
            for row in metadata:
                raw = row.get('item_id', row.get('id', row.get('asin')))
                if raw is None or str(raw) in catalog:
                    raise ValueError('Missing/duplicate item ID in items.json')
                catalog[str(raw)] = row
            metadata = catalog
        if not isinstance(metadata, dict):
            raise ValueError('items.json must be an item-keyed object or a list of item records')
        nt = int(config.get('num_train_users', 300))
        ne = int(config.get('num_test_users', 300))
        train = user_list(path('train_users_file', f'train_users_{nt}.json'))
        evaluate = user_list(path('eval_users_file', f'eval_users_{ne}.json'))
        if nt <= 0 or ne <= 0 or len(train) < nt or len(evaluate) < ne:
            raise ValueError(f'Not enough users: train={len(train)} need={nt}, eval={len(evaluate)} need={ne}')
        train, evaluate = train[:nt], evaluate[:ne]
        overlap = set(train) & set(evaluate)
        if overlap and not config.get('allow_overlap', False):
            raise ValueError(f'Train/eval overlap: {len(overlap)} users; set allow_overlap explicitly')
        self.overlap_count = len(overlap)
        missing = (set(train) | set(evaluate)) - set(seq)
        if missing:
            raise ValueError(f'Users missing from sequences: {sorted(missing)[:10]}')
        selected = set(train) | set(evaluate)
        # Internal IDs follow source sequence order; frozen lists define execution order.
        self.user_id_map = {u: i for i, u in enumerate(u for u in seq if u in selected)}
        self.raw_user_ids = {v: k for k, v in self.user_id_map.items()}
        self.train_user_ids = [self.user_id_map[u] for u in train]
        self.eval_user_ids = [self.user_id_map[u] for u in evaluate]
        self.graph_train_users = set(self.train_user_ids)
        candidate_path = path('candidate_file', 'user_candidates_testpool1000_seed42.json')
        if candidate_path.suffix == '.jsonl':
            rows = [json.loads(line) for line in candidate_path.read_text().splitlines() if line.strip()]
            cache = {str(row.get('user_id', row.get('uid'))): row for row in rows}
            if len(cache) != len(rows):
                raise ValueError('Duplicate users in candidate JSONL')
        else:
            cache = read_json(candidate_path)
            if isinstance(cache, dict) and isinstance(cache.get('users'), dict):
                cache = cache['users']
        if not isinstance(cache, dict):
            raise ValueError('Candidate cache must be a user-keyed object')
        self.item_id_map = {}
        def map_items(values):
            result = []
            for raw in ids(values):
                if raw not in self.item_id_map:
                    self.item_id_map[raw] = len(self.item_id_map)
                result.append(self.item_id_map[raw])
            return result
        self.train_data, self.full_train_data, self.valid_data, self.test_data = {}, {}, {}, {}
        self.training_negatives = {}
        for raw, u in self.user_id_map.items():
            row = seq[raw]
            self.full_train_data[u] = map_items(row.get('train', []))
            self.train_data[u] = self.full_train_data[u][-self.history_size:]
            for name, dest in [('val', self.valid_data), ('test', self.test_data)]:
                values = ids(row.get(name))
                if len(values) > 1:
                    raise ValueError(f'user={raw}: this adapter requires single-target leave-one-out {name}')
                if values:
                    dest[u] = map_items(values)[0]
            # Warm-up negatives come from declared negative pools, never test feedback.
            nr = neg.get(raw, {})
            values = nr.get('val_neg', nr.get('test_neg', [])) if isinstance(nr, dict) else nr
            self.training_negatives[u] = list(dict.fromkeys(map_items(values)))
        self.frozen_candidates = {}
        expected = config.get('n_eval_candidates')
        for raw in evaluate:
            u = self.user_id_map[raw]
            row = cache.get(raw)
            if row is None or u not in self.test_data:
                raise ValueError(f'user={raw}: missing candidate row or test target')
            if isinstance(row, list):
                values = row
            else:
                values = row.get('candidates', row.get('candidate_ids', row.get('candidate_item_ids')))
                cached_target = next((row[k] for k in ('target', 'targets', 'ground_truth', 'ground_truth_item_ids', 'target_item_ids', 'target_items', 'test') if k in row), None)
                if cached_target is not None and ids(cached_target) != ids(seq[raw].get('test')):
                    raise ValueError(f'user={raw}: cached target differs from sequence test target')
            candidates = map_items(values)
            if not candidates or len(candidates) != len(set(candidates)):
                raise ValueError(f'user={raw}: empty or duplicate candidates')
            if self.test_data[u] not in candidates:
                raise ValueError(f'user={raw}: target absent from candidates')
            if expected is not None and len(candidates) != int(expected):
                raise ValueError(f'user={raw}: expected {expected} candidates, got {len(candidates)}')
            self.frozen_candidates[u] = candidates
        self.raw_item_ids = {v: k for k, v in self.item_id_map.items()}
        # Explicit allowlist: descriptions/reviews never enter downstream state.
        self.item_metadata = {}
        self.missing_metadata_ids = []
        for raw, internal_id in self.item_id_map.items():
            meta = metadata.get(raw)
            if not isinstance(meta, dict):
                meta = {}
                self.missing_metadata_ids.append(raw)
            self.item_metadata[internal_id] = {
                'title': text(meta.get('title')).strip() or 'Unknown item',
                'category': text(meta.get('category') or meta.get('categories')).strip() or 'Unknown',
            }
        if self.missing_metadata_ids:
            print(f'[WARN] Missing metadata for {len(self.missing_metadata_ids)} items; '
                  f'examples={self.missing_metadata_ids[:10]}. '
                  'IDs and candidate order are preserved.')
        self.n_users, self.n_items = len(self.user_id_map), len(self.item_id_map)
        self.instructions = self.reviews = self.ranked_lists = None
        self.n_interactions = sum(map(len, self.full_train_data.values())) + len(self.valid_data) + len(self.test_data)

    def load_item_metadata(self, item_text_mode='category'):
        if item_text_mode != 'category':
            raise ValueError('Frozen MemRec accepts title/category only')

    def get_user_train_items(self, user_id):
        return list(self.train_data.get(user_id, []))

    def get_user_history(self, user_id, split='train'):
        # Deliberately never append validation/test targets.
        return self.get_user_train_items(user_id)

    def get_user_all_items(self, user_id):
        return list(self.full_train_data.get(user_id, []))

    def history_text(self, user_id):
        return json.dumps([self.item_metadata[i] for i in self.get_user_train_items(user_id)], ensure_ascii=False)

    def get_stats(self):
        ntrain = sum(len(self.full_train_data[u]) for u in self.train_user_ids)
        return {'n_users': self.n_users, 'n_items': self.n_items,
                'n_interactions': self.n_interactions, 'n_train_users': len(self.train_user_ids),
                'n_train_interactions': ntrain,
                'density': self.n_interactions / max(1, self.n_users*self.n_items)}

    def __repr__(self):
        return f'FrozenRecDataset(train={len(self.train_user_ids)}, eval={len(self.eval_user_ids)}, overlap={self.overlap_count})'
