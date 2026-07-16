"""
order_statistic_tree.py
------------------------
Phase 2 building block: the global order-statistic tree.

Keeps every node's retrieval key (Section 7.4 of the proposal) in sorted
order, network-wide, so the system can later answer "which nodes have a
key inside [low, high]?" quickly instead of scanning every node.

Backed by sortedcontainers.SortedList, a balanced structure that gives
amortized O(log n) insert / delete / rank / range operations -- the same
asymptotic behaviour the proposal specifies for this component (Section
11.1), without needing a hand-rolled balanced BST for a first prototype.

Node ids are assumed to be strings (matches the rest of this prototype).
"""

from sortedcontainers import SortedList

_MIN_ID = ""
_MAX_ID = "\U0010FFFF"


class GlobalOrderStatisticTree:
    def __init__(self):
        self._by_key = SortedList()   # sorted (key, node_id) tuples
        self._key_of = {}             # node_id -> current key (for O(1) lookup)

    def insert(self, node_id, key):
        if node_id in self._key_of:
            raise ValueError(f"{node_id!r} already indexed -- use update_key() instead")
        self._by_key.add((key, node_id))
        self._key_of[node_id] = key

    def delete(self, node_id):
        key = self._key_of.pop(node_id)
        self._by_key.remove((key, node_id))

    def update_key(self, node_id, new_key):
        """O(log n): delete old (key, node_id), insert new (key, node_id)."""
        old_key = self._key_of[node_id]
        if old_key == new_key:
            return
        self._by_key.remove((old_key, node_id))
        self._by_key.add((new_key, node_id))
        self._key_of[node_id] = new_key

    def key_of(self, node_id):
        return self._key_of[node_id]

    def rank(self, key):
        """Number of indexed nodes with retrieval key strictly less than `key`."""
        return self._by_key.bisect_left((key, _MIN_ID))

    def range_count(self, low, high):
        """O(log n): how many nodes have low <= key <= high."""
        lo = self._by_key.bisect_left((low, _MIN_ID))
        hi = self._by_key.bisect_right((high, _MAX_ID))
        return hi - lo

    def range_query(self, low, high):
        """O(log n + k): node ids with low <= key <= high."""
        lo = self._by_key.bisect_left((low, _MIN_ID))
        hi = self._by_key.bisect_right((high, _MAX_ID))
        return [nid for (_key, nid) in self._by_key[lo:hi]]

    def __len__(self):
        return len(self._by_key)

    def __repr__(self):
        return f"GlobalOrderStatisticTree(n={len(self)})"
