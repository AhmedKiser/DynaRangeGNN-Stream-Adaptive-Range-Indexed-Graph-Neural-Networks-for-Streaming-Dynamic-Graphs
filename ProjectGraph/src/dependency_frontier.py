"""
Adaptive Dependency Frontier Index (ADRI-DFI)
"""

from collections import defaultdict
import networkx as nx
import time


class DependencyFrontierIndex:

    def __init__(self, graph):

        self.graph = graph

        # Current graph version
        self.graph_version = 0

        # node -> embedding version
        self.embedding_version = defaultdict(int)

        # node -> query frequency
        self.query_frequency = defaultdict(int)

        # dirty frontier
        self.dirty_frontier = {}

        self.alpha = 0.40
        self.beta = 0.30
        self.gamma = 0.20
        self.delta = 0.10

        self.threshold = 0.50

    def increase_graph_version(self):

        self.graph_version += 1

    def record_query(self, node):

        self.query_frequency[node] += 1


    def normalize_degree(self, node):

        if self.graph.number_of_nodes() == 0:
            return 0

        max_degree = max(dict(self.graph.degree()).values())

        if max_degree == 0:
            return 0

        return self.graph.degree(node) / max_degree


    def normalize_query(self, node):

        if len(self.query_frequency) == 0:
            return 0

        maximum = max(self.query_frequency.values())

        if maximum == 0:
            return 0

        return self.query_frequency[node] / maximum

    def normalize_cache(self, node):

        version_gap = self.graph_version - self.embedding_version[node]

        if self.graph_version == 0:
            return 0

        return version_gap / self.graph_version

    def normalize_attribute(self,
                            old_value,
                            new_value):

        if old_value is None:
            return 1

        return abs(new_value-old_value)/100.0


    def impact_score(self,
                     structural,
                     attribute,
                     query,
                     cache):

        return (

                self.alpha * structural +

                self.beta * attribute +

                self.gamma * query +

                self.delta * cache

        )

    def mark_dirty(self,
                   node,
                   score,
                   reason):

        if score < self.threshold:
            return

        self.dirty_frontier[node] = {

            "impact_score": round(score, 3),

            "reason": reason,

            "graph_version": self.graph_version,

            "time": time.time()

        }

    def get_candidates(self,
                       updated_nodes):

        candidates = set()

        for node in updated_nodes:

            candidates.add(node)

            for nbr in self.graph.neighbors(node):

                candidates.add(nbr)

        return candidates

    def edge_insert(self,
                    u,
                    v):

        self.increase_graph_version()

        candidates = self.get_candidates([u, v])

        for node in candidates:

            structural = self.normalize_degree(node)

            attribute = 0

            query = self.normalize_query(node)

            cache = self.normalize_cache(node)

            score = self.impact_score(
                structural,
                attribute,
                query,
                cache
            )

            self.mark_dirty(
                node,
                score,
                "EDGE_INSERT"
            )

    def edge_delete(self,
                    u,
                    v):

        self.increase_graph_version()

        candidates = self.get_candidates([u, v])

        for node in candidates:

            structural = self.normalize_degree(node)

            attribute = 0

            query = self.normalize_query(node)

            cache = self.normalize_cache(node)

            score = self.impact_score(
                structural,
                attribute,
                query,
                cache
            )

            self.mark_dirty(
                node,
                score,
                "EDGE_DELETE"
            )

    def feature_update(self,
                   node,
                   old_value,
                   new_value):

        self.increase_graph_version()

        # Updated node is ALWAYS dirty

        self.dirty_frontier[node] = {
            "impact_score": 1.0,
            "reason": "ATTRIBUTE_UPDATE",
            "graph_version": self.graph_version,
            "time": time.time()
        }
        # Evaluate only neighbors
        for nbr in self.graph.neighbors(node):
            structural = self.normalize_degree(nbr)
            attribute = self.normalize_attribute(
                old_value,
                new_value
            )
            query = self.normalize_query(nbr)
            cache = self.normalize_cache(nbr)
            score = self.impact_score(
                structural,
                attribute,
                query,
                cache
            )
            self.mark_dirty(
                nbr,
                score,
                "NEIGHBOR_ATTRIBUTE_EFFECT"
            )

    def clear(self):

        self.dirty_frontier.clear()

    def print_dirty_frontier(self):

        print()

        print("========== DIRTY FRONTIER ==========")

        if len(self.dirty_frontier) == 0:

            print("No dirty nodes.")

            return

        for node, info in self.dirty_frontier.items():

            print()

            print(f"Node : {node}")

            print(f"Impact : {info['impact_score']}")

            print(f"Reason : {info['reason']}")

            print(f"Version : {info['graph_version']}")

        print()

    def update_embedding_version(self,
                                 node):

        self.embedding_version[node] = self.graph_version