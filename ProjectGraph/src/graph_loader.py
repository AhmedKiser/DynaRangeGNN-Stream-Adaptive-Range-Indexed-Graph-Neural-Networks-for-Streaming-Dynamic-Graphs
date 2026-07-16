import networkx as nx


class GraphLoader:

    def __init__(self, path):
        self.path = path

    def load_graph(self):
        graph = nx.read_edgelist(
            self.path,
            nodetype=int
        )

        return graph