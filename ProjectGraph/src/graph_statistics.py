import networkx as nx


class GraphStatistics:

    @staticmethod
    def show(graph):

        print("=" * 40)

        print("Graph Statistics")

        print("=" * 40)

        print(f"Nodes              : {graph.number_of_nodes()}")

        print(f"Edges              : {graph.number_of_edges()}")

        print(f"Average Degree     : {sum(dict(graph.degree()).values()) / graph.number_of_nodes():.2f}")

        print(f"Density            : {nx.density(graph):.6f}")

        print(f"Connected          : {nx.is_connected(graph)}")

        print("=" * 40)