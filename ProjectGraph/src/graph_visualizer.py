import matplotlib.pyplot as plt
import networkx as nx


class GraphVisualizer:

    @staticmethod
    def draw(graph):

        plt.figure(figsize=(10,10))

        nx.draw_networkx(
            graph,
            node_size=10,
            with_labels=False
        )

        plt.title("Facebook Graph")

        plt.show()