import networkx as nx


class DynamicGraphStorage:
    """
    Dynamic Graph Storage Module
    ----------------------------
    Maintains the evolving graph structure.

    Supported Operations
    --------------------
    • Insert Node
    • Delete Node
    • Insert Edge
    • Delete Edge
    • Update Node Attributes
    • Retrieve Neighbors
    • Degree Query
    """

    def __init__(self, graph: nx.Graph):
        self.graph = graph

    # Node Operations

    def insert_node(self, node_id, **attributes):

        if node_id in self.graph:
            print(f"[INFO] Node {node_id} already exists.")
            return

        self.graph.add_node(node_id, **attributes)

        print(f"[SUCCESS] Node {node_id} inserted.")

    def delete_node(self, node_id):

        if node_id not in self.graph:
            print(f"[ERROR] Node {node_id} not found.")
            return

        self.graph.remove_node(node_id)

        print(f"[SUCCESS] Node {node_id} deleted.")

    # Edge Operations
 
    def insert_edge(self, u, v):

        self.graph.add_edge(u, v)

        print(f"[SUCCESS] Edge ({u}, {v}) inserted.")

    def delete_edge(self, u, v):

        if self.graph.has_edge(u, v):

            self.graph.remove_edge(u, v)

            print(f"[SUCCESS] Edge ({u}, {v}) deleted.")

        else:

            print("[ERROR] Edge not found.")

    # Attribute Operations

    def update_attribute(self, node_id, key, value):

        if node_id not in self.graph:

            print("[ERROR] Node not found.")

            return

        self.graph.nodes[node_id][key] = value

        print(
            f"[SUCCESS] Node {node_id} : {key} updated to {value}"
        )

    # Query Operations

    def neighbors(self, node_id):

        if node_id not in self.graph:

            return []

        return list(self.graph.neighbors(node_id))

    def degree(self, node_id):

        if node_id not in self.graph:

            return 0

        return self.graph.degree(node_id)

    # Graph Information

    def graph_info(self):

        print("\nGraph Information")

        print("----------------------")

        print("Nodes :", self.graph.number_of_nodes())

        print("Edges :", self.graph.number_of_edges())