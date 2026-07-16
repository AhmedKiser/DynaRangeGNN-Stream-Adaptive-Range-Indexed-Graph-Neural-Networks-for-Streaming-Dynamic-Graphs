from bisect import bisect_left, bisect_right, insort


class AdaptiveAttributeIndex:
    """
    Adaptive Attribute Index (Prototype)

    Maintains ordered indexes for node attributes.

    Supported Operations
    --------------------
    • Build Index
    • Insert
    • Delete
    • Update
    • Range Query
    • Print Index
    """

    def __init__(self):
        # attribute_name -> sorted list of (value, node_id)
        self.index = {}

    # Build Index

    def build(self, graph, attribute):

        self.index[attribute] = []

        for node, attrs in graph.nodes(data=True):

            if attribute in attrs:
                insort(
                    self.index[attribute],
                    (attrs[attribute], node)
                )

        print(f"[SUCCESS] {attribute} index built.")

    # Insert

    def insert(self, node_id, attribute, value):

        if attribute not in self.index:
            self.index[attribute] = []

        insort(
            self.index[attribute],
            (value, node_id)
        )

    # Delete

    def delete(self, node_id, attribute, value):

        if attribute not in self.index:
            return

        item = (value, node_id)

        if item in self.index[attribute]:
            self.index[attribute].remove(item)

    # Update

    def update(self, node_id, attribute, old_value, new_value):

        self.delete(node_id, attribute, old_value)

        self.insert(node_id, attribute, new_value)

    # Range Query

    def range_query(self, attribute, low, high):

        if attribute not in self.index:
            return []

        data = self.index[attribute]

        left = bisect_left(data, (low, -1))
        right = bisect_right(data, (high, float("inf")))

        return data[left:right]

    # Print Index

    def print_index(self, attribute):

        if attribute not in self.index:
            print("Index not found.")
            return

        print("\n----------------------------")
        print(f"{attribute.upper()} INDEX")
        print("----------------------------")

        for value, node in self.index[attribute]:
            print(f"{value:5}  ---> Node {node}")