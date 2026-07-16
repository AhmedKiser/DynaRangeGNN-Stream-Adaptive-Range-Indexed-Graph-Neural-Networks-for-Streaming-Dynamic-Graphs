from graph_loader import GraphLoader
from graph_statistics import GraphStatistics
from graph_visualizer import GraphVisualizer
from graph_storage import DynamicGraphStorage

loader = GraphLoader("C:/Users/User/Desktop/CSE 511/ProjectGraph/ProjectGraph/data/facebook_combined.txt")
graph = loader.load_graph()

#GraphStatistics.show(graph)

#GraphVisualizer.draw(graph)

storage = DynamicGraphStorage(graph)

print("\nInitial Graph")
storage.graph_info()

# Insert Node

storage.insert_node(
    5000,
    risk=75,
    age=22,
    degree_score=10
)

# Insert Edge

storage.insert_edge(5000, 0)

# Update Attribute

storage.update_attribute(
    5000,
    "risk",
    90
)

# Degree

print(
    "\nDegree :",
    storage.degree(5000)
)

# Neighbor

print(
    "\nNeighbors :",
    storage.neighbors(5000)
)

# Delete Edge

storage.delete_edge(
    5000,
    0
)

# Delete Node

storage.delete_node(
    5000
)

#Final Graph

print("\nFinal Graph")

storage.graph_info()



from attribute_index import AdaptiveAttributeIndex

# Create Some Sample Attributes

storage.update_attribute(0, "risk", 78)
storage.update_attribute(1, "risk", 55)
storage.update_attribute(2, "risk", 91)
storage.update_attribute(3, "risk", 73)
storage.update_attribute(4, "risk", 82)

# Build Index

index = AdaptiveAttributeIndex()

index.build(graph, "risk")

# Print
index.print_index("risk")

# Range Query

result = index.range_query(
    "risk",
    70,
    90
)

print("\nRange Query : Risk = [70,90]")

for value, node in result:
    print(f"Node {node}  Risk={value}")

print("\nUpdating Node 0...")

old = 78

storage.update_attribute(
    0,
    "risk",
    95
)

index.update(
    0,
    "risk",
    old,
    95
)

print("\nQuery After Update")

result = index.range_query("risk", 70, 90)

for value, node in result:
    print(node, value)


from dependency_frontier import DependencyFrontierIndex

frontier = DependencyFrontierIndex(graph)
print("Dependency Frontier Demo")

# simulate user queries
frontier.record_query(0)
frontier.record_query(0)
frontier.record_query(0)

frontier.record_query(3)


print("\nInsert Edge (0,500)")

storage.insert_edge(0,500)

frontier.edge_insert(0,500)

frontier.print_dirty_frontier()


frontier.clear()

print("\nFeature Update")

old = 73

storage.update_attribute(
    3,
    "risk",
    88
)

frontier.feature_update(
    3,
    old,
    88
)

frontier.print_dirty_frontier()