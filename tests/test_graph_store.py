from dynadeltagnn.graph_store import GraphStore


def test_graph_add_delete_and_hash() -> None:
    g = GraphStore.from_edges(num_nodes=4, edges=[(0, 1), (1, 2)])

    assert g.has_edge(0, 1)
    assert g.has_edge(1, 0)
    assert not g.has_edge(0, 3)

    added = g.add_edge(2, 3)
    assert added
    assert g.has_edge(2, 3)
    assert 3 in g.adj[2]
    assert 2 in g.adj[3]

    added_again = g.add_edge(2, 3)
    assert not added_again

    deleted = g.delete_edge(1, 2)
    assert deleted
    assert not g.has_edge(1, 2)

    g.check_consistency()
