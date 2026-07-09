import torch

from dynadeltagnn import DynaDeltaEngine, EdgeAdd, EdgeDelete, FeatureUpdate, TwoLayerGraphSAGEMean
from dynadeltagnn.synthetic import classroom_graph, star_graph


def make_engine():
    torch.manual_seed(7)
    graph, x = classroom_graph()
    model = TwoLayerGraphSAGEMean(input_dim=3, hidden_dim=4, output_dim=4)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return DynaDeltaEngine(model=model, graph=graph, x=x)


def test_one_feature_update_matches_full_recompute() -> None:
    engine = make_engine()
    new_c = torch.tensor([0.78, 0.30, 0.40], dtype=torch.float64)
    stats = engine.process_event_batch([FeatureUpdate(node=2, new_value=new_c)])

    engine.assert_cache_invariants()
    max_error = engine.compare_to_full_recompute(atol=1e-10)

    assert stats.feature_nodes == 1
    assert max_error < 1e-10


def test_one_edge_insertion_matches_full_recompute() -> None:
    engine = make_engine()
    stats = engine.process_event_batch([EdgeAdd(1, 4)])  # B-E

    engine.assert_cache_invariants()
    max_error = engine.compare_to_full_recompute(atol=1e-10)

    assert stats.added_edges == 1
    assert max_error < 1e-10


def test_one_edge_deletion_matches_full_recompute() -> None:
    engine = make_engine()
    stats = engine.process_event_batch([EdgeDelete(0, 1)])  # A-B

    engine.assert_cache_invariants()
    max_error = engine.compare_to_full_recompute(atol=1e-10)

    assert stats.removed_edges == 1
    assert max_error < 1e-10


def test_mixed_batch_matches_full_recompute() -> None:
    engine = make_engine()
    new_c = torch.tensor([0.78, 0.30, 0.40], dtype=torch.float64)

    stats = engine.process_event_batch([
        FeatureUpdate(node=2, new_value=new_c),
        EdgeAdd(1, 4),       # B-E
        EdgeDelete(0, 1),    # A-B
    ])

    engine.assert_cache_invariants()
    max_error = engine.compare_to_full_recompute(atol=1e-10)

    assert stats.feature_nodes == 1
    assert stats.added_edges == 1
    assert stats.removed_edges == 1
    assert max_error < 1e-10


def test_add_then_delete_same_edge_has_no_net_topology_change() -> None:
    engine = make_engine()

    stats = engine.process_event_batch([
        EdgeAdd(1, 4),
        EdgeDelete(1, 4),
    ])

    engine.assert_cache_invariants()
    max_error = engine.compare_to_full_recompute(atol=1e-10)

    assert stats.added_edges == 0
    assert stats.removed_edges == 0
    assert max_error < 1e-10


def test_high_degree_hub_feature_update_matches_full_recompute() -> None:
    torch.manual_seed(7)
    graph, x = star_graph(num_leaves=1000, feature_dim=3)
    model = TwoLayerGraphSAGEMean(input_dim=3, hidden_dim=8, output_dim=4)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    engine = DynaDeltaEngine(model=model, graph=graph, x=x)

    new_hub_feature = x[0].clone()
    new_hub_feature[0] += 0.5

    stats = engine.process_event_batch([FeatureUpdate(node=0, new_value=new_hub_feature)])

    engine.assert_cache_invariants()
    max_error = engine.compare_to_full_recompute(atol=1e-10)

    assert stats.feature_nodes == 1
    assert max_error < 1e-10
    assert stats.layer1_dirty > 500
