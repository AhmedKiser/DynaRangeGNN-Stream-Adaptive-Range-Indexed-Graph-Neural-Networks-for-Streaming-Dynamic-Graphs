import torch

from dynadeltagnn.model import TwoLayerGraphSAGEMean, neighbor_sum
from dynadeltagnn.synthetic import classroom_graph


def test_static_cache_invariants() -> None:
    torch.manual_seed(7)
    graph, x = classroom_graph()

    model = TwoLayerGraphSAGEMean(input_dim=3, hidden_dim=4, output_dim=4)
    model.eval()

    cache = model.forward_with_cache(x, graph)

    expected_m1 = neighbor_sum(cache.H[0], graph)
    expected_m2 = neighbor_sum(cache.H[1], graph)

    torch.testing.assert_close(cache.M[1], expected_m1, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(cache.M[2], expected_m2, rtol=0.0, atol=1e-12)


def test_repeatable_full_forward() -> None:
    torch.manual_seed(7)
    graph, x = classroom_graph()

    model = TwoLayerGraphSAGEMean(input_dim=3, hidden_dim=4, output_dim=4)
    model.eval()

    cache1 = model.forward_with_cache(x, graph)
    cache2 = model.forward_with_cache(x, graph)

    torch.testing.assert_close(cache1.H[2], cache2.H[2], rtol=0.0, atol=1e-12)
