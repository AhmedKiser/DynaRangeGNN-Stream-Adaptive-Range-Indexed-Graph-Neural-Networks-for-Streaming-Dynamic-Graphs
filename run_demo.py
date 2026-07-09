import torch

from dynadeltagnn import DynaDeltaEngine, EdgeAdd, EdgeDelete, FeatureUpdate, TwoLayerGraphSAGEMean
from dynadeltagnn.synthetic import classroom_graph


def main() -> None:
    torch.manual_seed(7)

    graph, x = classroom_graph()
    model = TwoLayerGraphSAGEMean(input_dim=3, hidden_dim=4, output_dim=4)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    engine = DynaDeltaEngine(model=model, graph=graph, x=x)

    print("Initial final embeddings:")
    print(engine.cache.H[2])

    new_c = torch.tensor([0.78, 0.30, 0.40], dtype=torch.float64)
    stats = engine.process_event_batch([FeatureUpdate(node=2, new_value=new_c)])
    engine.assert_cache_invariants()
    max_error = engine.compare_to_full_recompute(atol=1e-10)
    print("\nAfter feature update C:")
    print(engine.cache.H[2])
    print("\nStats:", stats)
    print("Max error vs full recomputation:", max_error)

    stats = engine.process_event_batch([EdgeAdd(1, 4)])  # B-E
    engine.assert_cache_invariants()
    max_error = engine.compare_to_full_recompute(atol=1e-10)
    print("\nAfter edge add B-E:")
    print("Stats:", stats)
    print("Max error vs full recomputation:", max_error)

    stats = engine.process_event_batch([EdgeDelete(0, 1)])  # A-B
    engine.assert_cache_invariants()
    max_error = engine.compare_to_full_recompute(atol=1e-10)
    print("\nAfter edge delete A-B:")
    print("Stats:", stats)
    print("Max error vs full recomputation:", max_error)


if __name__ == "__main__":
    main()
