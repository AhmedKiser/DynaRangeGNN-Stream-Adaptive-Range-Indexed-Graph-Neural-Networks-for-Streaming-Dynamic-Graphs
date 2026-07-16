"""
link_predictor.py
------------------
The decoder referenced in the roadmap (Phase 6: "Add the link-prediction
decoder"). Turns a pair of node embeddings into a probability that an
edge exists / will exist between them.

score(u, v) = sigmoid(h_u . h_v + bias)

Why the learnable bias matters here: GraphSAGE-Mean's layers end in a
ReLU, so every embedding coordinate is >= 0, which makes every raw dot
product >= 0 too -- and sigmoid(x) >= 0.5 for any x >= 0. Without a bias,
EVERY pair would be classified "positive" regardless of how well the
embeddings actually separate true from false edges (you'd see this as
recall = 1.0 and accuracy stuck at the base rate, even though ROC-AUC /
Average Precision -- which don't depend on a fixed threshold -- look
fine). A single learned bias lets training move the decision boundary
to wherever actually separates the two classes.
"""

import torch
import torch.nn as nn


class DotProductDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))

    def logits(self, embeddings, pairs):
        """embeddings: dict node_id -> tensor. pairs: list of (u, v).
        Returns raw (pre-sigmoid) scores, one per pair."""
        hu = torch.stack([embeddings[u] for u, _ in pairs])
        hv = torch.stack([embeddings[v] for _, v in pairs])
        return (hu * hv).sum(dim=-1) + self.bias

    def probs(self, embeddings, pairs):
        return torch.sigmoid(self.logits(embeddings, pairs))
