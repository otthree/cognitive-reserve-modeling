"""
Joint Loss for 3D Decoupling AD Network.

L_Total = L_CE + alpha * L_SC

L_SC = (1/2m) * sum_i ||S_i - SC_{y_i}||^2

where S_i is the feature vector of sample i and SC_{y_i} is the running
center of class y_i, updated each batch.
"""

import torch
import torch.nn as nn


class ClusteringLoss(nn.Module):
    """
    Clustering loss L_SC.

    Minimizes intra-class distance between sample features and their class center.
    Centers are initialised to zero and updated incrementally each forward pass
    (only during training — call loss.train() / loss.eval() accordingly).
    """

    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__()
        self.num_classes = num_classes
        # Centers are not parameters (no gradient); stored as buffers
        self.register_buffer("centers", torch.zeros(num_classes, feat_dim))

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        features : (B, feat_dim)
        labels   : (B,)  integer class indices
        """
        centers_batch = self.centers[labels]           # (B, feat_dim)
        loss = 0.5 * ((features - centers_batch) ** 2).sum(dim=1).mean()

        if self.training:
            self._update_centers(features.detach(), labels)

        return loss

    @torch.no_grad()
    def _update_centers(self, features: torch.Tensor, labels: torch.Tensor):
        """Incremental center update (eq. 6 in the paper)."""
        for j in range(self.num_classes):
            mask = labels == j
            if mask.sum() == 0:
                continue
            class_feats = features[mask]
            # delta = (center - mean_of_class_feats) / (1 + count)
            delta = (self.centers[j] - class_feats).mean(dim=0)
            self.centers[j] -= delta / (1.0 + mask.sum().float())


class JointLoss(nn.Module):
    """
    L_Total = L_CE + alpha * L_SC
    """

    def __init__(self, num_classes: int, feat_dim: int, alpha: float = 0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.sc = ClusteringLoss(num_classes, feat_dim)
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, features: torch.Tensor,
                labels: torch.Tensor):
        """
        Returns:
            total_loss : scalar
            ce_loss    : scalar (for logging)
            sc_loss    : scalar (for logging)
        """
        l_ce = self.ce(logits, labels)
        l_sc = self.sc(features, labels)
        return l_ce + self.alpha * l_sc, l_ce.item(), l_sc.item()
