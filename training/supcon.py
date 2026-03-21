"""Supervised Contrastive Loss and value binning utilities.

SupConLoss is the official implementation from:
https://github.com/HobbitLong/SupContrast/blob/master/losses.py
"""

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR.

    Source: https://github.com/HobbitLong/SupContrast
    """

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss.

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.

        Returns:
            A loss scalar.
        """
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def discretize_value(
    values: torch.Tensor,
    num_bins: int = 12,
    strategy: str = "equal_width",
) -> torch.Tensor:
    """Convert continuous values in [-1, 1] to discrete bin indices.

    Args:
        values: Tensor of shape [batch_size] with values in [-1, 1]
        num_bins: Number of bins to use
        strategy: "equal_width" or "equal_frequency"

    Returns:
        Tensor of shape [batch_size] with integer bin labels in [0, num_bins-1]
    """
    values = values.detach().clamp(-1.0, 1.0)

    if strategy == "equal_width":
        # Map [-1, 1] -> [0, num_bins-1]
        bins = torch.linspace(-1.0, 1.0, num_bins + 1, device=values.device)
        labels = torch.bucketize(values, bins[1:-1])
    elif strategy == "equal_frequency":
        quantiles = torch.linspace(0, 1, num_bins + 1, device=values.device)
        bins = torch.quantile(values.float(), quantiles)
        labels = torch.bucketize(values, bins[1:-1])
    else:
        raise ValueError(f"Unknown binning strategy: {strategy}")

    return labels.long()


def prepare_supcon_features(
    embeddings: torch.Tensor,
    noise_std: float = 0.01,
) -> torch.Tensor:
    """Prepare features for SupConLoss by creating two views.

    The first view is the original embedding, the second is a noisy copy.
    This is necessary because SupConLoss expects [bsz, n_views, dim].

    Args:
        embeddings: L2-normalized embeddings [bsz, dim]
        noise_std: Standard deviation of Gaussian noise for second view

    Returns:
        Features tensor [bsz, 2, dim]
    """
    view1 = embeddings
    view2 = embeddings + torch.randn_like(embeddings) * noise_std
    # Re-normalize the noisy view
    view2 = torch.nn.functional.normalize(view2, dim=1)
    return torch.stack([view1, view2], dim=1)


def filter_small_bins(
    labels: torch.Tensor, min_bin_size: int = 4
) -> torch.Tensor:
    """Create a mask that excludes samples in bins smaller than min_bin_size.

    Args:
        labels: Bin labels [batch_size]
        min_bin_size: Minimum bin size to keep

    Returns:
        Boolean mask [batch_size] — True for samples to keep
    """
    unique, counts = torch.unique(labels, return_counts=True)
    valid_bins = unique[counts >= min_bin_size]
    mask = torch.isin(labels, valid_bins)
    return mask
