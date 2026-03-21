"""Export PyTorch weights to custom binary format for Rust inference."""

import struct
from pathlib import Path

import torch
import torch.nn as nn

MAGIC = 0xCE550001
VERSION = 1


def fuse_bn_into_conv(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse BatchNorm parameters into convolution weights and bias.

    This eliminates BatchNorm at inference time (zero runtime cost).

    Returns:
        (fused_weight, fused_bias)
    """
    with torch.no_grad():
        # BN parameters
        gamma = bn.weight
        beta = bn.bias
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps

        # Compute scale factor per output channel
        inv_std = 1.0 / torch.sqrt(var + eps)
        scale = gamma * inv_std

        # Fuse into conv weight: new_w = w * scale
        # Conv weight shape: [out_c, in_c, kH, kW]
        fused_weight = conv.weight * scale.view(-1, 1, 1, 1)

        # Fuse into conv bias: new_b = (old_b - mean) * scale + beta
        if conv.bias is not None:
            fused_bias = (conv.bias - mean) * scale + beta
        else:
            fused_bias = -mean * scale + beta

        return fused_weight, fused_bias


def write_tensor(f, tensor: torch.Tensor):
    """Write a tensor as (u32 length, f32[] data)."""
    data = tensor.detach().cpu().contiguous().float().numpy().flatten()
    f.write(struct.pack("<I", len(data)))
    f.write(data.tobytes())


def export_weights(model: nn.Module, path: str | Path, fuse_batchnorm: bool = True):
    """Export model weights to binary format for Rust inference.

    Args:
        model: ChessNet instance
        path: Output file path
        fuse_batchnorm: If True, fuse BN into conv weights (recommended for inference)
    """
    model.eval()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    num_blocks = model.num_blocks
    num_filters = model.num_filters

    with open(path, "wb") as f:
        # Header
        f.write(struct.pack("<I", MAGIC))
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", num_blocks))
        f.write(struct.pack("<I", num_filters))

        if fuse_batchnorm:
            # Stem (fused)
            w, b = fuse_bn_into_conv(model.stem_conv, model.stem_bn)
            write_tensor(f, w)
            write_tensor(f, b)
            # Write identity BN params (gamma=1, beta=0, mean=0, var=1)
            nf = num_filters
            write_tensor(f, torch.ones(nf))   # bn_weight
            write_tensor(f, torch.zeros(nf))  # bn_bias
            write_tensor(f, torch.zeros(nf))  # bn_mean
            write_tensor(f, torch.ones(nf))   # bn_var

            # Residual blocks (fused)
            for block in model.residual_tower:
                w1, b1 = fuse_bn_into_conv(block.conv1, block.bn1)
                write_tensor(f, w1)
                write_tensor(f, b1)
                write_tensor(f, torch.ones(nf))
                write_tensor(f, torch.zeros(nf))
                write_tensor(f, torch.zeros(nf))
                write_tensor(f, torch.ones(nf))

                w2, b2 = fuse_bn_into_conv(block.conv2, block.bn2)
                write_tensor(f, w2)
                write_tensor(f, b2)
                write_tensor(f, torch.ones(nf))
                write_tensor(f, torch.zeros(nf))
                write_tensor(f, torch.zeros(nf))
                write_tensor(f, torch.ones(nf))

            # Policy head (fused)
            pw, pb = fuse_bn_into_conv(model.policy_head.conv, model.policy_head.bn)
            write_tensor(f, pw)
            write_tensor(f, pb)
            pc = model.policy_head.conv.out_channels
            write_tensor(f, torch.ones(pc))
            write_tensor(f, torch.zeros(pc))
            write_tensor(f, torch.zeros(pc))
            write_tensor(f, torch.ones(pc))
            write_tensor(f, model.policy_head.fc.weight)
            write_tensor(f, model.policy_head.fc.bias)

            # Value head (fused)
            vw, vb = fuse_bn_into_conv(model.value_head.conv, model.value_head.bn)
            write_tensor(f, vw)
            write_tensor(f, vb)
            write_tensor(f, torch.ones(1))
            write_tensor(f, torch.zeros(1))
            write_tensor(f, torch.zeros(1))
            write_tensor(f, torch.ones(1))
            write_tensor(f, model.value_head.fc1.weight)
            write_tensor(f, model.value_head.fc1.bias)
            write_tensor(f, model.value_head.fc_value.weight)
            write_tensor(f, model.value_head.fc_value.bias)
        else:
            # Stem (unfused)
            write_tensor(f, model.stem_conv.weight)
            write_tensor(f, model.stem_conv.bias)
            write_tensor(f, model.stem_bn.weight)
            write_tensor(f, model.stem_bn.bias)
            write_tensor(f, model.stem_bn.running_mean)
            write_tensor(f, model.stem_bn.running_var)

            for block in model.residual_tower:
                write_tensor(f, block.conv1.weight)
                write_tensor(f, block.conv1.bias)
                write_tensor(f, block.bn1.weight)
                write_tensor(f, block.bn1.bias)
                write_tensor(f, block.bn1.running_mean)
                write_tensor(f, block.bn1.running_var)

                write_tensor(f, block.conv2.weight)
                write_tensor(f, block.conv2.bias)
                write_tensor(f, block.bn2.weight)
                write_tensor(f, block.bn2.bias)
                write_tensor(f, block.bn2.running_mean)
                write_tensor(f, block.bn2.running_var)

            write_tensor(f, model.policy_head.conv.weight)
            write_tensor(f, model.policy_head.conv.bias)
            write_tensor(f, model.policy_head.bn.weight)
            write_tensor(f, model.policy_head.bn.bias)
            write_tensor(f, model.policy_head.bn.running_mean)
            write_tensor(f, model.policy_head.bn.running_var)
            write_tensor(f, model.policy_head.fc.weight)
            write_tensor(f, model.policy_head.fc.bias)

            write_tensor(f, model.value_head.conv.weight)
            write_tensor(f, model.value_head.conv.bias)
            write_tensor(f, model.value_head.bn.weight)
            write_tensor(f, model.value_head.bn.bias)
            write_tensor(f, model.value_head.bn.running_mean)
            write_tensor(f, model.value_head.bn.running_var)
            write_tensor(f, model.value_head.fc1.weight)
            write_tensor(f, model.value_head.fc1.bias)
            write_tensor(f, model.value_head.fc_value.weight)
            write_tensor(f, model.value_head.fc_value.bias)

    print(f"Weights exported to {path} ({path.stat().st_size / 1024:.1f} KB)")
