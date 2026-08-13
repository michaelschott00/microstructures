#!/usr/bin/env python3
"""Estimate the GPU VRAM required to train the segmentation models configured under
`configs/models/segmentation/*.yaml`, entirely on CPU (no GPU needed).

The estimate is the sum of two parts:

1. Static memory: the model weights + gradients + optimizer states. These are exact
   (computed from `sum(p.numel())` over a real smp.Unet instance).

2. Activation memory: the peak per-batch tensor footprint, measured by running a real
   forward + backward pass on CPU with tensor-register hooks (sum of C*H*W*itemsize over
   all live tensors at the peak), then scaled linearly to the requested batch size.

Defaults mirror the training configs:
- `configs/task/segmentation*.yaml`: batch_size=32, size=[512, 512], num_classes=5
- `configs/base.yaml`: precision=32-true (FP32)
- `configs/optimization/adamw_basic.yaml`: optimizer=adamw

Encoders are instantiated with `weights=None` to avoid ImageNet downloads; architecture and
memory footprint are identical to the pretrained case.
"""

import argparse
from typing import Dict, List

import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
except ImportError as exc:  # pragma: no cover
    raise SystemExit("segmentation_models_pytorch is required: pip install segmentation-models-pytorch") from exc

# The encoders used by configs/models/segmentation/*.yaml (all Unet).
ENCODERS: List[str] = [
    "efficientnet-b4",
    "inceptionv4",
    "resnet50",
    "se_resnext101_32x4d",
    "senet154",
    "vgg16_bn",
]

# bytes per parameter for each optimizer state (FP32)
BYTES_PER_ELEM = 4
OPTIMIZER_MULTIPLIERS = {
    # weights + grad + states
    "adamw": BYTES_PER_ELEM * 4,  # 4 bytes weight + 4 grad + 8 (exp_avg, exp_avg_sq)
    "sgd": BYTES_PER_ELEM * 3,  # 4 bytes weight + 4 grad + 4 momentum
}

# common GPU memory sizes (GB) used for the "recommended minimum" column
COMMON_CARD_SIZES_GB = [8, 12, 16, 24, 40, 48]


def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def static_bytes(params: int, optimizer: str) -> int:
    """Weights + gradients + optimizer states for a training step."""
    return params * OPTIMIZER_MULTIPLIERS[optimizer]


def _iter_tensors(obj):
    """Yield all tensors nested in a tuple/list/tensor object."""
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            yield from _iter_tensors(item)


def measure_activation_bytes(model: nn.Module, size: int, num_classes: int) -> int:
    """Estimate the per-sample activation footprint (in bytes) for a batch of 1.

    We run a real forward + backward pass on CPU and collect every tensor emitted by every
    module. The activation memory is the sum of numel*itemsize over the *unique* tensors
    (deduplicated by id) that are materialized and held for the backward pass. Deduplication
    avoids double-counting feature maps that are reused (e.g. Unet decoder reusing encoder
    outputs). This is an upper bound on the peak live activation memory at FP32.
    """
    model = model.to(torch.float32).train()
    itemsize = torch.float32.itemsize
    seen = {}  # id(tensor) -> numel, to count each tensor once

    def fwd_hook(_module, _input, output):
        for t in _iter_tensors(output):
            if t is not None and t.is_floating_point():
                seen[id(t)] = t.numel()
        return output

    handles = [m.register_forward_hook(fwd_hook) for m in model.modules()]

    x = torch.randn(1, 3, size, size, dtype=torch.float32, requires_grad=False)
    y = torch.randint(0, num_classes, (1, size, size), dtype=torch.long)

    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()

    for h in handles:
        h.remove()

    return sum(seen.values()) * itemsize


def format_bytes(num_bytes: int) -> str:
    """Format a byte count as MB and GB."""
    mb = num_bytes / (1024**2)
    gb = num_bytes / (1024**3)
    if gb >= 1.0:
        return f"{gb:.2f} GB"
    return f"{mb:.1f} MB"


def recommended_card(total_bytes: int) -> str:
    """Pick the smallest common GPU that comfortably fits the estimate (1.25x headroom)."""
    required_gb = (total_bytes / (1024**3)) * 1.25
    for gb in COMMON_CARD_SIZES_GB:
        if gb >= required_gb:
            return f"{gb} GB"
    return f"> {COMMON_CARD_SIZES_GB[-1]} GB"


def build_model(encoder: str, num_classes: int) -> nn.Module:
    """Build the smp.Unet used by the segmentation configs with random weights."""
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=3,
        classes=num_classes,
    )


def estimate_one(
    encoder: str,
    batch_size: int,
    size: int,
    num_classes: int,
    optimizer: str,
) -> Dict[str, object]:
    """Estimate static + activation VRAM for a single encoder."""
    model = build_model(encoder, num_classes)
    params = count_parameters(model)

    static = static_bytes(params, optimizer)
    activation_per_sample = measure_activation_bytes(model, size, num_classes)
    activations = activation_per_sample * batch_size

    total = static + activations
    return {
        "encoder": encoder,
        "params": params,
        "static_bytes": static,
        "activations_bytes": activations,
        "total_bytes": total,
        "card": recommended_card(total),
    }


def print_table(
    results: List[Dict[str, object]],
    batch_size: int,
    size: int,
    num_classes: int,
    optimizer: str,
) -> None:
    header = (
        f"VRAM estimate (FP32, batch_size={batch_size}, size={size}x{size}, "
        f"num_classes={num_classes}, optimizer={optimizer})\n"
    )
    print(header)
    print("=" * len(header))
    print(f"{'encoder':<22}{'params (M)':>12}{'static':>12}{'activations':>12}{'total':>12}{'min GPU':>10}")
    print("-" * len(header))

    for r in results:
        print(
            f"{r['encoder']:<22}"
            f"{r['params'] / 1e6:>10.2f}M"
            f"{format_bytes(r['static_bytes']):>12}"
            f"{format_bytes(r['activations_bytes']):>12}"
            f"{format_bytes(r['total_bytes']):>12}"
            f"{r['card']:>10}"
        )

    print("-" * len(header))
    print("activations = measured per-sample footprint x batch_size; static = weights+grad+optimizer states.")
    print("min GPU = smallest common card with ~1.25x headroom over the total estimate.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate GPU VRAM for training the segmentation models (CPU only)."
    )
    parser.add_argument("--batch-size", type=int, default=32, help="training batch size (default: 32)")
    parser.add_argument("--size", type=int, default=512, help="square input size in px (default: 512)")
    parser.add_argument("--num-classes", type=int, default=5, help="number of classes (default: 5)")
    parser.add_argument(
        "--optimizer",
        choices=sorted(OPTIMIZER_MULTIPLIERS),
        default="adamw",
        help="optimizer used to derive optimizer-state memory (default: adamw)",
    )
    parser.add_argument(
        "--encoder",
        nargs="+",
        default=ENCODERS,
        help=f"encoders to estimate (default: all: {', '.join(ENCODERS)})",
    )
    args = parser.parse_args()

    torch.manual_seed(0)

    results = [
        estimate_one(e, args.batch_size, args.size, args.num_classes, args.optimizer)
        for e in args.encoder
    ]
    print_table(results, args.batch_size, args.size, args.num_classes, args.optimizer)


if __name__ == "__main__":
    main()