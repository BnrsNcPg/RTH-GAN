"""Generator-only builders and explicit contraction-count profiling.

Counts dense convolution/linear contractions and both RPA attention products.
Elementwise, interpolation, normalization and transcendental costs are excluded;
the resulting 2*MACs is an estimate, never a complete operator FLOP count.
"""
from __future__ import annotations

from collections import Counter
import math

import torch
from torch import nn
from torchvision.ops import DeformConv2d

from models.rghsg import ReflectancePriorAttention, RetinexGuidedHyperbolicGenerator


def build_profile_generator(method, config, device, checkpoint=None):
    name = method.get("method")
    if not method.get("available") or name not in {"pix2pix", "pix2pix_pinn", "rthgan_manuscript"}:
        raise RuntimeError(f"No local model builder is available for {name}")
    if name == "rthgan_manuscript":
        if checkpoint:
            state = torch.load(checkpoint, map_location=device, weights_only=False)
            if state.get("format") != "rthgan_manuscript_v1":
                raise ValueError("Expected a manuscript RTH-GAN checkpoint")
            config = state["config"]  # Profile exactly the saved architecture.
        if config["model"]["name"] != name:
            raise ValueError("Method and model architecture do not match")
        generator = RetinexGuidedHyperbolicGenerator(**config["model"]["generator"]).to(device)
        if checkpoint:
            generator.load_state_dict(state["trainer"]["generator"], strict=True)
    else:
        # Legacy builders remain usable in a research checkout, but are not
        # dependencies of the independently published RTH-GAN core.
        from models.pix2pix import load_checkpoint
        from models.pix2pix_pinn import build_from_config
        bundle = build_from_config(config, device)
        generator = bundle.generator
        has_prior = hasattr(generator, "pinn_retinex")
        if has_prior != (name == "pix2pix_pinn"):
            raise ValueError("Method and model PINN configuration do not match")
        if checkpoint:
            load_checkpoint(checkpoint, bundle, map_location=device)
    return generator.eval()


def parameter_counts(generator):
    prior = getattr(generator, "retinex", getattr(generator, "pinn_retinex", None))
    return {
        "params": sum(p.numel() for p in generator.parameters()),
        "trainable_params": sum(p.numel() for p in generator.parameters() if p.requires_grad),
        "pinn_params": sum(p.numel() for p in prior.parameters()) if prior is not None else 0,
        "prior_projection_params": int(getattr(generator, "projection_parameter_count", 0)),
        "fusion_params": (sum(p.numel() for p in generator.fusion.parameters())
                          if hasattr(generator, "fusion") else int(getattr(generator, "fusion_parameter_count", 0))),
    }


def inference_scope(generator):
    return {
        "scope": "visible_tensor_to_generated_tensor",
        "includes_pinn_prior": bool(parameter_counts(generator)["pinn_params"]),
        "includes_pinn_pde_loss": False,
        "includes_discriminator_or_thermal_auxiliary": False,
        "includes_data_loading_or_export": False,
    }


def contraction_counts(generator, sample):
    counts = Counter()
    handles = []
    training = {module: module.training for module in generator.modules()}

    def count(name, module, inputs, output):
        category = "pinn_linear" if name.startswith(("retinex.", "pinn_retinex.")) else "linear"
        if isinstance(module, nn.Linear):
            counts[category] += output.numel() * module.in_features
        elif isinstance(module, nn.ConvTranspose2d):
            counts["transposed_convolution"] += inputs[0].numel() * (module.out_channels // module.groups) * math.prod(module.kernel_size)
        elif isinstance(module, (nn.Conv2d, DeformConv2d)):
            category = "deformable_convolution" if isinstance(module, DeformConv2d) else "convolution"
            counts[category] += output.numel() * module.weight.shape[1] * math.prod(module.kernel_size)
        elif isinstance(module, ReflectancePriorAttention):
            batch, channels, height, width = output.shape
            # QK^T and softmax(QK^T)V: two B*C*N*N contractions.
            counts["attention_products"] += 2 * batch * channels * (height * width) ** 2

    try:
        for name, module in generator.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d, DeformConv2d, ReflectancePriorAttention)):
                handles.append(module.register_forward_hook(
                    lambda mod, inputs, output, name=name: count(name, mod, inputs, output)))
        generator.eval()
        with torch.inference_mode():
            generator(sample)
    finally:
        for handle in handles:
            handle.remove()
        for module, flag in training.items():
            module.training = flag
    return {
        "macs": sum(counts.values()), "flops": 2 * sum(counts.values()),
        "macs_by_component": dict(sorted(counts.items())),
        "flops_complete": False,
        "flops_convention": "2 x contraction MACs; convolution, transposed convolution (input-site convention), linear, deformable weighted sums and RPA QK/AV products",
        "excluded_operations": ["bias", "activation/softmax", "normalization/GRN", "interpolation/grid/deformable sampling", "positional encoding", "Retinex division/clipping", "hyperbolic/elementwise arithmetic", "validation/reductions"],
        "tool": "explicit_module_hooks_v1",
    }
