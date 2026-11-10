"""Configurable implementation of manuscript Fig. 5-6 and Eq. (8)-(18).

Channel counts/depths are engineering choices, not recovered paper weights.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.ops import DeformConv2d

from .modules.manuscript_fusion import GeodesicDualSpaceFusion
from .modules.manuscript_retinex import ManuscriptRetinex


class ConvNeXtV2Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.spatial = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.expand = nn.Conv2d(channels, 4 * channels, 1)
        self.contract = nn.Conv2d(4 * channels, channels, 1)
        self.grn_scale = nn.Parameter(torch.zeros(1, 4 * channels, 1, 1))
        self.grn_bias = nn.Parameter(torch.zeros(1, 4 * channels, 1, 1))

    def forward(self, value):
        residual = value
        value = self.spatial(value)
        value = self.norm(value.movedim(1, -1)).movedim(-1, 1)
        value = F.gelu(self.expand(value))
        response = torch.linalg.vector_norm(value, dim=(-2, -1), keepdim=True)
        normalized_response = response / (response.mean(1, keepdim=True) + 1e-6)
        value = value + self.grn_scale * value * normalized_response + self.grn_bias
        return residual + self.contract(value)


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, value):
        return self.norm(value.movedim(1, -1)).movedim(-1, 1)


class FeaturePyramid(nn.Module):
    def __init__(self, input_channels, widths, depths):
        super().__init__()
        self.stages = nn.ModuleList()
        previous = input_channels
        for index, (width, depth) in enumerate(zip(widths, depths)):
            stride = 4 if index == 0 else 2
            downsample = [nn.Conv2d(previous, width, stride, stride), ChannelLayerNorm(width)] if index == 0 else [ChannelLayerNorm(previous), nn.Conv2d(previous, width, stride, stride)]
            self.stages.append(nn.Sequential(*downsample,
                *[ConvNeXtV2Block(width) for _ in range(depth)]))
            previous = width

    def forward(self, value):
        outputs = []
        for stage in self.stages:
            value = stage(value)
            outputs.append(value)
        return outputs


class IlluminationGatedAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.visible, self.illumination = nn.Conv2d(channels, channels, 1), nn.Conv2d(channels, channels, 1)
        self.gate = nn.Conv2d(2 * channels, channels, 1)
        self.spatial = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.GELU(), nn.Conv2d(channels, channels, 1))
        self.residual_gate, self.residual_value = nn.Conv2d(channels, channels, 1), nn.Conv2d(channels, channels, 1)

    def forward(self, visible, illumination):
        visible, illumination = self.visible(visible).tanh(), self.illumination(illumination).tanh()
        gate = self.gate(torch.cat([visible, illumination], dim=1)).sigmoid()
        feature = visible * gate * self.spatial(illumination * gate).sigmoid()
        return feature + self.residual_gate(feature).sigmoid().square() * self.residual_value(feature).tanh()


class AtrousContext(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.branches = nn.ModuleList([nn.Conv2d(channels, channels, 3, padding=rate, dilation=rate) for rate in (1, 3, 5)])
        self.merge = nn.Conv2d(3 * channels, channels, 1)

    def forward(self, value):
        return self.merge(torch.cat([F.gelu(branch(value)) for branch in self.branches], dim=1))


class LearnedDeformableConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.offsets = nn.Conv2d(channels, 18, 3, padding=1)
        nn.init.zeros_(self.offsets.weight)
        nn.init.zeros_(self.offsets.bias)
        self.convolution = DeformConv2d(channels, channels, 3, padding=1)

    def forward(self, value):
        return self.convolution(value, self.offsets(value))


class ReflectancePriorAttention(nn.Module):
    def __init__(self, low_channels, channels, heads, query_chunk_size=0):
        super().__init__()
        self.heads = heads
        self.query_chunk_size = query_chunk_size
        self.upsample = nn.Sequential(nn.Conv2d(low_channels, channels * 4, 1), nn.PixelShuffle(2))
        self.high = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.LeakyReLU(.2))
        self.high_context, self.prior_context = AtrousContext(channels), AtrousContext(channels)
        self.deform = nn.ModuleList([LearnedDeformableConv(channels) for _ in range(3)])
        self.merge = nn.Conv2d(3 * channels, channels, 1)
        self.query_key_value = nn.Conv2d(channels, 3 * channels, 1)
        self.prior_value = nn.Conv2d(channels, channels, 1)
        self.output = nn.Conv2d(channels, channels, 1)

    def forward(self, low, high, reflectance):
        low = self.upsample(low)
        high = self.high_context(self.high(high))
        prior = self.prior_context(reflectance)
        branches = [low + prior, high + prior, low + high + prior]
        fused = self.merge(torch.cat([layer(value) for layer, value in zip(self.deform, branches)], dim=1))
        batch, channels, height, width = fused.shape
        query, key, value = self.query_key_value(fused).chunk(3, dim=1)
        value = value * self.prior_value(reflectance).sigmoid()
        def heads(tensor):
            return tensor.reshape(batch, self.heads, channels // self.heads, height * width).transpose(-2, -1)
        query, key, value = heads(query), heads(key), heads(value)
        if self.query_chunk_size and query.shape[-2] > self.query_chunk_size:
            # Every query still attends to ALL keys: exact global attention.
            # Recompute each chunk in backward instead of retaining N x N scores.
            chunks = []
            for part in query.split(self.query_chunk_size, dim=-2):
                if torch.is_grad_enabled() and any(t.requires_grad for t in (part, key, value)):
                    attended_part = checkpoint(F.scaled_dot_product_attention, part, key, value,
                                               use_reentrant=False, preserve_rng_state=False)
                else:
                    attended_part = F.scaled_dot_product_attention(part, key, value)
                chunks.append(attended_part)
            attended = torch.cat(chunks, dim=-2)
        else:
            attended = F.scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(-2, -1).reshape(batch, channels, height, width)
        return fused + self.output(attended)


class PlainSkipFusion(nn.Module):
    """Prospective w/o RPA: ordinary upsample + visual skip convolution."""

    def __init__(self, low_channels, channels):
        super().__init__()
        self.upsample = nn.Sequential(nn.Conv2d(low_channels, channels * 4, 1), nn.PixelShuffle(2))
        self.merge = nn.Sequential(nn.Conv2d(2 * channels, channels, 3, padding=1), nn.GELU())

    def forward(self, low, high, reflectance):
        # Retain the common call signature; this replacement does not use RPA's
        # per-scale reflectance modulation, deformable convolutions or QK/AV.
        return self.merge(torch.cat([self.upsample(low), high], dim=1))


class RetinexGuidedHyperbolicGenerator(nn.Module):
    def __init__(self, *, widths=(40, 80, 160, 320), depths=(2, 2, 6, 2),
                 feature_channels=64, masks=4, attention_heads=4, curvature=1., tangent_bound=.5, pinn=None,
                 attention_query_chunk_size=0, fusion_variant="full", rpa_variant="full"):
        super().__init__()
        if len(widths) != 4 or len(depths) != 4 or any(type(v) is not int or v < 1 for v in (*widths, *depths, feature_channels, masks, attention_heads)):
            raise ValueError("Specify four positive encoder widths/depths and positive feature/mask/head counts")
        if any(width % attention_heads for width in widths):
            raise ValueError("Encoder widths must be divisible by attention_heads")
        if type(attention_query_chunk_size) is not int or attention_query_chunk_size < 0:
            raise ValueError("attention_query_chunk_size must be a nonnegative integer")
        if rpa_variant not in {"full", "plain_skip"}:
            raise ValueError("RPA variant must be full or plain_skip")
        self.rpa_variant = rpa_variant
        self.retinex = ManuscriptRetinex(pinn)
        self.visible_encoder = FeaturePyramid(3, widths, depths)
        self.illumination_encoder = FeaturePyramid(1, widths, depths)
        self.reflectance_encoder = FeaturePyramid(3, widths, depths)
        self.illumination_attention = nn.ModuleList([IlluminationGatedAttention(width) for width in widths])
        self.scale_projections = nn.ModuleList([nn.Conv2d(width, feature_channels, 1) for width in widths])
        self.semantic_context = AtrousContext(feature_channels)
        self.semantic_merge = nn.Conv2d(4 * feature_channels, feature_channels, 1)
        self.semantic_logits = nn.Conv2d(feature_channels, masks, 1)
        self.rpa = nn.ModuleList([
            ReflectancePriorAttention(widths[i+1], widths[i], attention_heads, attention_query_chunk_size)
            if rpa_variant == "full" else PlainSkipFusion(widths[i+1], widths[i])
            for i in range(3)])
        self.mask_decoder = nn.ModuleList([ConvNeXtV2Block(widths[i]) for i in range(3)])
        self.mask_head = nn.ConvTranspose2d(widths[0], masks + 1, 4, stride=2, padding=1)
        self.visible_projection = nn.Conv2d(widths[0], feature_channels, 1)
        self.fusion = GeodesicDualSpaceFusion(feature_channels, curvature=curvature, tangent_bound=tangent_bound,
                                             variant=fusion_variant)
        self.decoder = nn.Sequential(ConvNeXtV2Block(feature_channels), nn.Conv2d(feature_channels, 4 * feature_channels, 1),
            nn.PixelShuffle(2), nn.GELU(), nn.Conv2d(feature_channels, 1, 3, padding=1), nn.Tanh())
        self.freeze_retinex()

    def freeze_retinex(self):
        self.retinex.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        self.retinex.eval()
        return self

    def forward(self, visible, *, return_details=False):
        if visible.ndim != 4 or visible.shape[1] != 3 or min(visible.shape[-2:]) < 32 or any(size % 32 for size in visible.shape[-2:]):
            raise ValueError("RG-HSG requires Bx3xHxW, H/W multiples of 32")
        if not torch.isfinite(visible).all() or (visible < 0).any() or (visible > 1).any():
            raise ValueError("RG-HSG visible surface-reflectance representation must be finite in [0, 1]")
        with torch.no_grad():
            priors = self.retinex(visible)
        visible_features = self.visible_encoder(visible)
        illumination_features = self.illumination_encoder(priors.illumination)
        reflectance_features = self.reflectance_encoder(priors.reflectance)
        size = visible_features[0].shape[-2:]
        semantic = [F.interpolate(projection(attention(v, illumination)), size=size, mode="bilinear", align_corners=False)
            for projection, attention, v, illumination in zip(self.scale_projections, self.illumination_attention, visible_features, illumination_features)]
        semantic = self.semantic_context(self.semantic_merge(torch.cat(semantic, dim=1)))
        decoded = visible_features[-1] + reflectance_features[-1]
        for index in (2, 1, 0):
            decoded = self.mask_decoder[index](self.rpa[index](decoded, visible_features[index], reflectance_features[index]))
        masks = self.mask_head(decoded).tanh()
        attention = F.interpolate(self.semantic_logits(semantic), size=masks.shape[-2:], mode="bilinear", align_corners=False).softmax(dim=1)
        features = F.interpolate(self.visible_projection(visible_features[0]), size=masks.shape[-2:], mode="bilinear", align_corners=False)
        foreground = features * (masks[:, :-1] * attention).sum(dim=1, keepdim=True)
        background = features * masks[:, -1:]
        fused, geometry = self.fusion(background, foreground, return_details=True)
        generated = self.decoder(fused)
        if return_details:
            return generated, {"attention": attention, "masks": masks, "foreground": foreground, "background": background, **geometry}
        return generated
