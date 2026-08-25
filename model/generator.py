import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

# 导入你基础网络组件
from model.network import ConvNeXtV2DownBlock, ConvNeXtV2UpBlock, ResidualBlock, ConvNeXtV2Block


# ==========================================
# 1. 基础组件与注意力模块
# ==========================================
class ASPP(nn.Module):
    """空洞空间金字塔池化 (Atrous Spatial Pyramid Pooling)"""

    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.pool_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.project = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(x)
        c3 = self.conv3(x)
        c4 = self.conv4(x)
        p = self.pool_conv(self.pool(x))
        p = F.interpolate(p, size=x.shape[2:], mode='bilinear', align_corners=False)
        out = torch.cat([c1, c2, c3, c4, p], dim=1)
        return self.project(out)


class DeformableConv2d(nn.Module):
    """利用 torchvision 原生 API 封装的可变形卷积层"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DeformableConv2d, self).__init__()
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=kernel_size,
                                     padding=padding)
        self.mask_conv = nn.Conv2d(in_channels, kernel_size * kernel_size, kernel_size=kernel_size, padding=padding)
        self.regular_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        offset = self.offset_conv(x)
        mask = 2.0 * torch.sigmoid(self.mask_conv(x))  # 归一化到 0~2
        return ops.deform_conv2d(x, offset, self.regular_conv.weight, None, padding=self.regular_conv.padding,
                                 mask=mask)


class RPAM(nn.Module):
    """反射先验注意力模块"""

    def __init__(self, in_low, in_high, out_channels):
        super(RPAM, self).__init__()
        self.pixel_shuffle_conv = nn.Conv2d(in_low, out_channels * 4, kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.high_conv = nn.Sequential(
            nn.Conv2d(in_high, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.aspp = ASPP(in_high, out_channels)
        self.fusion_conv = nn.Conv2d(out_channels + in_low, out_channels, kernel_size=1)
        self.re_proj = nn.Conv2d(1, out_channels, kernel_size=1)
        self.dconv_low = DeformableConv2d(out_channels, out_channels)
        self.dconv_high = DeformableConv2d(out_channels, out_channels)
        self.dconv_fusion = DeformableConv2d(out_channels, out_channels)
        self.concat_proj = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)
        self.num_heads = 4
        self.q_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.k_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.v_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.out_proj = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x_low, x_high, x_re):
        x_re_resized = F.interpolate(x_re, size=x_high.shape[2:], mode='bilinear', align_corners=False)
        x_re_p = torch.tanh(self.re_proj(x_re_resized))
        prior_residual = 0.1 * x_re_p
        y_low = self.pixel_shuffle(self.pixel_shuffle_conv(x_low))
        y_high = self.high_conv(x_high)
        aspp_out = self.aspp(x_high)
        aspp_down = F.adaptive_avg_pool2d(aspp_out, x_low.shape[2:])
        fusion_concat = torch.cat([aspp_down, x_low], dim=1)
        y_fusion = F.interpolate(self.fusion_conv(fusion_concat), scale_factor=2, mode='bilinear', align_corners=False)
        y_low_prime = self.dconv_low(y_low + prior_residual)
        y_high_prime = self.dconv_high(y_high + prior_residual)
        y_fusion_prime = self.dconv_fusion(y_fusion + prior_residual)
        z_fusion = self.concat_proj(torch.cat([y_low_prime, y_high_prime, y_fusion_prime], dim=1))

        B, C, H, W = z_fusion.shape
        Q = self.q_conv(z_fusion).view(B, self.num_heads, C // self.num_heads, H * W)
        K = self.k_conv(z_fusion).view(B, self.num_heads, C // self.num_heads, H * W)
        V = self.v_conv(z_fusion)
        V_guided = (V * (1.0 + prior_residual)).view(
            B, self.num_heads, C // self.num_heads, H * W
        )
        attn = torch.matmul(Q, K.transpose(-2, -1)) / ((C // self.num_heads) ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn_out = torch.matmul(attn, V_guided).view(B, C, H, W)
        out = self.out_proj(attn_out) + z_fusion
        return out


# ==========================================
# 2. 新增的物理特征域对齐调制模块 (LAT)
# ==========================================
class LAT_Module(nn.Module):
    """
    可学习的仿射变换模块 (Learnable Affine Transformation)
    利用光照先验动态生成仿射参数 (w, b)，对可见光特征进行域对齐。
    """

    def __init__(self, in_channels=1, out_channels=256, mid_channels=64):
        super(LAT_Module, self).__init__()
        # 1. 全局平均池化 (提取整图的物理统计信息)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # 2. 共享编码器
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # 3. 生成权重 w 和 偏置 b
        self.fc_w = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        self.fc_b = nn.Conv2d(mid_channels, out_channels, kernel_size=1)

        # 【关键初始化】：初始化为恒等映射 (Identity Mapping)
        # 保证初始阶段特征不变： F_out = 1.0 * F_in + 0.0
        nn.init.constant_(self.fc_w.weight, 0)
        nn.init.constant_(self.fc_w.bias, 1.0)
        nn.init.constant_(self.fc_b.weight, 0)
        nn.init.constant_(self.fc_b.bias, 0.0)

    def forward(self, x):
        h = self.pool(x)  # [B, 1, 1, 1]
        h = self.shared_mlp(h)  # [B, mid_channels, 1, 1]
        w = self.fc_w(h)  # [B, out_channels, 1, 1]
        b = self.fc_b(h)  # [B, out_channels, 1, 1]
        return w, b


# ==========================================
# 3. SMGN 模块 (显存优化版，接入物理反射张量)
# ==========================================
class SMGN(nn.Module):
    """基于 ConvNeXt v2 和 RPAM 跳跃连接的显著性掩码生成网络"""

    def __init__(self, in_channels, num_masks=7):
        super(SMGN, self).__init__()
        self.num_masks = num_masks

        # Encoder
        self.down1 = ConvNeXtV2DownBlock(in_channels, 32, kernel_size=7, stride=2, padding=3)
        self.down2 = ConvNeXtV2DownBlock(32, 64, kernel_size=5, stride=2, padding=2)
        self.down3 = ConvNeXtV2DownBlock(64, 128, kernel_size=3, stride=2, padding=1)
        self.down4 = ConvNeXtV2DownBlock(128, 256, kernel_size=3, stride=2, padding=1)

        # Decoder
        self.rpam1 = RPAM(in_low=256, in_high=128, out_channels=128)
        self.up_conv1 = ConvNeXtV2Block(dim=128)
        self.rpam2 = RPAM(in_low=128, in_high=64, out_channels=64)
        self.up_conv2 = ConvNeXtV2Block(dim=64)
        self.rpam3 = RPAM(in_low=64, in_high=32, out_channels=32)
        self.up_conv3 = ConvNeXtV2Block(dim=32)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(32, num_masks + 1, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.Tanh()
        )

    # 【已修改】：删除 extract_reflection_prior，直接接收外部计算好的物理反射张量 x_re
    def forward(self, x, x_re):
        e1 = self.down1(x)
        e2 = self.down2(e1)
        e3 = self.down3(e2)
        e4 = self.down4(e3)

        d1 = self.rpam1(x_low=e4, x_high=e3, x_re=x_re)
        d1 = self.up_conv1(d1)
        d2 = self.rpam2(x_low=d1, x_high=e2, x_re=x_re)
        d2 = self.up_conv2(d2)
        d3 = self.rpam3(x_low=d2, x_high=e1, x_re=x_re)
        d3 = self.up_conv3(d3)

        out = self.final_up(d3)
        M_f = out[:, :self.num_masks, :, :]
        M_b = out[:, self.num_masks:, :, :]
        return M_f, M_b


# ==========================================
# 4. IG_SAGN 模块 (含 LAT 域对齐升级版，接入物理光照张量)
# ==========================================
class IG_SAGN(nn.Module):
    """光照引导语义注意力生成网络"""

    def __init__(self, in_channels=3, num_attention_maps=7, dim=256):
        super(IG_SAGN, self).__init__()

        self.down1 = ConvNeXtV2DownBlock(in_channels, 32, kernel_size=7, stride=2, padding=3)
        self.down2 = ConvNeXtV2DownBlock(32, 64, kernel_size=5, stride=2, padding=2)
        self.down3 = ConvNeXtV2DownBlock(64, 128, kernel_size=3, stride=2, padding=1)
        self.down4 = ConvNeXtV2DownBlock(128, dim, kernel_size=3, stride=2, padding=1)

        # 【新增】：LAT 模块实例化 (将光照域转换为 dim(256) 维的特征缩放/平移参数)
        self.lat_module = LAT_Module(in_channels=1, out_channels=dim, mid_channels=64)

        # 自适应门控注意力单元 (AGAU)
        self.conv_v = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv_i = nn.Conv2d(1, dim, kernel_size=1)
        self.conv_z = nn.Sequential(nn.Conv2d(dim * 2, dim, kernel_size=1), nn.Sigmoid())

        # 空间交互单元 (S-I)
        self.si_conv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.si_gelu = nn.GELU()
        self.si_conv2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.si_sigmoid = nn.Sigmoid()

        # 门控模内特征增强组件
        self.gate_sigma2 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1), nn.Sigmoid())
        self.gate_tanh = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1), nn.Tanh())

        # 多尺度上下文提取
        self.dconv_rate1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, dilation=1)
        self.dconv_rate3 = nn.Conv2d(dim, dim, kernel_size=3, padding=3, dilation=3)
        self.dconv_rate5 = nn.Conv2d(dim, dim, kernel_size=3, padding=5, dilation=5)

        self.final_proj = nn.Conv2d(dim * 3, num_attention_maps, kernel_size=1)

    # 【已修改】：删除 extract_illumination_component，直接接收外部传入的光照先验 illum
    def forward(self, x, illum):
        feat = self.down1(x)
        feat = self.down2(feat)
        feat = self.down3(feat)
        F_v = self.down4(feat)

        # 统一分辨率
        F_i = F.interpolate(illum, size=F_v.shape[2:], mode='bilinear', align_corners=False)

        # ===============================================
        # 【核心新增】：基于 LAT 的特征级域对齐
        # 利用提取出来的真实光照去调制(平移+缩放)当前的特征图
        # ===============================================
        w_i, b_i = self.lat_module(F_i)
        F_v_lat = w_i * F_v + b_i

        # AGAU 处理 (送入的可见光特征已在通道层面被 LAT 矫正)
        H_v = torch.tanh(self.conv_v(F_v_lat))
        H_i = torch.tanh(self.conv_i(F_i))

        Z_concat = torch.cat([H_v, H_i], dim=1)
        Z = self.conv_z(Z_concat)

        F_vg = Z * H_v
        F_ig = Z * H_i

        # S-I 和门控特征增强
        S_I = self.si_sigmoid(self.si_conv2(self.si_gelu(self.si_conv1(F_ig))))
        F_vs = F_vg * S_I

        sigma2_out = self.gate_sigma2(F_vs)
        tanh_out = self.gate_tanh(F_vs)
        F_final = F_vs + (sigma2_out * tanh_out)

        # 多尺度融合
        d1 = self.dconv_rate1(F_final)
        d3 = self.dconv_rate3(F_final)
        d5 = self.dconv_rate5(F_final)
        concat_features = torch.cat([d1, d3, d5], dim=1)

        out = self.final_proj(concat_features)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        A = F.softmax(out, dim=1)

        return A


# ==========================================
# 5. Task-Adaptive Dual-Space Fusion (TA-DSF)
# ==========================================
class TaskAdaptiveDualSpaceFusion(nn.Module):
    """Fuse foreground and background features in Euclidean and Poincare spaces."""

    def __init__(
        self,
        background_channels,
        foreground_channels,
        out_channels,
        hidden_channels=64,
        curvature=1.0,
        eps=1e-5,
        fusion_mode="full",
        hyperagg_mode="channel",
        gate_bias=1.0,
    ):
        super(TaskAdaptiveDualSpaceFusion, self).__init__()
        if curvature <= 0:
            raise ValueError("curvature must be positive")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        valid_fusion_modes = {
            "full",
            "euc_only",
            "hyp_only",
            "no_gate",
            "no_hyperagg",
        }
        if fusion_mode not in valid_fusion_modes:
            raise ValueError(
                f"fusion_mode must be one of {sorted(valid_fusion_modes)}"
            )
        valid_hyperagg_modes = {"spatial", "channel"}
        if hyperagg_mode not in valid_hyperagg_modes:
            raise ValueError(
                f"hyperagg_mode must be one of {sorted(valid_hyperagg_modes)}"
            )

        aggregation_channels = max(1, hidden_channels // 4)
        hyperagg_out_channels = (
            hidden_channels if hyperagg_mode == "channel" else 1
        )
        self.curvature = float(curvature)
        self.eps = eps
        self.tangent_scale = hidden_channels ** -0.5
        self.fusion_mode = fusion_mode
        self.hyperagg_mode = hyperagg_mode

        self.euclidean_branch = nn.Sequential(
            nn.Conv2d(
                background_channels + foreground_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(hidden_channels),
            nn.GELU(),
        )
        self.background_projection = nn.Conv2d(
            background_channels,
            hidden_channels,
            kernel_size=1,
        )
        self.foreground_projection = nn.Conv2d(
            foreground_channels,
            hidden_channels,
            kernel_size=1,
        )

        # HyperAgg is a learned interpolation along the Poincare geodesic.
        self.hyperbolic_aggregation = nn.Sequential(
            nn.Conv2d(3, aggregation_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(aggregation_channels, hyperagg_out_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.task_gate = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        if gate_bias is not None:
            nn.init.zeros_(self.task_gate[0].weight)
            nn.init.constant_(self.task_gate[0].bias, gate_bias)
        self.decoder = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def _safe_eps(self, x):
        return max(self.eps, 4.0 * torch.finfo(x.dtype).eps)

    def _project_to_ball(self, x):
        sqrt_c = self.curvature ** 0.5
        safe_eps = self._safe_eps(x)
        max_norm = (1.0 - safe_eps) / sqrt_c
        norm = torch.linalg.vector_norm(x, dim=1, keepdim=True).clamp_min(safe_eps)
        scale = torch.clamp(max_norm / norm, max=1.0)
        return x * scale

    def _expmap0(self, x):
        sqrt_c = self.curvature ** 0.5
        safe_eps = self._safe_eps(x)
        norm = torch.linalg.vector_norm(x, dim=1, keepdim=True).clamp_min(safe_eps)
        mapped = torch.tanh(sqrt_c * norm) * x / (sqrt_c * norm)
        return self._project_to_ball(mapped)

    def _logmap0(self, x):
        sqrt_c = self.curvature ** 0.5
        x = self._project_to_ball(x)
        safe_eps = self._safe_eps(x)
        norm = torch.linalg.vector_norm(x, dim=1, keepdim=True).clamp_min(safe_eps)
        scaled_norm = (sqrt_c * norm).clamp(max=1.0 - safe_eps)
        return torch.atanh(scaled_norm) * x / (sqrt_c * norm)

    def _mobius_add(self, x, y):
        c = self.curvature
        x2 = torch.sum(x * x, dim=1, keepdim=True)
        y2 = torch.sum(y * y, dim=1, keepdim=True)
        xy = torch.sum(x * y, dim=1, keepdim=True)
        numerator = (1.0 + 2.0 * c * xy + c * y2) * x
        numerator = numerator + (1.0 - c * x2) * y
        denominator = 1.0 + 2.0 * c * xy + (c * c) * x2 * y2
        denominator = denominator.clamp_min(self._safe_eps(denominator))
        return self._project_to_ball(numerator / denominator)

    def _poincare_distance(self, x, y):
        sqrt_c = self.curvature ** 0.5
        delta = self._mobius_add(-x, y)
        norm = torch.linalg.vector_norm(delta, dim=1, keepdim=True)
        scaled_norm = (sqrt_c * norm).clamp(
            max=1.0 - self._safe_eps(norm)
        )
        return 2.0 * torch.atanh(scaled_norm) / sqrt_c

    def _distance_to_origin(self, x):
        sqrt_c = self.curvature ** 0.5
        norm = torch.linalg.vector_norm(x, dim=1, keepdim=True)
        scaled_norm = (sqrt_c * norm).clamp(
            max=1.0 - self._safe_eps(norm)
        )
        return 2.0 * torch.atanh(scaled_norm) / sqrt_c

    def _hyperbolic_features(self, background, foreground):
        background_tangent = self.background_projection(background)
        foreground_tangent = self.foreground_projection(foreground)
        background_tangent = background_tangent * self.tangent_scale
        foreground_tangent = foreground_tangent * self.tangent_scale
        if self.fusion_mode == "no_hyperagg":
            return 0.5 * (background_tangent + foreground_tangent)

        background_h = self._expmap0(background_tangent)
        foreground_h = self._expmap0(foreground_tangent)
        distance = self._poincare_distance(background_h, foreground_h)
        background_radius = self._distance_to_origin(background_h)
        foreground_radius = self._distance_to_origin(foreground_h)
        mix = self.hyperbolic_aggregation(
            torch.cat([distance, background_radius, foreground_radius], dim=1)
        )

        displacement = self._mobius_add(-background_h, foreground_h)
        geodesic_step = self._expmap0(mix * self._logmap0(displacement))
        aggregated_h = self._mobius_add(background_h, geodesic_step)
        return self._logmap0(aggregated_h)

    def forward(self, background, foreground):
        euclidean = None
        if self.fusion_mode != "hyp_only":
            euclidean = self.euclidean_branch(
                torch.cat([background, foreground], dim=1)
            )
        if self.fusion_mode == "euc_only":
            return self.decoder(euclidean)

        hyperbolic = self._hyperbolic_features(background, foreground)
        if self.fusion_mode == "hyp_only":
            return self.decoder(hyperbolic)
        if self.fusion_mode == "no_gate":
            return self.decoder(0.5 * (euclidean + hyperbolic))
        gate = self.task_gate(torch.cat([euclidean, hyperbolic], dim=1))
        fused = gate * euclidean + (1.0 - gate) * hyperbolic
        return self.decoder(fused)


# ==========================================
# 6. 最终生成器总装 (ThermalMaskGenerator)
# ==========================================
class ThermalMaskGenerator(nn.Module):
    """
    完整的跨模态生成器
    统筹调用 SMGN (负责掩码) 和 IG-SAGN (负责语义注意力)，并接入物理张量流
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        num_features=7,
        tadsf_hidden_channels=64,
        tadsf_fusion_mode="full",
        tadsf_hyperagg_mode="channel",
        tadsf_gate_bias=1.0,
    ):
        super(ThermalMaskGenerator, self).__init__()
        self.smgn = SMGN(in_channels, num_masks=num_features)
        self.sagn = IG_SAGN(in_channels, num_attention_maps=num_features)
        self.tadsf = TaskAdaptiveDualSpaceFusion(
            background_channels=in_channels,
            foreground_channels=num_features,
            out_channels=out_channels,
            hidden_channels=tadsf_hidden_channels,
            fusion_mode=tadsf_fusion_mode,
            hyperagg_mode=tadsf_hyperagg_mode,
            gate_bias=tadsf_gate_bias,
        )

    def forward(self, x, illum_map, reflect_map):
        # 1. 传入反射图和原图，生成掩码
        M_f, M_b = self.smgn(x, reflect_map)

        # 2. 传入光照图和原图，生成注意力
        A = self.sagn(x, illum_map)

        # 3. 聚合特征并生成目标图像
        foreground_features = A * M_f
        background_features = x * M_b
        return self.tadsf(background_features, foreground_features)
