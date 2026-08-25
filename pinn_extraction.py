from __future__ import annotations
import glob, os
import deepxde as dde
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ==========================================
# 0. 全局设置
# ==========================================
dde.backend.set_default_backend("pytorch")
dde.config.set_default_float("float32")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

H, W = 256, 256
LAMBDA = 0.1
ALPHA = 1.0
EPSILON = 1e-4
L_MAX = 1.0

SAVE_DIR = r""
os.makedirs(SAVE_DIR, exist_ok=True)

S_tensor = None
C_tensor = None
dCdx_tensor = None
dCdy_tensor = None

# ==========================================
# 1. 物理场预计算
# ==========================================
def precompute_c_and_gradients(S: torch.Tensor):
    sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                            [-2.0, 0.0, 2.0],
                            [-1.0, 0.0, 1.0]],
                           dtype=S.dtype, device=S.device).view(1, 1, 3, 3) / 8.0

    sobel_y = torch.tensor([[-1.0, -2.0, -1.0],
                            [0.0, 0.0, 0.0],
                            [1.0, 2.0, 1.0]],
                           dtype=S.dtype, device=S.device).view(1, 1, 3, 3) / 8.0

    grad_S_x = F.conv2d(S, sobel_x, padding=1)
    grad_S_y = F.conv2d(S, sobel_y, padding=1)
    grad_S_mag = torch.sqrt(grad_S_x**2 + grad_S_y**2 + 1e-8)

    c_tensor = 1.0 / (torch.pow(grad_S_mag, ALPHA) + EPSILON)

    dc_dx_tensor = F.conv2d(c_tensor, sobel_x, padding=1)
    dc_dy_tensor = F.conv2d(c_tensor, sobel_y, padding=1)

    return c_tensor, dc_dx_tensor, dc_dy_tensor


def get_image_features(points):
    global S_tensor, C_tensor, dCdx_tensor, dCdy_tensor

    pts = points.to(device=S_tensor.device, dtype=S_tensor.dtype)
    N = pts.shape[0]

    grid_x = pts[:, 0:1] / (W / 2.0) - 1.0
    grid_y = pts[:, 1:2] / (H / 2.0) - 1.0
    grid = torch.cat([grid_x, grid_y], dim=-1).view(1, N, 1, 2)

    def sample(tensor_map):
        sampled = F.grid_sample(
            tensor_map,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.view(N, 1)

    return sample(S_tensor), sample(C_tensor), sample(dCdx_tensor), sample(dCdy_tensor)

# ==========================================
# 2. 单一 PDE（核心修复）
# ==========================================
def retinex_pde(x, L_hat):
    S_val, c_val, dc_dx_val, dc_dy_val = get_image_features(x)

    dL_dx = dde.grad.jacobian(L_hat, x, i=0, j=0)
    dL_dy = dde.grad.jacobian(L_hat, x, i=0, j=1)

    d2L_dx2 = dde.grad.hessian(L_hat, x, i=0, j=0)
    d2L_dy2 = dde.grad.hessian(L_hat, x, i=1, j=1)

    # ===== PDE 主项 =====
    div_term = (
        dc_dx_val * dL_dx +
        c_val * d2L_dx2 +
        dc_dy_val * dL_dy +
        c_val * d2L_dy2
    )
    pde_res = -LAMBDA * div_term + L_hat - S_val

    # ===== 平滑项 =====
    smooth = c_val * (dL_dx**2 + dL_dy**2)

    # ===== 锚点约束 =====
    mask = (S_val > 0.7).float()
    anchor = mask * (L_hat - S_val)**2

    # ===== 数值约束 =====
    value = F.relu(-L_hat) + F.relu(L_hat - L_MAX)

    # ===== 总残差（关键）=====
    total = (
        1.0 * pde_res +
        0.2 * smooth +
        0.5 * anchor +
        0.1 * value
    )

    return total

# ==========================================
# 3. 网络
# ==========================================
def input_transform(inputs):
    x_norm = inputs[:, 0:1] / (W / 2.0) - 1.0
    y_norm = inputs[:, 1:2] / (H / 2.0) - 1.0
    return torch.cat([x_norm, y_norm], dim=1)


def output_transform(inputs, outputs):
    return 0.01 + 0.99 * torch.sigmoid(outputs)

geom = dde.geometry.Rectangle([0, 0], [W, H])

# ==========================================
# 4. 主函数
# ==========================================
def main():
    global S_tensor, C_tensor, dCdx_tensor, dCdy_tensor

    image_paths = glob.glob(r"*.png")

    for idx, img_path in enumerate(image_paths):
        print(f"\n处理图像: {img_path}")

        img = Image.open(img_path).convert("RGB").resize((W, H))
        img_rgb = np.array(img, dtype=np.float32) / 255.0

        # 使用 V 通道
        S_v = np.max(img_rgb, axis=2)
        S_tensor = torch.tensor(S_v, device=device).view(1, 1, H, W)

        C_tensor, dCdx_tensor, dCdy_tensor = precompute_c_and_gradients(S_tensor)

        data = dde.data.PDE(
            geom,
            retinex_pde,
            [],
            num_domain=10000,
        )

        net = dde.nn.FNN([2] + [64] * 4 + [1], "tanh", "Glorot uniform")
        net.apply_feature_transform(input_transform)
        net.apply_output_transform(output_transform)

        model = dde.Model(data, net)

        # ✅ 关键：单 loss
        model.compile("adam", lr=1e-3, loss="MSE")

        print("阶段1: Adam训练")
        model.train(iterations=5000, display_every=500)

        print("阶段2: L-BFGS优化")
        model.compile("L-BFGS")
        model.train()

        # ==========================================
        # 推理
        # ==========================================
        x_lin = np.linspace(0, W, W, dtype=np.float32)
        y_lin = np.linspace(0, H, H, dtype=np.float32)
        X, Y = np.meshgrid(x_lin, y_lin)
        pts = np.stack([X.ravel(), Y.ravel()], axis=1)

        L_pred = model.predict(pts).reshape(H, W)
        L_img = np.clip(L_pred, 0.01, 1.0)

        R = img_rgb / L_img[..., None]

        p1, p99 = np.percentile(R, (1, 99))
        R = np.clip((R - p1) / (p99 - p1 + 1e-8), 0, 1)

        base = os.path.splitext(os.path.basename(img_path))[0]

        Image.fromarray((R * 255).astype(np.uint8)).save(
            os.path.join(SAVE_DIR, f"{base}_R.png")
        )
        Image.fromarray((L_img * 255).astype(np.uint8), mode="L").save(
            os.path.join(SAVE_DIR, f"{base}_L.png")
        )

        print("完成\n")

# ==========================================
if __name__ == "__main__":
    main()