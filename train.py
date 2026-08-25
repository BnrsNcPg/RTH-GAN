import PIL.Image

if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS

import os
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import StepLR
import itertools

from options.train_options import TrainOptions
from utils.visualizer import Visualizer

from model.generator import ThermalMaskGenerator
from model.discriminator import PatchGANDiscriminator, PhysicsInformedDiscriminator
from utils.dataset import PairedDataset
from utils.losses import (
    DomainConversionLoss,
    EdgeLoss,
    HingeLoss,
    MultiScaleSSIMLoss,
    PSCLoss,
    TotalVariationLoss,
)
from utils.metrics import (
    InceptionV3Features,
    batch_mae,
    batch_ms_ssim,
    batch_psnr,
    batch_rmse,
    batch_ssim,
    frechet_inception_distance,
    rgb_to_gray,
)


def validate_generator(model, dataloader, device):
    model.eval()
    totals = {
        "mae": 0.0,
        "psnr": 0.0,
        "rmse": 0.0,
        "ssim": 0.0,
        "ms_ssim": 0.0,
    }
    image_count = 0
    fid_model = InceptionV3Features().to(device).eval()
    real_features = []
    generated_features = []

    with torch.inference_mode():
        for batch in dataloader:
            visible = batch["A"].to(device)
            target = ((batch["B"].to(device) + 1.0) / 2.0).clamp(0.0, 1.0)
            illumination = batch["A_illum"].to(device)
            reflectance = batch["A_reflect"].to(device)

            generated = model(visible, illumination, reflectance)
            generated = ((generated + 1.0) / 2.0).clamp(0.0, 1.0)

            batch_size = visible.shape[0]
            totals["mae"] += batch_mae(generated, target).sum().item()
            totals["psnr"] += batch_psnr(generated, target).sum().item()
            totals["rmse"] += batch_rmse(generated, target).sum().item()
            totals["ssim"] += batch_ssim(generated, target).sum().item()
            totals["ms_ssim"] += batch_ms_ssim(generated, target).sum().item()
            real_features.append(fid_model(target).cpu())
            generated_features.append(fid_model(generated).cpu())
            image_count += batch_size

    metrics = {key: value / image_count for key, value in totals.items()}
    metrics["fid"] = frechet_inception_distance(
        torch.cat(real_features),
        torch.cat(generated_features),
    )
    del fid_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model.train()
    return metrics


def validation_score(metrics, best_metric):
    if best_metric == "ssim":
        return metrics["ssim"]
    return (
        metrics["ssim"]
        + 0.1 * metrics["ms_ssim"]
        + 0.01 * metrics["psnr"]
        - metrics["mae"]
    )


def train():
    opt = TrainOptions().parse()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if opt.early_stop_patience < 0:
        raise ValueError("--early_stop_patience must be >= 0.")
    if opt.min_delta < 0:
        raise ValueError("--min_delta must be >= 0.")

    visualizer = Visualizer(opt)

    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(save_dir, exist_ok=True)

    batch_size = getattr(opt, 'batch_size', 1)
    lr = opt.lr

    alpha_cyc = 2.0
    beta_adv = 0.3
    delta_dom = 1.0
    eta_psc = opt.lambda_psc
    gamma_rad = opt.lambda_rad
    lambda_pair = opt.lambda_pair
    lambda_gray = opt.lambda_gray
    lambda_ssim = opt.lambda_ssim
    lambda_edge = opt.lambda_edge
    lambda_tv = opt.lambda_tv

    generator_kwargs = {
        "tadsf_hidden_channels": opt.tadsf_hidden_channels,
        "tadsf_fusion_mode": opt.tadsf_fusion_mode,
        "tadsf_hyperagg_mode": opt.tadsf_hyperagg_mode,
        "tadsf_gate_bias": opt.tadsf_gate_bias,
    }
    G = ThermalMaskGenerator(
        in_channels=3,
        out_channels=3,
        num_features=7,
        **generator_kwargs,
    ).to(device)
    F = ThermalMaskGenerator(
        in_channels=3,
        out_channels=3,
        num_features=7,
        **generator_kwargs,
    ).to(device)

    D_X = PatchGANDiscriminator(input_channels=3, hidden_channels=32).to(device)
    D_Y_spatial = PatchGANDiscriminator(input_channels=3, hidden_channels=32).to(device)
    D_Y_physics = PhysicsInformedDiscriminator(image_channels=3, radiance_channels=1).to(device)

    criterion_adv = HingeLoss().to(device)
    criterion_cyc = nn.L1Loss().to(device)
    criterion_dom = DomainConversionLoss().to(device)
    criterion_psc = PSCLoss().to(device)
    criterion_rad = nn.L1Loss().to(device)
    criterion_pair = nn.L1Loss().to(device)
    criterion_ssim = MultiScaleSSIMLoss().to(device)
    criterion_edge = EdgeLoss().to(device)
    criterion_tv = TotalVariationLoss().to(device)
    optimizer_G = optim.Adam(itertools.chain(G.parameters(), F.parameters()), lr=lr, betas=(opt.beta1, 0.999))
    optimizer_D = optim.Adam(itertools.chain(D_X.parameters(),
                                             D_Y_spatial.parameters(),
                                             D_Y_physics.parameters()), lr=lr * 0.02, betas=(opt.beta1, 0.999))

    scheduler_G = StepLR(optimizer_G, step_size=50, gamma=0.5)
    scheduler_D = StepLR(optimizer_D, step_size=50, gamma=0.5)

    dataset = PairedDataset(
        root_A='./data/trainA',
        root_B='./data/trainB',
        root_C='./data/trainC',
        root_D='./data/trainD',
        root_E='./data/trainE',
        is_train=True,
        img_size=opt.img_size
    )
    if not 0.0 <= opt.val_ratio < 1.0:
        raise ValueError("--val_ratio must be in the range [0, 1).")

    val_count = int(round(len(dataset) * opt.val_ratio))
    if opt.val_ratio > 0.0:
        val_count = max(1, val_count)
    val_count = min(val_count, len(dataset) - 1)

    split_generator = torch.Generator().manual_seed(opt.val_seed)
    indices = torch.randperm(len(dataset), generator=split_generator).tolist()
    val_indices = sorted(indices[:val_count])
    train_indices = sorted(indices[val_count:])

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices) if val_indices else None
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_dataloader = (
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        if val_dataset is not None
        else None
    )

    with open(os.path.join(save_dir, "validation_split.txt"), "w", encoding="utf-8") as file:
        file.write(f"seed={opt.val_seed}\n")
        file.write(f"train_count={len(train_indices)}\n")
        file.write(f"val_count={len(val_indices)}\n")
        file.write("validation_names:\n")
        for index in val_indices:
            file.write(f"{dataset.samples[index]['A'].stem}\n")

    validation_log = os.path.join(save_dir, "validation_metrics.csv")
    with open(validation_log, "w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow([
            "epoch",
            "mae",
            "psnr",
            "rmse",
            "ssim",
            "ms_ssim",
            "fid",
            "score",
            "best",
        ])

    print(
        f"Training pairs: {len(train_indices)}, validation pairs: {len(val_indices)}, "
        f"image size: {opt.img_size}"
    )
    print(
        f"TA-DSF: fusion={opt.tadsf_fusion_mode}, "
        f"hyperagg={opt.tadsf_hyperagg_mode}, "
        f"hidden={opt.tadsf_hidden_channels}, gate_bias={opt.tadsf_gate_bias}"
    )
    print(f"Best model criterion: {opt.best_metric}")
    if opt.early_stop_patience > 0:
        print(
            f"Early stopping: patience={opt.early_stop_patience} validation runs, "
            f"min_delta={opt.min_delta}"
        )

    total_iters = 0
    total_epochs = opt.n_epochs + opt.n_epochs_decay

    loss_D = torch.tensor(0.0).to(device)
    loss_rad = torch.tensor(0.0).to(device)
    best_score = float("-inf")
    no_improve_count = 0
    stop_training = False
    for epoch in range(opt.epoch_count, total_epochs + 1):
        iter_data_time = time.time()

        for i, batch in enumerate(dataloader):
            iter_start_time = time.time()
            total_iters += batch_size

            real_X = batch['A'].to(device)
            real_Y = batch['B'].to(device)

            real_X_illum = batch['A_illum'].to(device)
            real_X_reflect = batch['A_reflect'].to(device)
            real_X_radiance = batch['A_radiance'].to(device)

            optimizer_G.zero_grad()

            zero_illum = torch.zeros_like(real_X_illum)
            zero_reflect = torch.zeros_like(real_X_reflect)

            loss_dom_Y = criterion_dom(G(real_Y, zero_illum, zero_reflect), real_Y)
            loss_dom_X = criterion_dom(F(real_X, real_X_illum, real_X_reflect), real_X)
            loss_dom = loss_dom_Y + loss_dom_X

            fake_Y = G(real_X, real_X_illum, real_X_reflect)
            fake_X = F(real_Y, zero_illum, zero_reflect)

            fake_Y_unnorm = (fake_Y + 1.0) / 2.0
            fake_Y_gray = 0.299 * fake_Y_unnorm[:, 0:1, :, :] + \
                          0.587 * fake_Y_unnorm[:, 1:2, :, :] + \
                          0.114 * fake_Y_unnorm[:, 2:3, :, :]
            loss_rad = criterion_rad(fake_Y_gray, real_X_radiance)
            pred_fake_Y_spatial = D_Y_spatial(fake_Y)
            pred_fake_Y_physics = D_Y_physics(fake_Y, real_X_radiance)
            loss_adv_G_spatial = criterion_adv.get_G_loss(pred_fake_Y_spatial)
            loss_adv_G_physics = criterion_adv.get_G_loss(pred_fake_Y_physics)
            loss_adv_G = loss_adv_G_spatial + loss_adv_G_physics
            pred_fake_X = D_X(fake_X)
            loss_adv_F = criterion_adv.get_G_loss(pred_fake_X)
            loss_adv = loss_adv_G + loss_adv_F
            rec_X = F(fake_Y, zero_illum, zero_reflect)
            rec_Y = G(fake_X, real_X_illum, real_X_reflect)
            loss_cyc = criterion_cyc(rec_X, real_X) + criterion_cyc(rec_Y, real_Y)
            loss_pair = criterion_pair(fake_Y, real_Y)
            fake_Y_gray = rgb_to_gray(fake_Y)
            loss_gray = criterion_pair(
                fake_Y,
                fake_Y_gray.repeat(1, fake_Y.shape[1], 1, 1),
            )
            real_Y_01 = ((real_Y + 1.0) / 2.0).clamp(0.0, 1.0)
            loss_ssim = criterion_ssim(fake_Y_unnorm, real_Y_01)
            loss_edge = criterion_edge(fake_Y_unnorm, real_Y_01)
            loss_tv = criterion_tv(fake_Y_unnorm)
            l_pix, l_dfft = criterion_psc(real_Y, fake_Y)
            loss_psc = l_pix + l_dfft
            loss_G = (alpha_cyc * loss_cyc) + \
                     (beta_adv * loss_adv) + \
                     (delta_dom * loss_dom) + \
                     (eta_psc * loss_psc) + \
                     (gamma_rad * loss_rad) + \
                     (lambda_pair * loss_pair) + \
                     (lambda_gray * loss_gray) + \
                     (lambda_ssim * loss_ssim) + \
                     (lambda_edge * loss_edge) + \
                     (lambda_tv * loss_tv)

            loss_G.backward()
            optimizer_G.step()
            if i % 3 == 0:
                optimizer_D.zero_grad()

                pred_real_Y_spatial = D_Y_spatial(real_Y)
                pred_fake_Y_spatial_det = D_Y_spatial(fake_Y.detach())
                loss_D_Y_spatial = criterion_adv.get_D_loss(pred_real_Y_spatial, pred_fake_Y_spatial_det)

                pred_real_Y_physics = D_Y_physics(real_Y, real_X_radiance)
                pred_fake_Y_physics_det = D_Y_physics(fake_Y.detach(), real_X_radiance)
                loss_D_Y_physics = criterion_adv.get_D_loss(pred_real_Y_physics, pred_fake_Y_physics_det)

                loss_D_Y = loss_D_Y_spatial + loss_D_Y_physics
                pred_real_X = D_X(real_X)
                pred_fake_X_det = D_X(fake_X.detach())
                loss_D_X = criterion_adv.get_D_loss(pred_real_X, pred_fake_X_det)

                loss_D = loss_D_X + loss_D_Y
                loss_D.backward()
                optimizer_D.step()

            if total_iters % opt.print_freq == 0:
                losses = {
                    'D_total': loss_D.item(),
                    'G_total': loss_G.item(),
                    'Adv': loss_adv.item(),
                    'Pair': loss_pair.item(),
                    'Gray': loss_gray.item(),
                    'SSIM': loss_ssim.item(),
                    'Edge': loss_edge.item(),
                    'TV': loss_tv.item(),
                    'PSC': loss_psc.item(),
                    'Rad': loss_rad.item()
                }
                t_comp = (time.time() - iter_start_time) / batch_size
                t_data = iter_start_time - iter_data_time
                visualizer.print_current_losses(epoch, total_iters, losses, t_comp, t_data)

            if total_iters % opt.display_freq == 0:
                save_result = True

                visuals = {
                    'real_Y': real_Y,
                    'fake_Y': fake_Y
                }

                visualizer.display_current_results(visuals, epoch, save_result)

            iter_data_time = time.time()

        scheduler_G.step()
        scheduler_D.step()

        if val_dataloader is not None and epoch % opt.val_freq == 0:
            metrics = validate_generator(G, val_dataloader, device)
            score = validation_score(metrics, opt.best_metric)
            is_best = score > best_score + opt.min_delta
            if is_best:
                best_score = score
                no_improve_count = 0
                torch.save(G.state_dict(), os.path.join(save_dir, "G_best.pth"))
            elif opt.early_stop_patience > 0:
                no_improve_count += 1

            with open(validation_log, "a", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow([
                    epoch,
                    f"{metrics['mae']:.8f}",
                    f"{metrics['psnr']:.8f}",
                    f"{metrics['rmse']:.8f}",
                    f"{metrics['ssim']:.8f}",
                    f"{metrics['ms_ssim']:.8f}",
                    f"{metrics['fid']:.8f}",
                    f"{score:.8f}",
                    int(is_best),
                ])
            print(
                f"[validation] epoch: {epoch}, MAE: {metrics['mae']:.4f}, "
                f"PSNR: {metrics['psnr']:.3f}, RMSE: {metrics['rmse']:.3f}, "
                f"SSIM: {metrics['ssim']:.4f}, "
                f"MS-SSIM: {metrics['ms_ssim']:.4f}, "
                f"FID: {metrics['fid']:.3f}, "
                f"score: {score:.4f}, "
                f"best: {is_best}"
            )
            if opt.early_stop_patience > 0:
                print(
                    f"[early stopping] no improvement: {no_improve_count}/"
                    f"{opt.early_stop_patience}"
                )
                if no_improve_count >= opt.early_stop_patience:
                    print(
                        f"[early stopping] stopped at epoch {epoch}; "
                        f"best score: {best_score:.4f}"
                    )
                    stop_training = True

        if epoch % opt.save_epoch_freq == 0:
            torch.save(G.state_dict(), os.path.join(save_dir, f'G_epoch_{epoch}.pth'))
            torch.save(F.state_dict(), os.path.join(save_dir, f'F_epoch_{epoch}.pth'))
            torch.save(D_Y_spatial.state_dict(), os.path.join(save_dir, f'DY_spatial_epoch_{epoch}.pth'))
            torch.save(D_Y_physics.state_dict(), os.path.join(save_dir, f'DY_physics_epoch_{epoch}.pth'))

        if stop_training:
            break


if __name__ == '__main__':
    train()
