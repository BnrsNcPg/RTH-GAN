import argparse
import csv
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image

from model.generator import ThermalMaskGenerator
from utils.metrics import (
    InceptionV3Features,
    batch_mae,
    batch_ms_ssim,
    batch_psnr,
    batch_rmse,
    batch_ssim,
    frechet_inception_distance,
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate infrared images and evaluate them against testB."
    )
    parser.add_argument(
        "--weights",
        default="./checkpoints/pinn_paired_exp01/G_epoch_200.pth",
    )
    parser.add_argument("--input_dir", default="./data/testA")
    parser.add_argument("--target_dir", default="./data/testB")
    parser.add_argument("--illum_dir", default="./data/testC")
    parser.add_argument("--reflect_dir", default="./data/testD")
    parser.add_argument("--output_dir", default="./results/pinn_paired_exp01_epoch_200")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--tadsf_hidden_channels", type=int, default=64)
    parser.add_argument(
        "--tadsf_fusion_mode",
        default="full",
        choices=["full", "euc_only", "hyp_only", "no_gate", "no_hyperagg"],
    )
    parser.add_argument(
        "--tadsf_hyperagg_mode",
        default="auto",
        choices=["auto", "spatial", "channel"],
        help="Use auto to infer old spatial or new channel HyperAgg from the checkpoint.",
    )
    parser.add_argument("--tadsf_gate_bias", type=float, default=1.0)
    return parser.parse_args()


def index_images(directory):
    directory = Path(directory)
    if not directory.is_dir():
        return {}
    return {
        path.stem: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def load_tensor(path, transform, mode):
    with Image.open(path) as image:
        return transform(image.convert(mode)).unsqueeze(0)


def infer_hyperagg_mode(state_dict, hidden_channels):
    key = "tadsf.hyperbolic_aggregation.2.weight"
    if key not in state_dict:
        return "channel"
    out_channels = state_dict[key].shape[0]
    if out_channels == 1:
        return "spatial"
    if out_channels == hidden_channels:
        return "channel"
    raise RuntimeError(
        f"Cannot infer HyperAgg mode from {key} with shape "
        f"{tuple(state_dict[key].shape)}."
    )


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = Path(args.weights)
    output_dir = Path(args.output_dir)

    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    image_sets = {
        "input": index_images(args.input_dir),
        "target": index_images(args.target_dir),
        "illumination": index_images(args.illum_dir),
        "reflectance": index_images(args.reflect_dir),
    }
    input_names = set(image_sets["input"])
    if not input_names:
        raise RuntimeError(f"No test images found in: {args.input_dir}")

    errors = []
    for label, images in image_sets.items():
        missing = sorted(input_names - set(images))
        extra = sorted(set(images) - input_names)
        if missing:
            errors.append(f"{label} missing {len(missing)}: {', '.join(missing[:10])}")
        if extra:
            errors.append(f"{label} extra {len(extra)}: {', '.join(extra[:10])}")
    if errors:
        raise RuntimeError("Test dataset validation failed:\n" + "\n".join(errors))

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir = output_dir / "comparisons"
    comparison_dir.mkdir(exist_ok=True)

    rgb_transform = transforms.Compose([
        transforms.Resize(
            (args.img_size, args.img_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
    ])
    prior_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(
            (args.img_size, args.img_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
    ])

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    hyperagg_mode = args.tadsf_hyperagg_mode
    if hyperagg_mode == "auto":
        hyperagg_mode = infer_hyperagg_mode(
            state_dict,
            args.tadsf_hidden_channels,
        )
    model = ThermalMaskGenerator(
        in_channels=3,
        out_channels=3,
        num_features=7,
        tadsf_hidden_channels=args.tadsf_hidden_channels,
        tadsf_fusion_mode=args.tadsf_fusion_mode,
        tadsf_hyperagg_mode=hyperagg_mode,
        tadsf_gate_bias=args.tadsf_gate_bias,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    rows = []
    fid_model = InceptionV3Features().to(device).eval()
    real_features = []
    generated_features = []
    print(f"Device: {device}")
    print(f"Weights: {weights_path}")
    print(
        f"TA-DSF: fusion={args.tadsf_fusion_mode}, "
        f"hyperagg={hyperagg_mode}, hidden={args.tadsf_hidden_channels}"
    )
    print(f"Validated test pairs: {len(input_names)}")

    with torch.inference_mode():
        for name in sorted(input_names):
            visible_01 = load_tensor(image_sets["input"][name], rgb_transform, "RGB")
            target_01 = load_tensor(image_sets["target"][name], rgb_transform, "RGB")
            illumination = load_tensor(
                image_sets["illumination"][name], prior_transform, "L"
            )
            reflectance = load_tensor(
                image_sets["reflectance"][name], prior_transform, "L"
            )

            visible = (visible_01 * 2.0 - 1.0).to(device)
            target = target_01.to(device)
            generated = model(
                visible,
                illumination.to(device),
                reflectance.to(device),
            )
            generated = ((generated + 1.0) / 2.0).clamp(0.0, 1.0)

            mae = batch_mae(generated, target).item()
            psnr = batch_psnr(generated, target).item()
            rmse = batch_rmse(generated, target).item()
            ssim = batch_ssim(generated, target).item()
            ms_ssim = batch_ms_ssim(generated, target).item()
            real_features.append(fid_model(target).cpu())
            generated_features.append(fid_model(generated).cpu())
            rows.append({
                "name": name,
                "mae": mae,
                "psnr": psnr,
                "rmse": rmse,
                "ssim": ssim,
                "ms_ssim": ms_ssim,
                "fid": "",
            })

            save_image(generated.cpu(), output_dir / f"{name}_fake_IR.png")
            comparison = torch.cat(
                [visible_01, generated.cpu(), target_01],
                dim=3,
            )
            save_image(comparison, comparison_dir / f"{name}_comparison.png")
            print(
                f"{name}: PSNR={psnr:.3f}, RMSE={rmse:.3f}, "
                f"SSIM={ssim:.4f}, MS-SSIM={ms_ssim:.4f}, MAE={mae:.4f}"
            )

    summary = {
        "mae": sum(row["mae"] for row in rows) / len(rows),
        "psnr": sum(row["psnr"] for row in rows) / len(rows),
        "rmse": sum(row["rmse"] for row in rows) / len(rows),
        "ssim": sum(row["ssim"] for row in rows) / len(rows),
        "ms_ssim": sum(row["ms_ssim"] for row in rows) / len(rows),
        "fid": frechet_inception_distance(
            torch.cat(real_features),
            torch.cat(generated_features),
        ),
    }
    with open(output_dir / "metrics.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "name",
                "mae",
                "psnr",
                "rmse",
                "ssim",
                "ms_ssim",
                "fid",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({"name": "AVERAGE", **summary})

    with open(output_dir / "summary.txt", "w", encoding="utf-8") as file:
        file.write(f"weights: {weights_path}\n")
        file.write(f"images: {len(rows)}\n")
        file.write(f"MAE: {summary['mae']:.6f}\n")
        file.write(f"PSNR: {summary['psnr']:.6f}\n")
        file.write(f"RMSE: {summary['rmse']:.6f}\n")
        file.write(f"SSIM: {summary['ssim']:.6f}\n")
        file.write(f"MS-SSIM: {summary['ms_ssim']:.6f}\n")
        file.write(f"FID: {summary['fid']:.6f}\n")

    print(
        f"Average: PSNR={summary['psnr']:.3f}, "
        f"RMSE={summary['rmse']:.3f}, SSIM={summary['ssim']:.4f}, "
        f"MS-SSIM={summary['ms_ssim']:.4f}, FID={summary['fid']:.3f}, "
        f"MAE={summary['mae']:.4f}"
    )
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
