from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from experiment_utils import package_version, write_json
from .image_metrics import (
    compute_metrics,
    compute_ms_ssim,
    evaluation_array,
    evaluation_to_lpips_range,
    grayscale_to_three_channel,
)


DEFAULT_METRICS = {
    "psnr": True,
    "ssim": True,
    "rmse": True,
    "lpips": True,
    "fid": True,
    "kid": True,
    "ms_ssim": False,
}
PAIRWISE = ("psnr", "ssim", "rmse", "lpips", "ms_ssim")


def descriptive_statistics(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if np.isposinf(values).all():
        return {"mean": "Infinity", "std": 0.0}
    if not np.isfinite(values).all():
        mean = "Infinity" if np.isposinf(values).any() else None
        return {"mean": mean, "std": None}
    return {"mean": float(values.mean()), "std": float(values.std(ddof=0))}


def evaluation_settings(config: dict | None) -> dict:
    config = config or {}
    evaluation = config.get("evaluation", config)
    metrics = {**DEFAULT_METRICS, **(evaluation.get("metrics") or {})}
    return {
        "metrics": {name: bool(metrics[name]) for name in DEFAULT_METRICS},
        "metric_seed": int(evaluation.get("metric_seed", 42)),
        "lpips": {"backbone": "alex", **(evaluation.get("lpips") or {})},
        "fid": {"feature": 2048, **(evaluation.get("fid") or {})},
        "kid": {"feature": 2048, "subsets": 100, "subset_size": 50, **(evaluation.get("kid") or {})},
    }


class UnifiedEvaluator:
    """One preprocessing policy for all methods and all formal metrics."""

    def __init__(
        self,
        config: dict | None = None,
        device: str | torch.device = "cpu",
        num_samples: int | None = None,
    ):
        self.settings = evaluation_settings(config)
        self.device = torch.device(device)
        self.rows: list[dict] = []
        self._lpips = None
        self._fid = None
        self._kid = None
        self.num_samples = num_samples
        if self.settings["metrics"]["lpips"]:
            try:
                from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
            except ImportError as exc:
                raise RuntimeError("LPIPS requires the declared torchmetrics dependency") from exc
            self._lpips = LearnedPerceptualImagePatchSimilarity(
                net_type=self.settings["lpips"]["backbone"], normalize=False
            ).to(self.device).eval()
        if self.settings["metrics"]["fid"] or self.settings["metrics"]["kid"]:
            if num_samples is None or num_samples < 2:
                raise ValueError("FID/KID require the declared evaluation-set size (at least two)")
            try:
                from torchmetrics.image.fid import FrechetInceptionDistance
                from torchmetrics.image.kid import KernelInceptionDistance
            except ImportError as exc:
                raise RuntimeError("FID/KID require torchmetrics and its torch-fidelity image dependency") from exc
            if self.settings["metrics"]["fid"]:
                self._fid = FrechetInceptionDistance(
                    feature=int(self.settings["fid"]["feature"]), normalize=True
                ).to(self.device)
            if self.settings["metrics"]["kid"]:
                self._kid = KernelInceptionDistance(
                    feature=int(self.settings["kid"]["feature"]),
                    subsets=int(self.settings["kid"]["subsets"]),
                    subset_size=min(int(self.settings["kid"]["subset_size"]), num_samples),
                    normalize=True,
                ).to(self.device)

    def add(self, pair_id: str, prediction, target, *, dataset: str, subset: str, method: str) -> dict:
        prediction_array = evaluation_array(prediction, name=f"prediction {pair_id}")
        target_array = evaluation_array(target, name=f"target {pair_id}")
        if prediction_array.shape != target_array.shape:
            raise ValueError(f"Shape mismatch for {pair_id}: {prediction_array.shape} != {target_array.shape}")
        enabled = self.settings["metrics"]
        base = compute_metrics(prediction_array, target_array)
        row = {"pair_id": pair_id, "dataset": dataset, "subset": subset, "method": method}
        for name in ("psnr", "ssim", "rmse"):
            if enabled[name]:
                row[name] = base[name]
        if enabled["lpips"]:
            prediction_lpips = evaluation_to_lpips_range(torch.from_numpy(prediction_array)).to(self.device)
            target_lpips = evaluation_to_lpips_range(torch.from_numpy(target_array)).to(self.device)
            with torch.inference_mode():
                row["lpips"] = float(self._lpips(prediction_lpips, target_lpips).detach().cpu())
            self._lpips.reset()
        if enabled["ms_ssim"]:
            row["ms_ssim"] = compute_ms_ssim(prediction_array, target_array)
        if enabled["fid"] or enabled["kid"]:
            generated_rgb = grayscale_to_three_channel(torch.from_numpy(prediction_array)).to(self.device)
            target_rgb = grayscale_to_three_channel(torch.from_numpy(target_array)).to(self.device)
            with torch.inference_mode():
                if self._fid is not None:
                    self._fid.update(target_rgb, real=True)
                    self._fid.update(generated_rgb, real=False)
                if self._kid is not None:
                    self._kid.update(target_rgb, real=True)
                    self._kid.update(generated_rgb, real=False)
        self.rows.append(row)
        return row

    def _distribution_metrics(self) -> tuple[float | None, dict | None, list[str]]:
        enabled = self.settings["metrics"]
        if not (enabled["fid"] or enabled["kid"]):
            return None, None, []
        count = len(self.rows)
        if count < 2:
            raise ValueError("FID/KID require at least two generated/reference images")
        if self.num_samples is not None and count != self.num_samples:
            raise ValueError(f"Evaluated {count} images but expected {self.num_samples}")
        warnings = []
        if count < 100:
            warnings.append("FID may exhibit high variance for small evaluation sets.")
        torch.manual_seed(self.settings["metric_seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.settings["metric_seed"])
        fid_value = None
        if enabled["fid"]:
            fid_value = float(self._fid.compute().detach().cpu())
        kid_value = None
        if enabled["kid"]:
            mean, std = self._kid.compute()
            kid_value = {"mean": float(mean.detach().cpu()), "std": float(std.detach().cpu())}
        return fid_value, kid_value, warnings

    def summary(self, metadata: dict) -> dict:
        if not self.rows:
            raise RuntimeError("No image pairs were evaluated")
        fid_value, kid_value, warnings = self._distribution_metrics()
        payload = {
            "schema_version": 1,
            **metadata,
            "num_test_images": len(self.rows),
            "pairwise_std_semantics": "population standard deviation across evaluated images (ddof=0), not seed-level std",
            "metric_implementation": {
                "pixel_structure": "rth-gan-v2/scikit-image",
                "perceptual_distribution": "torchmetrics",
                "torchmetrics_version": package_version("torchmetrics"),
                "torch_fidelity_version": package_version("torch-fidelity"),
                "lpips_backbone": self.settings["lpips"]["backbone"],
                "metric_seed": self.settings["metric_seed"],
                "image_range": "float32 [0,1]",
                "grayscale_policy": "replicate to [I,I,I] only for LPIPS/FID/KID",
            },
            "warnings": warnings,
        }
        for metric in PAIRWISE:
            if self.settings["metrics"].get(metric):
                values = np.asarray([row[metric] for row in self.rows], dtype=np.float64)
                payload[metric.upper().replace("_", "-")] = descriptive_statistics(values)
        if self.settings["metrics"]["fid"]:
            payload["FID"] = fid_value
        if self.settings["metrics"]["kid"]:
            payload["KID"] = kid_value
        return payload

    def write(self, result_dir: str | Path, metadata: dict) -> dict:
        result_dir = Path(result_dir)
        fieldnames = ["pair_id", "dataset", "subset", "method"] + [
            metric for metric in PAIRWISE if self.settings["metrics"].get(metric)
        ]
        with (result_dir / "metrics_per_image.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        summary = self.summary(metadata)
        write_json(result_dir / "summary.json", summary)
        return summary
