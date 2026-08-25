from pathlib import Path

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def index_images(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    images = {}
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            if path.stem in images:
                raise RuntimeError(f"Duplicate base name in {root}: {path.stem}")
            images[path.stem] = path
    return images


class PairedDataset(Dataset):
    def __init__(
        self,
        root_A,
        root_B,
        root_C="./data/trainC",
        root_D="./data/trainD",
        root_E="./data/trainE",
        is_train=True,
        img_size=128,
    ):
        del is_train

        image_sets = {
            "A": index_images(root_A),
            "B": index_images(root_B),
            "C": index_images(root_C),
            "D": index_images(root_D),
            "E": index_images(root_E),
        }
        reference_names = set(image_sets["A"])
        if not reference_names:
            raise RuntimeError(f"No images found in: {root_A}")

        errors = []
        for key, images in image_sets.items():
            names = set(images)
            missing = sorted(reference_names - names)
            extra = sorted(names - reference_names)
            if missing:
                errors.append(f"{key} missing {len(missing)}: {', '.join(missing[:10])}")
            if extra:
                errors.append(f"{key} extra {len(extra)}: {', '.join(extra[:10])}")
        if errors:
            raise RuntimeError("Paired dataset validation failed:\n" + "\n".join(errors))

        self.samples = [
            {
                key: image_sets[key][name]
                for key in image_sets
            }
            for name in sorted(reference_names)
        ]

        self.transform_rgb = transforms.Compose([
            transforms.Resize(
                (img_size, img_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.transform_physics = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(
                (img_size, img_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
        ])

        print(f"Validated {len(self.samples)} strictly paired training samples.")

    def __getitem__(self, index):
        sample = self.samples[index]

        with Image.open(sample["A"]) as image:
            item_A = self.transform_rgb(image.convert("RGB"))
        with Image.open(sample["B"]) as image:
            item_B = self.transform_rgb(image.convert("RGB"))
        with Image.open(sample["C"]) as image:
            illumination = self.transform_physics(image)
        with Image.open(sample["D"]) as image:
            reflectance = self.transform_physics(image)
        with Image.open(sample["E"]) as image:
            radiance = self.transform_physics(image)

        return {
            "A": item_A,
            "B": item_B,
            "A_illum": illumination,
            "A_reflect": reflectance,
            "A_radiance": radiance,
            "name": sample["A"].stem,
        }

    def __len__(self):
        return len(self.samples)


# Keep the old import name working for external scripts.
UnpairedDataset = PairedDataset
