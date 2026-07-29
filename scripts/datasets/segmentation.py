"""Utilities for inspecting the segmentation dataset."""

import math
import random
import shutil
from collections import Counter
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

CMAP = plt.get_cmap("tab10")
PAD_LABEL = 255
PAD_COLOR = np.array([1.0, 0.0, 1.0])  # magenta


def find_pairs(
    data_dir: Path,
    num_images: int | None = None,
    image_dir: str = "images",
    label_dir: str = "labels",
):
    images_dir = data_dir / image_dir
    labels_dir = data_dir / label_dir

    image_paths = sorted(images_dir.iterdir())[:num_images]
    pairs = []
    for image_path in image_paths:
        label_paths = sorted(labels_dir.glob(f"{image_path.stem}.*"))
        if not label_paths:
            raise FileNotFoundError(
                f"No label ({image_path.stem}.*) found for image {image_path}"
            )
        if len(label_paths) > 1:
            raise FileNotFoundError(
                f"Multiple labels found for image {image_path}: {label_paths}"
            )
        pairs.append((image_path, label_paths[0]))
    return pairs


def load_label(label_path: Path) -> np.ndarray:
    """Load a label mask, normalizing binary [0, 255] masks to [0, 1].

    Without this, a binary mask's foreground value of 255 would collide
    with `PAD_LABEL`, causing it to be drawn as padding in `preview`.
    """
    label = np.array(Image.open(label_path).convert("L"))
    if set(np.unique(label).tolist()) <= {0, 255}:
        label = (label // 255).astype(label.dtype)
    return label


def image_label_dir_options(func):
    func = click.option(
        "--image-dir",
        default="images",
        show_default=True,
        help="Name of the image subdirectory within input (and output) dir(s).",
    )(func)
    func = click.option(
        "--label-dir",
        default="labels",
        show_default=True,
        help="Name of the label subdirectory within input (and output) dir(s).",
    )(func)
    return func


def pad_label_to_shape(
    label: np.ndarray, shape: tuple[int, int], center: bool = False
) -> np.ndarray:
    """Pad a label to match `shape`.

    By default the label is assumed to occupy the top-left corner of the
    target shape (e.g. a metadata bar cropped off the bottom/right). Pass
    `center=True` when the label instead corresponds to the center of the
    target shape (e.g. an overlap-tile image, whose mask covers only the
    inner region with an equal margin of context on every side).
    """
    height, width = shape
    pad_height = max(height - label.shape[0], 0)
    pad_width = max(width - label.shape[1], 0)
    if pad_height == 0 and pad_width == 0:
        return label
    if center:
        top, left = pad_height // 2, pad_width // 2
    else:
        top, left = 0, 0
    bottom, right = pad_height - top, pad_width - left
    return np.pad(label, ((top, bottom), (left, right)), constant_values=PAD_LABEL)


def _blend_colors(alpha: float, image: np.ndarray, color: np.ndarray) -> np.ndarray:
    return (1 - alpha) * image + alpha * color


def overlay_mask(
    image: np.ndarray, label: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    overlay = image.astype(float) / 255.0
    for class_id in np.unique(label):
        if class_id == 0:
            continue
        if class_id == PAD_LABEL:
            mask = label == class_id
            overlay[mask] = _blend_colors(alpha, overlay[mask], PAD_COLOR)
            continue
        color = np.array(CMAP(class_id % CMAP.N)[:3])
        mask = label == class_id
        overlay[mask] = _blend_colors(alpha, overlay[mask], color)
    return overlay


@click.group()
def cli():
    """Utilities for inspecting the segmentation dataset."""


@cli.command()
@click.option(
    "--input-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "-n",
    "--num-images",
    type=int,
    default=9,
    show_default=True,
    help="Number of images to preview.",
)
@click.option(
    "--pad-top-left",
    is_flag=True,
    default=False,
    help=(
        "Pad masks smaller than their image to fit, treating the mask as "
        "covering the top-left corner of the image (e.g. a cropped "
        "metadata bar), instead of requiring matching sizes."
    ),
)
@click.option(
    "--pad-center",
    is_flag=True,
    default=False,
    help=(
        "Pad masks smaller than their image to fit, treating the mask as "
        "covering the center of the image (e.g. overlap-tile output), "
        "instead of requiring matching sizes."
    ),
)
@image_label_dir_options
def preview(
    input_dir: Path,
    num_images: int,
    pad_top_left: bool,
    pad_center: bool,
    image_dir: str,
    label_dir: str,
):
    """Preview segmentation masks overlaid on their images in a grid."""
    pairs = find_pairs(input_dir, num_images, image_dir, label_dir)
    if not pairs:
        raise ValueError(f"No images found in {input_dir / image_dir}")

    num_cols = math.ceil(math.sqrt(len(pairs)))
    num_rows = math.ceil(len(pairs) / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (image_path, label_path) in zip(axes, pairs):
        image = np.array(Image.open(image_path).convert("RGB"))
        label = load_label(label_path)
        if pad_top_left or pad_center:
            label = pad_label_to_shape(label, image.shape[:2], center=pad_center)
        ax.imshow(overlay_mask(image, label))
        ax.set_title(image_path.stem, fontsize=9)
        ax.axis("off")

    for ax in axes[len(pairs) :]:
        ax.axis("off")

    fig.tight_layout()
    plt.show()


@cli.command()
@click.option(
    "--input-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@image_label_dir_options
def sizes(input_dir: Path, image_dir: str, label_dir: str):
    """Print a table of image/mask sizes and their counts in the dataset."""
    pairs = find_pairs(input_dir, image_dir=image_dir, label_dir=label_dir)
    if not pairs:
        raise ValueError(f"No images found in {input_dir / image_dir}")

    counts = Counter()
    for image_path, label_path in pairs:
        image_size = Image.open(image_path).size
        mask_size = Image.open(label_path).size
        counts[(image_size, mask_size)] += 1

    header = f"{'image size':>16}  {'mask size':>16}  {'count':>6}"
    print(header)
    print("-" * len(header))
    for (image_size, mask_size), count in sorted(
        counts.items(), key=lambda item: item[1], reverse=True
    ):
        image_size_str = f"{image_size[0]}x{image_size[1]}"
        mask_size_str = f"{mask_size[0]}x{mask_size[1]}"
        print(f"{image_size_str:>16}  {mask_size_str:>16}  {count:>6}")


@cli.command()
@click.option(
    "--input-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
)
@image_label_dir_options
def crop(input_dir: Path, output_dir: Path, image_dir: str, label_dir: str):
    """Crop the metadata bar from images so they match their mask size."""
    pairs = find_pairs(input_dir, image_dir=image_dir, label_dir=label_dir)
    if not pairs:
        raise ValueError(f"No images found in {input_dir / image_dir}")

    out_images_dir = output_dir / image_dir
    out_labels_dir = output_dir / label_dir
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    for image_path, label_path in pairs:
        image = Image.open(image_path)
        label = Image.open(label_path)
        width, _ = image.size
        _, mask_height = label.size
        cropped_image = image.crop((0, 0, width, mask_height))
        cropped_image.save(out_images_dir / image_path.name)
        label.save(out_labels_dir / label_path.name)


@cli.command()
@click.option(
    "--input-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option(
    "-t",
    "--train",
    "train_ratio",
    required=True,
    type=float,
    help="Fraction of the dataset to put in the train split.",
)
@click.option(
    "-d",
    "--dev",
    "dev_ratio",
    required=True,
    type=float,
    help="Fraction of the dataset to put in the dev split.",
)
@click.option(
    "-e",
    "--test",
    "test_ratio",
    required=True,
    type=float,
    help="Fraction of the dataset to put in the test split.",
)
@click.option(
    "--seed",
    type=int,
    default=0,
    show_default=True,
    help="Random seed used to shuffle the dataset before splitting.",
)
@image_label_dir_options
def split(
    input_dir: Path,
    output_dir: Path,
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    seed: int,
    image_dir: str,
    label_dir: str,
):
    """Split a dataset into train, dev and test sets."""
    ratio_sum = train_ratio + dev_ratio + test_ratio
    if not math.isclose(ratio_sum, 1.0, abs_tol=1e-6):
        raise ValueError(
            f"--train, --dev and --test must sum to 1.0, got {ratio_sum}"
        )

    pairs = find_pairs(input_dir, image_dir=image_dir, label_dir=label_dir)
    if not pairs:
        raise ValueError(f"No images found in {input_dir / image_dir}")

    random.Random(seed).shuffle(pairs)

    num_pairs = len(pairs)
    num_train = round(num_pairs * train_ratio)
    num_dev = round(num_pairs * dev_ratio)

    splits = {
        "train": pairs[:num_train],
        "dev": pairs[num_train : num_train + num_dev],
        "test": pairs[num_train + num_dev :],
    }

    for split_name, split_pairs in splits.items():
        out_images_dir = output_dir / split_name / image_dir
        out_labels_dir = output_dir / split_name / label_dir
        out_images_dir.mkdir(parents=True, exist_ok=True)
        out_labels_dir.mkdir(parents=True, exist_ok=True)
        for image_path, label_path in split_pairs:
            shutil.copy2(image_path, out_images_dir / image_path.name)
            shutil.copy2(label_path, out_labels_dir / label_path.name)
        print(f"{split_name}: {len(split_pairs)} pairs")


if __name__ == "__main__":
    cli()
