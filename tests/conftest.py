import numpy as np
import cv2
import pytest


def _write_image(path, size=(32, 32), color=(255, 0, 0)):
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    img[:] = color
    cv2.imwrite(str(path), img)


def _write_mask(path, size=(32, 32), value=1):
    mask = np.full((size[0], size[1]), value, dtype=np.uint8)
    cv2.imwrite(str(path), mask)


@pytest.fixture
def classification_data_dir(tmp_path):
    """Builds a tiny classification dataset with two classes and train/dev/test splits."""
    root = tmp_path / "classification"
    for split in ["train", "dev", "test"]:
        for class_name in ["class_a", "class_b"]:
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True)
            for i in range(3):
                _write_image(class_dir / f"img_{i}.png")
    return root


@pytest.fixture
def segmentation_data_dir(tmp_path):
    """Builds a tiny tiled segmentation dataset with train/dev/test splits."""
    root = tmp_path / "segmentation"
    for split in ["train", "dev", "test"]:
        img_dir = root / "tiled" / split / "Original"
        mask_dir = root / "tiled" / split / "Masks"
        img_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        for i in range(3):
            _write_image(img_dir / f"img_{i}.png")
            _write_mask(mask_dir / f"img_{i}.png", value=1 if i % 2 else 0)
    return root
