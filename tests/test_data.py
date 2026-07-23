import numpy as np
import pytest

from transfer_learning.data import (
    ClassificationDataModule,
    ClassificationDataset,
    SegmentationDataModule,
    SegmentationDataset,
)


class TestClassificationDataset:
    def test_rejects_unknown_split(self, classification_data_dir):
        with pytest.raises(ValueError):
            ClassificationDataset(split="bogus", root_dir=str(classification_data_dir))

    def test_rejects_missing_root_dir(self, tmp_path):
        with pytest.raises(AssertionError):
            ClassificationDataset(split="train", root_dir=str(tmp_path / "does_not_exist"))

    def test_discovers_classes_from_subfolders(self, classification_data_dir):
        dataset = ClassificationDataset(split="train", root_dir=str(classification_data_dir))

        assert set(dataset.LABELS.keys()) == {"class_a", "class_b"}

    def test_len_counts_all_files_across_classes(self, classification_data_dir):
        dataset = ClassificationDataset(split="train", root_dir=str(classification_data_dir))

        assert len(dataset) == 6  # 2 classes * 3 images

    def test_getitem_returns_image_and_correct_label(self, classification_data_dir):
        dataset = ClassificationDataset(split="train", root_dir=str(classification_data_dir))

        for img, label in dataset:
            assert img.shape == (32, 32, 3)
            assert label in dataset.LABELS.values()

    def test_getitem_rejects_slicing(self, classification_data_dir):
        dataset = ClassificationDataset(split="train", root_dir=str(classification_data_dir))

        with pytest.raises(AssertionError):
            dataset[0:2]


class TestSegmentationDataset:
    def test_rejects_unknown_split(self, segmentation_data_dir):
        with pytest.raises(ValueError):
            SegmentationDataset(
                split="bogus", root_dir=str(segmentation_data_dir), num_classes=1
            )

    def test_len_matches_number_of_image_mask_pairs(self, segmentation_data_dir):
        dataset = SegmentationDataset(
            split="train", root_dir=str(segmentation_data_dir), num_classes=1
        )

        assert len(dataset) == 3

    def test_binary_mask_is_thresholded(self, segmentation_data_dir):
        dataset = SegmentationDataset(
            split="train", root_dir=str(segmentation_data_dir), num_classes=1
        )

        for _, mask in dataset:
            assert set(np.unique(mask)).issubset({0, 1})

    def test_multiclass_mask_keeps_raw_values(self, segmentation_data_dir):
        dataset = SegmentationDataset(
            split="train", root_dir=str(segmentation_data_dir), num_classes=2
        )

        values = set()
        for _, mask in dataset:
            values.update(np.unique(mask).tolist())
        assert values.issubset({0, 1})

    def test_getitem_returns_matching_image_and_mask_shape(self, segmentation_data_dir):
        dataset = SegmentationDataset(
            split="train", root_dir=str(segmentation_data_dir), num_classes=1
        )

        img, mask = dataset[0]
        assert img.shape[:2] == mask.shape[:2]


class TestDataModuleAugmentationAndPreprocessing:
    def _make_classification_module(self, data_dir, **overrides):
        defaults = dict(
            data_dir=str(data_dir),
            size=[16, 16],
            imagenet_preprocessing=False,
            encoder=None,
        )
        defaults.update(overrides)
        return ClassificationDataModule(**defaults)

    def test_train_augmentations_include_optional_steps_only_when_enabled(
        self, classification_data_dir
    ):
        bare = self._make_classification_module(classification_data_dir)
        augmented = self._make_classification_module(
            classification_data_dir,
            random_horizontal_flip=True,
            random_vertical_flip=True,
            random_rotation=True,
        )

        assert len(augmented.get_train_augmentations().transforms) > len(
            bare.get_train_augmentations().transforms
        )

    def test_dev_and_test_augmentations_are_disabled(self, classification_data_dir):
        module = self._make_classification_module(classification_data_dir)

        assert module.get_dev_augmentations() is None
        assert module.get_test_augmentations() is None

    def test_preprocessing_resizes_and_converts_to_tensor(self, classification_data_dir):
        module = self._make_classification_module(classification_data_dir)
        img = np.zeros((32, 32, 3), dtype=np.uint8)

        result = module.get_preprocessing()(image=img)["image"]

        assert tuple(result.shape) == (3, 16, 16)

    def test_setup_fit_builds_train_and_dev_datasets(self, classification_data_dir):
        module = self._make_classification_module(classification_data_dir)

        module.setup(stage="fit")

        assert len(module.train_dataset) == 6
        assert len(module.dev_dataset) == 6

    def test_setup_fit_respects_sample_size(self, classification_data_dir):
        module = self._make_classification_module(classification_data_dir, sample_size=2)

        module.setup(stage="fit")

        assert len(module.train_dataset) == 2

    def test_setup_test_builds_test_dataset(self, classification_data_dir):
        module = self._make_classification_module(classification_data_dir)

        module.setup(stage="test")

        assert len(module.test_dataset) == 6

    def test_setup_predict_raises_not_implemented(self, classification_data_dir):
        module = self._make_classification_module(classification_data_dir)

        with pytest.raises(NotImplementedError):
            module.setup(stage="predict")


class TestSegmentationDataModule:
    def test_setup_fit_builds_train_and_dev_datasets(self, segmentation_data_dir):
        module = SegmentationDataModule(
            data_dir=str(segmentation_data_dir),
            size=[16, 16],
            imagenet_preprocessing=False,
            num_classes=1,
        )

        module.setup(stage="fit")

        assert len(module.train_dataset) == 3
        assert len(module.dev_dataset) == 3
