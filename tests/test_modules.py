import pytest

from transfer_learning.modules import ClassificationModule, SegmentationModule


class TestClassificationModuleValidation:
    @pytest.mark.parametrize("freeze_encoder_after_epoch", [0, 1])
    def test_rejects_freezing_encoder_with_random_init(self, freeze_encoder_after_epoch):
        with pytest.raises(AssertionError):
            ClassificationModule(
                encoder="resnet18",
                num_classes=2,
                pretrained_weights="none",
                optimizer="adamw",
                scheduler="none",
                lr=1e-3,
                freeze_encoder_after_epoch=freeze_encoder_after_epoch,
            )

    def test_rejects_lr_dict_missing_keys(self):
        with pytest.raises(AssertionError):
            ClassificationModule(
                encoder="resnet18",
                num_classes=2,
                pretrained_weights="imagenet",
                optimizer="adamw",
                scheduler="none",
                lr={"encoder": 1e-4},
            )

    def test_rejects_lr_dict_with_encoder_lr_not_smaller(self):
        with pytest.raises(AssertionError):
            ClassificationModule(
                encoder="resnet18",
                num_classes=2,
                pretrained_weights="imagenet",
                optimizer="adamw",
                scheduler="none",
                lr={"encoder": 1e-2, "other": 1e-3},
            )


class TestSegmentationModuleValidation:
    def test_rejects_freezing_encoder_with_random_init(self):
        with pytest.raises(AssertionError):
            SegmentationModule(
                architecture="Unet",
                encoder="resnet18",
                num_classes=1,
                pretrained_weights="none",
                optimizer="adamw",
                scheduler="none",
                lr=1e-3,
                freeze_encoder_after_epoch=1,
            )

    def test_rejects_lr_dict_missing_keys(self):
        with pytest.raises(AssertionError):
            SegmentationModule(
                architecture="Unet",
                encoder="resnet18",
                num_classes=1,
                pretrained_weights="imagenet",
                optimizer="adamw",
                scheduler="none",
                lr={"other": 1e-3},
            )

    def test_rejects_lr_dict_with_encoder_lr_not_smaller(self):
        with pytest.raises(AssertionError):
            SegmentationModule(
                architecture="Unet",
                encoder="resnet18",
                num_classes=1,
                pretrained_weights="imagenet",
                optimizer="adamw",
                scheduler="none",
                lr={"encoder": 1e-2, "other": 1e-3},
            )
