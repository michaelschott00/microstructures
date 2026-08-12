from pathlib import Path

import pytest
from omegaconf import OmegaConf

from transfer_learning.modules import ClassificationModule, SegmentationModule

_CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def _load(path: Path) -> dict:
    return dict(OmegaConf.load(path))


def _model_init_args(module: str) -> list[dict]:
    """Collects all composed `model.init_args` combinations that appear in the configs.

    A fully-specified model config is built by merging the model, optimization, pretraining and
    task configs the same way the training pipeline composes them.
    """
    model_dir = _CONFIGS_DIR / "models" / module
    optimization = _load(_CONFIGS_DIR / "optimization" / "adamw_basic.yaml")["model"][
        "init_args"
    ]
    pretraining = sorted((_CONFIGS_DIR / "pretraining").glob("*.yaml"))
    task_files = {
        "classification": "classification_1.yaml",
        "segmentation": "segmentation_1.yaml",
    }
    task = _load(
        _CONFIGS_DIR / "task" / task_files[module]
    )["model"]["init_args"]

    combos = []
    for model_file in sorted(model_dir.glob("*.yaml")):
        model_init = _load(model_file)["model"]["init_args"]
        for pretraining_file in pretraining:
            # same composition order as the training CLI: task, model, optimization, pretraining
            combos.append(
                {
                    **task,
                    **model_init,
                    **optimization,
                    **_load(pretraining_file)["model"]["init_args"],
                }
            )
    return combos


class TestModulesInstantiation:
    def test_classification_module_instantiates(self):
        for init_args in _model_init_args("classification"):
            ClassificationModule(**init_args)

    def test_segmentation_module_instantiates(self):
        for init_args in _model_init_args("segmentation"):
            SegmentationModule(**init_args)


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