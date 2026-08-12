from pathlib import Path

import pytest
import torch
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


class TestOptimizersAndSchedulers:
    """Checks that `configure_optimizers` can instantiate every supported optimizer and scheduler."""

    _BASE_KWARGS = {
        "encoder": "resnet18",
        "pretrained_weights": "none",
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "T_max": 10,
        "step_size": 5,
        "gamma": 0.9,
    }

    @staticmethod
    def _assert_configured(module, optimizer, scheduler):
        configured = module.configure_optimizers()
        if scheduler == "none":
            optim, sched = configured, None
        else:
            optims, scheds = configured
            optim, sched = optims[0], scheds[0]

        expected_optimizer = (
            torch.optim.AdamW if optimizer == "adamw" else torch.optim.SGD
        )
        assert isinstance(optim, expected_optimizer)

        if scheduler == "none":
            assert sched is None
        elif scheduler == "cosine":
            assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)
        else:
            assert isinstance(sched, torch.optim.lr_scheduler.StepLR)

    @pytest.mark.parametrize("optimizer", ["adamw", "sgd"])
    @pytest.mark.parametrize("scheduler", ["none", "cosine", "step"])
    def test_classification_optimizers_and_schedulers(self, optimizer, scheduler):
        module = ClassificationModule(
            optimizer=optimizer,
            scheduler=scheduler,
            num_classes=2,
            **self._BASE_KWARGS,
        )
        self._assert_configured(module, optimizer, scheduler)

    @pytest.mark.parametrize("optimizer", ["adamw", "sgd"])
    @pytest.mark.parametrize("scheduler", ["none", "cosine", "step"])
    def test_segmentation_optimizers_and_schedulers(self, optimizer, scheduler):
        module = SegmentationModule(
            architecture="Unet",
            optimizer=optimizer,
            scheduler=scheduler,
            num_classes=1,
            momentum=0.9,
            **self._BASE_KWARGS,
        )
        self._assert_configured(module, optimizer, scheduler)


class TestForwardPass:
    def test_classification_module_forward(self):
        module = ClassificationModule(
            encoder="resnet18",
            num_classes=3,
            pretrained_weights="none",
            optimizer="adamw",
            scheduler="none",
            lr=1e-3,
        )
        x = torch.randn(2, 3, 224, 224)
        logits = module(x)
        assert logits.shape == (2, 3)

    def test_segmentation_module_forward(self):
        module = SegmentationModule(
            architecture="Unet",
            encoder="resnet18",
            num_classes=1,
            pretrained_weights="none",
            optimizer="adamw",
            scheduler="none",
            lr=1e-3,
        )
        x = torch.randn(2, 3, 224, 224)
        logits = module(x)
        assert logits.shape == (2, 1, 224, 224)


class TestClassificationModuleValidation:
    @pytest.mark.parametrize("freeze_encoder_after_epoch", [0, 1])
    def test_rejects_freezing_encoder_with_random_init(self, freeze_encoder_after_epoch):
        with pytest.raises(AssertionError):
            ClassificationModule(
                encoder="resnet50",
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
                encoder="resnet50",
                num_classes=2,
                pretrained_weights="imagenet",
                optimizer="adamw",
                scheduler="none",
                lr={"encoder": 1e-4},
            )

    def test_rejects_lr_dict_with_encoder_lr_not_smaller(self):
        with pytest.raises(AssertionError):
            ClassificationModule(
                encoder="resnet50",
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
                encoder="resnet50",
                num_classes=1,
                pretrained_weights="none",
                optimizer="adamw",
                scheduler="none",
                lr=1e-3,
                freeze_encoder_after_epoch=1,  # type: ignore
            )

    def test_rejects_lr_dict_missing_keys(self):
        with pytest.raises(AssertionError):
            SegmentationModule(
                architecture="Unet",
                encoder="resnet50",
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
                encoder="resnet50",
                num_classes=1,
                pretrained_weights="imagenet",
                optimizer="adamw",
                scheduler="none",
                lr={"encoder": 1e-2, "other": 1e-3},
            )