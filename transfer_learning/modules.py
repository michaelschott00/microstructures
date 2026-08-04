import lightning.pytorch as pl
import matplotlib
import torch
import torch.nn.functional as F
from sklearn.metrics import ConfusionMatrixDisplay
from torch import nn

matplotlib.use("Agg")  # crashed for other backends
from typing import Dict, Literal, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torchmetrics
import torchmetrics.classification
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from transfer_learning import util


def _image_to_numpy(image: torch.Tensor) -> np.ndarray:
    """Converts a single CHW image tensor to a HWC uint8 numpy array."""
    image = image.detach().cpu().clamp(0, 1)
    return (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _log_labeled_images(
    writer,
    tag: str,
    X: torch.Tensor,
    y: torch.Tensor,
    step: int,
    num_images: int = 8,
) -> None:
    """Logs individual images to tensorboard with the corresponding class label baked into the tag.

    Args:
        writer: a tensorboard SummaryWriter.
        tag: the base tag to log the images under.
        X: a batch of images of shape (N, C, H, W).
        y: a batch of integer class labels of shape (N,).
        step: the global step to log the images at.
        num_images: the maximum number of images to log.
    """
    n = min(num_images, X.shape[0])
    for i in range(n):
        writer.add_image(
            f"{tag}/{i}_label_{y[i].item()}",
            _image_to_numpy(X[i]),
            step,
            dataformats="HWC",
        )


def _log_mask_overlay_images(
    writer,
    tag: str,
    X: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int,
    step: int,
    num_images: int = 8,
) -> None:
    """Logs images with the segmentation mask overlaid, baked in as a colorized, alpha-blended
    layer using matplotlib's colormap, since tensorboard has no native mask-overlay support.

    Args:
        writer: a tensorboard SummaryWriter.
        tag: the base tag to log the images under.
        X: a batch of images of shape (N, C, H, W).
        mask: a batch of integer class-index masks of shape (N, H, W) or (N, 1, H, W).
        num_classes: the number of classes, used to build the class-label mapping.
        step: the global step to log the images at.
        num_images: the maximum number of images to log.
    """
    if mask.dim() == 4:
        mask = mask.squeeze(1)
    mask = mask.detach().cpu().long()

    cmap = plt.get_cmap("tab20", max(num_classes, 2))
    n = min(num_images, X.shape[0])
    for i in range(n):
        image = _image_to_numpy(X[i]).astype(np.float32)
        mask_rgb = (cmap(mask[i].numpy())[..., :3] * 255).astype(np.float32)
        overlay = (0.5 * image + 0.5 * mask_rgb).astype(np.uint8)
        writer.add_image(f"{tag}/{i}", overlay, step, dataformats="HWC")


class _EncoderClassifier(nn.Module):
    """Wraps a segmentation_models_pytorch encoder with global average pooling and a linear
    classification head, so it can be used as a classification model regardless of the spatial
    size of its feature maps."""

    def __init__(self, encoder: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(encoder.out_channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)[-1]  # smp encoders return the feature maps of every stage
        return self.fc(self.pool(features).flatten(1))


class ClassificationModule(pl.LightningModule):
    """A classification lightning module for classification.

    Args:
        encoder: The encoder to use. Make sure that the pretrained weights are available for it.
        num_classes: The number of classes to classify.
        pretrained_weights: The pretrained weights to use. Must be one of ['none', 'imagenet', 'micronet', 'image-micronet'].
        optimizer: The optimizer to use. Currently implemented are ['adamw', 'sgd'] although only adamw is tested.
        scheduler: The scheduler to use. Currently implemented are ['none', 'cosine', 'step'].
        lr: The learning rate to use. Can be a float or a dict with keys 'encoder' and 'other'.
        weight_decay: The weight decay to use.
        T_max: The T_max parameter for the cosine annealing scheduler.
        step_size: The step_size parameter for the step scheduler.
        gamma: The gamma parameter for the step scheduler.
        freeze_encoder_after_epoch: The epoch after which to freeze the encoder. Set to 0 to freeze in general.
        train_last: The number of layers to train counted from the end of the model.
    """

    def __init__(
        self,
        encoder: str,
        num_classes: int,
        pretrained_weights: Literal["none", "imagenet", "micronet", "image-micronet"],
        optimizer: Literal["adamw", "sgd"],
        scheduler: Literal["none", "cosine", "step"],
        lr: Union[
            float, Dict[str, float]
        ],  # dict allows different learning rates for encoder and classifier
        weight_decay: float | None = None,
        T_max: int | None = None,
        step_size: int | None = None,
        gamma: float | None = None,
        freeze_encoder_after_epoch: int | None = None,  # set to 0 to freeze in general
        train_last: int | None = None,
    ) -> None:
        super().__init__()

        assert not (
            pretrained_weights == "none" and freeze_encoder_after_epoch is not None
        ), "Freezing the encoder at random initialization is not useful."

        if isinstance(lr, dict):
            assert "encoder" in lr and "other" in lr, (
                "If lr is a dict, it must contain learning rates for encoder and head."
            )
            assert lr["encoder"] < lr["other"], (
                "Remove this assertion if you know what you are doing."
            )

        self.save_hyperparameters()

        self.model = self.create_classification_model(encoder, pretrained_weights)

        self.loss_func = nn.CrossEntropyLoss()

        # Metrics, we need separate classes for validation and test since they aggregate the statistics internally across batches

        # Accuracy
        self.val_acc = torchmetrics.classification.MulticlassAccuracy(
            num_classes=self.hparams["num_classes"], average="macro"
        )

        self.test_acc = torchmetrics.classification.MulticlassAccuracy(
            num_classes=self.hparams["num_classes"], average="macro"
        )

        # F1
        self.val_f1 = torchmetrics.classification.MulticlassF1Score(
            num_classes=self.hparams["num_classes"],
            average="macro",  # micro averages are all the same as accuracy, so we use macro
        )

        self.test_f1 = torchmetrics.classification.MulticlassF1Score(
            num_classes=self.hparams["num_classes"],
            average="macro",  # micro averages are all the same as accuracy, so we use macro
        )

        # Precision
        self.val_precision = torchmetrics.classification.MulticlassPrecision(
            num_classes=self.hparams["num_classes"], average="macro"
        )

        self.test_precision = torchmetrics.classification.MulticlassPrecision(
            num_classes=self.hparams["num_classes"], average="macro"
        )

        # Recall
        self.val_recall = torchmetrics.classification.MulticlassRecall(
            num_classes=self.hparams["num_classes"], average="macro"
        )

        self.test_recall = torchmetrics.classification.MulticlassRecall(
            num_classes=self.hparams["num_classes"], average="macro"
        )

        # Confusion matrix
        self.val_confmat = torchmetrics.classification.MulticlassConfusionMatrix(
            num_classes=self.hparams["num_classes"]
        )

        self.test_confmat = torchmetrics.classification.MulticlassConfusionMatrix(
            num_classes=self.hparams["num_classes"]
        )

    def create_classification_model(
        self, encoder: str, pretrained_weights: str
    ) -> nn.Module:
        """Builds an encoder via segmentation_models_pytorch's encoder registry and attaches a
        global-average-pool + linear classification head.

        Also takes care of freezing layers if specified.
        A further expansion of this function could allow initializing the last couple of layers randomly.

        Args:
            encoder: The encoder to use. Must be one of segmentation_models_pytorch's supported
                encoder names (see `smp.encoders.encoders`). Make sure that the pretrained weights
                are available for it.
            pretrained_weights: The pretrained weights to use. Must be one of ['none', 'imagenet', 'micronet', 'image-micronet']. If 'none', the model is initialized randomly.
        """
        if pretrained_weights not in ["none", "imagenet", "micronet", "image-micronet"]:
            raise NotImplementedError(
                f"Pretrained weights {pretrained_weights} are not supported."
            )

        smp_weights = "imagenet" if pretrained_weights == "imagenet" else None
        backbone = smp.encoders.get_encoder(encoder, weights=smp_weights)

        if pretrained_weights in ["micronet", "image-micronet"]:
            state_dict = util.load_micronet_weights(encoder, pretrained_weights)
            backbone.load_state_dict(
                state_dict, strict=False
            )  # strict=False ignores parameters that don't match the model

        model = _EncoderClassifier(backbone, self.hparams["num_classes"])

        if self.hparams["train_last"] is not None:
            util.freeze_encoder_layers(model, self.hparams["train_last"])

        return model

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure optimizers for encoder and classifier layers, allowing different learning rates using parameter groups."""
        if isinstance(self.hparams["lr"], dict):
            encoder_group = []
            classifier_group = []
            for name, parameter_groups in self.named_parameters():
                if not name.startswith("model.fc"):
                    encoder_group.append(
                        {
                            "params": parameter_groups,
                            "lr": self.hparams["lr"]["encoder"],
                        }
                    )
                else:
                    classifier_group.append(
                        {"params": parameter_groups, "lr": self.hparams["lr"]["other"]}
                    )
            parameter_groups = encoder_group + classifier_group
        else:
            parameter_groups = [{"params": self.parameters(), "lr": self.hparams["lr"]}]

        # treat optimizer as hyperparameter
        if self.hparams["optimizer"] == "adamw":
            optimizer = torch.optim.AdamW(
                parameter_groups, weight_decay=self.hparams["weight_decay"]
            )
        elif self.hparams["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(
                parameter_groups, weight_decay=self.hparams["weight_decay"]
            )
        else:
            raise NotImplementedError(
                f"Optimizer {self.hparams['optimizer']} is not yet supported."
            )

        if self.hparams["scheduler"] == "none":
            return optimizer
        elif self.hparams["scheduler"] == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams["T_max"]
            )
        elif self.hparams["scheduler"] == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.hparams["step_size"],
                gamma=self.hparams["gamma"],
            )
        else:
            raise NotImplementedError(
                f"Scheduler {self.hparams['scheduler']} is not yet supported."
            )

        return [optimizer], [scheduler]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        X, y = batch

        logits = self.forward(X)
        assert not logits.isnan().any().item(), "NaN logits"

        loss = self.loss_func(logits, y)

        self.log("loss/train", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        """Freezes the encoder after a certain number of epochs."""
        if self.current_epoch == self.hparams["freeze_encoder_after_epoch"]:
            util.freeze_encoder_layers(self.model, ignore_last=0)

    def validation_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        X, y = batch

        logits = self.forward(X)
        loss = self.loss_func(logits, y)

        self.val_acc(logits, y)
        self.val_f1(logits, y)
        self.val_precision(logits, y)
        self.val_recall(logits, y)

        self.val_confmat.update(logits, y)

        self.log("loss/validation", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "accuracy/validation",
            self.val_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "f1/validation", self.val_f1, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "precision/validation",
            self.val_precision,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "recall/validation",
            self.val_recall,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        return {"loss": loss, "logits": logits, "labels": y}

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        X, y = batch

        logits = self.forward(X)
        loss = self.loss_func(logits, y)

        self.test_acc(logits, y)
        self.test_f1(logits, y)
        self.test_precision(logits, y)
        self.test_recall(logits, y)

        self.test_confmat.update(logits, y)

        self.log("loss/test", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "accuracy/test", self.test_acc, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("f1/test", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "precision/test",
            self.test_precision,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "recall/test",
            self.test_recall,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        return {
            "loss": loss,
            "logits": logits,
            "labels": y,
        }

    def on_validation_epoch_end(self) -> None:
        assert isinstance(self.logger, TensorBoardLogger), (
            "This hook requires a TensorBoardLogger to be configured."
        )

        confmat = self.val_confmat.compute().cpu().numpy()

        fig = plt.figure()
        ConfusionMatrixDisplay(confmat).plot(ax=fig.gca())
        self.logger.experiment.add_figure(
            "confusion_matrix/validation", fig, self.current_epoch
        )
        plt.close(fig)

        self._last_val_confmat = confmat
        self.val_confmat.reset()

    def on_validation_end(self) -> None:
        """Caches the scalar validation metrics from this epoch for storing once training finishes."""
        self._last_val_metrics = {
            name: value.item()
            for name, value in self.trainer.callback_metrics.items()
            if value.numel() == 1
        }

    def on_fit_end(self) -> None:
        """Stores the validation metrics and confusion matrix from the final validation epoch."""
        self.trainer.store.store(
            self.trainer.run_id,
            {
                "val_metrics": pd.DataFrame([self._last_val_metrics]),
                "val_confmat": self._last_val_confmat,
            },
        )

    def on_train_batch_start(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        if (self.current_epoch == 0) and (batch_idx == 0):
            assert isinstance(self.logger, TensorBoardLogger), (
                "This hook requires a TensorBoardLogger to be configured."
            )
            X, y = batch
            _log_labeled_images(
                self.logger.experiment, "input/train", X, y, self.current_epoch
            )

    def on_validation_batch_start(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        if (self.current_epoch == 0) and (batch_idx == 0):
            assert isinstance(self.logger, TensorBoardLogger), (
                "This hook requires a TensorBoardLogger to be configured."
            )
            X, y = batch
            _log_labeled_images(
                self.logger.experiment, "input/validation", X, y, self.current_epoch
            )


class SegmentationModule(pl.LightningModule):
    """A PyTorch Lightning module for segmentation tasks. It is very similar to the ClassificationModule. So we try to repeat as little documentation
    as possible. Thus, refer to the ClassificationModule for more information on the individual methods.

    Args:
        architecture: The name of the architecture to use.
        encoder: The name of the encoder to use.
        num_classes: The number of classes to predict.
        pretrained_weights: The weights to use for the encoder. Can be "none", "imagenet", "micronet", "image-micronet". If "none", the encoder is randomly initialized.
        optimizer: The optimizer to use. Can be "adamw" or "sgd". Only "adamw" is tested.
        scheduler: The scheduler to use. Can be "none", "cosine" or "step". Only "none" is tested.
        lr: The learning rate to use. Can be a float or a dict with keys "encoder" and "other", where the latter allows different learning rates.
        T_max: The number of epochs to use for the cosine annealing scheduler.
        weight_decay: The weight decay to use.
        momentum: The momentum to use.
        gamma: The gamma to use for the step scheduler.
        step_size: The step size to use for the step scheduler.
        freeze_encoder_after_epoch: Whether to freeze the encoder after the given number of epochs.
        train_last: The number of layers to train in the encoder. If None, all layers are trained.
    """

    def __init__(
        self,
        architecture: str,
        encoder: str,
        num_classes: int,
        pretrained_weights: Literal["none", "imagenet", "micronet", "image-micronet"],
        optimizer: Literal["adamw", "sgd"],
        scheduler: Literal["none", "cosine", "step"],
        lr: Union[float, Dict[str, float]],
        T_max: int | None = None,
        weight_decay: float | None = None,
        momentum: int | None = None,
        gamma: float | None = None,
        step_size: int | None = None,
        freeze_encoder_after_epoch: bool = False,
        train_last: int | None = None,
    ) -> None:
        super().__init__()

        # input sanity checks
        assert not (pretrained_weights == "none" and freeze_encoder_after_epoch != 0), (
            "Freezing the encoder at random initialization is not useful."
        )

        if isinstance(lr, dict):
            assert "encoder" in lr and "other" in lr, (
                "If lr is a dict, it must contain learning rates for encoder and classifier."
            )
            assert lr["encoder"] < lr["other"], (
                "Remove this assertion if you know what you are doing."
            )

        self.save_hyperparameters()

        self.model = self.create_segmentation_model()

        if self.hparams["num_classes"] == 1:
            self.train_iou = torchmetrics.classification.BinaryJaccardIndex()
            self.val_iou = torchmetrics.classification.BinaryJaccardIndex()
            self.test_iou = torchmetrics.classification.BinaryJaccardIndex()
            self.loss_func = nn.BCEWithLogitsLoss()  # this allows using the logits directly without applying softmax first and is thus consistent with CrossEntropyLoss
        else:
            self.train_iou = torchmetrics.classification.MulticlassJaccardIndex(
                num_classes=self.hparams["num_classes"], average="macro"
            )
            self.val_iou = torchmetrics.classification.MulticlassJaccardIndex(
                num_classes=self.hparams["num_classes"], average="macro"
            )
            self.test_iou = torchmetrics.classification.MulticlassJaccardIndex(
                num_classes=self.hparams["num_classes"], average="macro"
            )
            self.loss_func = nn.CrossEntropyLoss()

        # keep track of a batch of training and validation data to visualize how masks change across epochs
        self.X_t_train, self.y_t_train = None, None
        self.X_t_dev, self.y_t_dev = None, None

    def create_segmentation_model(self) -> nn.Module:
        """Create a segmentation model with the specified encoder and backbone and initialize with specified pretrained weights.
        Also takes care of freezing encoder layers."""

        # smp supports `imagenet` out of the box, micronet must be loaded manually
        initial_weights = (
            "imagenet" if self.hparams["pretrained_weights"] == "imagenet" else None
        )

        model = getattr(smp, self.hparams["architecture"])(
            encoder_name=self.hparams["encoder"],
            encoder_weights=initial_weights,
            in_channels=3,
            classes=self.hparams["num_classes"],
        )

        if self.hparams["pretrained_weights"] in ["micronet", "image-micronet"]:
            state_dict = util.load_micronet_weights(
                self.hparams["encoder"], self.hparams["pretrained_weights"]
            )
            model.encoder.load_state_dict(state_dict)

        if self.hparams["train_last"] is not None:
            util.freeze_encoder_layers(model.encoder, self.hparams["train_last"])

        return model

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure optimizers for encoder and backbone layers, allowing different learning rates for each to avoid undoing pretraing while
        sufficiently training the classifier."""

        # create parameter groups for encoder and backbone
        if isinstance(self.hparams["lr"], dict):
            encoder_group = []
            backbone_group = []
            for name, parameter_groups in self.named_parameters():
                if "encoder" in name:
                    encoder_group.append(
                        {
                            "params": parameter_groups,
                            "lr": self.hparams["lr"]["encoder"],
                        }
                    )
                else:
                    backbone_group.append(
                        {"params": parameter_groups, "lr": self.hparams["lr"]["other"]}
                    )
            parameter_groups = encoder_group + backbone_group
        else:
            parameter_groups = [{"params": self.parameters(), "lr": self.hparams["lr"]}]

        if self.hparams["optimizer"] == "adamw":
            optimizer = torch.optim.AdamW(
                parameter_groups, weight_decay=self.hparams["weight_decay"]
            )
        elif self.hparams["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(
                parameter_groups,
                weight_decay=self.hparams["weight_decay"],
                momentum=self.hparams["momentum"],
            )
        else:
            raise NotImplementedError(
                f"Optimizer {self.hparams['optimizer']} is not yet supported."
            )

        if self.hparams["scheduler"] == "none":
            return optimizer

        if self.hparams["scheduler"] == "step":
            assert (
                self.hparams["step_size"] is not None
                and self.hparams["gamma"] is not None
            ), "step_size and gamma must be specified for step LR"
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.hparams["step_size"],
                gamma=self.hparams["gamma"],
            )
        elif self.hparams["scheduler"] == "cosine":
            assert self.hparams["T_max"] is not None, (
                "T_max must be specified for cosine annealing"
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams["T_max"]
            )
        else:
            raise NotImplementedError(
                f"Scheduler {self.hparams['scheduler']} is not yet supported."
            )

        return [optimizer], [scheduler]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def predict_full_image(self, image: torch.Tensor) -> torch.Tensor:
        """Predicts the logits for a single, full-sized image by tiling it into patches of the size
        the model was trained on (`SegmentationDataModule`'s `size` parameter), running the model on
        each tile, and stitching the resulting logits back together.

        Args:
            image: a single preprocessed image of shape (C, H, W).
        """
        tile_size = self.trainer.datamodule.hparams["size"]  # type: ignore
        return util.tile_and_predict(
            predict_fn=self.forward,
            image=image,
            tile_size=tile_size,
            num_classes=self.hparams["num_classes"],
        )

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        X, y = batch

        if self.hparams["num_classes"] > 1:
            y = y.squeeze(1).long()

        if batch_idx == 0 and self.current_epoch == 0:
            n = min(5, len(X))
            self.X_t_train, self.y_t_train = X.detach().cpu()[:n], y.detach().cpu()[:n]

        logits = self.forward(X)
        loss = self.loss_func(logits, y)

        self.train_iou(logits, y)

        self.log("loss/train", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log(
            "iou/train", self.train_iou, on_step=False, on_epoch=True, prog_bar=True
        )

        return loss

    def validation_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        X, y = batch
        assert X.shape[0] == 1, "full-image evaluation requires a batch size of 1"

        if self.hparams["num_classes"] > 1:
            y = y.squeeze(1).long()

        if batch_idx == 0 and self.current_epoch == 0:
            self.X_t_dev, self.y_t_dev = X.detach().cpu(), y.detach().cpu()

        logits = self.predict_full_image(X[0]).unsqueeze(0)
        loss = self.loss_func(logits, y)

        self.val_iou(logits, y)

        self.log("loss/validation", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "iou/validation", self.val_iou, on_step=False, on_epoch=True, prog_bar=True
        )

        return {"loss": loss, "logits": logits, "labels": y}

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        X, y = batch
        assert X.shape[0] == 1, "full-image evaluation requires a batch size of 1"

        if self.hparams["num_classes"] > 1:
            y = y.squeeze(1).long()

        if batch_idx == 0 and self.current_epoch == 0:
            self.X_t_dev, self.y_t_dev = X.detach().cpu(), y.detach().cpu()

        logits = self.predict_full_image(X[0]).unsqueeze(0)
        loss = self.loss_func(logits, y)

        self.test_iou(logits, y)

        self.log("loss/test", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("iou/test", self.test_iou, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "logits": logits, "labels": y}

    def on_train_batch_start(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        if (self.current_epoch == 0) and (batch_idx == 0):
            assert isinstance(self.logger, TensorBoardLogger), (
                "This hook requires a TensorBoardLogger to be configured."
            )
            X, y = batch
            _log_mask_overlay_images(
                self.logger.experiment,
                "input/train",
                X,
                y,
                self.hparams["num_classes"],
                self.current_epoch,
            )

    def on_validation_batch_start(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        if (self.current_epoch == 0) and (batch_idx == 0):
            assert isinstance(self.logger, TensorBoardLogger), (
                "This hook requires a TensorBoardLogger to be configured."
            )
            X, y = batch
            _log_mask_overlay_images(
                self.logger.experiment,
                "input/validation",
                X,
                y,
                self.hparams["num_classes"],
                self.current_epoch,
            )

    def on_validation_end(self) -> None:
        """Caches the scalar validation metrics from this epoch for storing once training finishes."""
        self._last_val_metrics = {
            name: value.item()
            for name, value in self.trainer.callback_metrics.items()
            if value.numel() == 1
        }

    def on_fit_end(self) -> None:
        """Stores the validation metrics from the final validation epoch."""
        self.trainer.store.store(
            self.trainer.run_id,
            {"val_metrics": pd.DataFrame([self._last_val_metrics])},
        )

    def on_train_epoch_end(self) -> None:
        """We allow freezing the encoder after a certain number of epochs. Also, we log the predicted masks of the model"""
        assert isinstance(self.logger, TensorBoardLogger), (
            "This hook requires a TensorBoardLogger to be configured."
        )
        assert self.X_t_train is not None, "X_t_train"
        assert self.X_t_dev is not None, "X_t_dev"
        assert self.y_t_train is not None, "y_t_train"
        assert self.y_t_dev is not None, "y_t_dev"

        if self.current_epoch == self.hparams["freeze_encoder_after_epoch"]:
            assert isinstance(self.model.encoder, nn.Module), type(self.model.encoder)
            util.freeze_encoder_layers(self.model.encoder, ignore_last=0)

        # visualize predicted masks
        self.X_t_train = self.X_t_train.to(self.device)
        self.X_t_dev = self.X_t_dev.to(self.device)

        with torch.no_grad():
            # X_t_dev is a full-sized image, so it must go through tiled inference just like in validation_step
            dev_logits = self.predict_full_image(self.X_t_dev[0]).unsqueeze(0).cpu()
            if self.hparams["num_classes"] == 1:
                pred_train = (
                    F.sigmoid(self.forward(self.X_t_train).detach().cpu()) > 0.5
                ).float()
                pred_dev = (F.sigmoid(dev_logits) > 0.5).float()
            else:
                pred_train = (
                    F.softmax(self.forward(self.X_t_train).detach().cpu(), dim=1)
                    .argmax(1)
                    .float()
                )
                pred_dev = F.softmax(dev_logits, dim=1).argmax(1).float()

        num_classes = self.hparams["num_classes"]

        if self.current_epoch == 0:
            _log_mask_overlay_images(
                self.logger.experiment,
                "predictions/train_ground_truth",
                self.X_t_train.cpu(),
                self.y_t_train,
                num_classes,
                self.current_epoch,
            )
            _log_mask_overlay_images(
                self.logger.experiment,
                "predictions/validation_ground_truth",
                self.X_t_dev.cpu(),
                self.y_t_dev,
                num_classes,
                self.current_epoch,
            )

        _log_mask_overlay_images(
            self.logger.experiment,
            "predictions/train",
            self.X_t_train.cpu(),
            pred_train,
            num_classes,
            self.current_epoch,
        )
        _log_mask_overlay_images(
            self.logger.experiment,
            "predictions/validation",
            self.X_t_dev.cpu(),
            pred_dev,
            num_classes,
            self.current_epoch,
        )
