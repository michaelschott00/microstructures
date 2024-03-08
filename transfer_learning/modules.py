import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg')  # crashed for other backends
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torchmetrics
from typing import Dict, Union, Literal, List, Tuple, Optional
from transfer_learning import util


class ClassificationModule(pl.LightningModule):
    """A classification lightning module for classification.

    Args:
        encoder: The encoder to use. Make sure that the pretrained weights are available for it.
        num_classes: The number of classes to classify.
        pretrained_weights: The pretrained weights to use. Must be one of ['none', 'imagenet', 'micronet', 'image-micronet'].
        optimizer: The optimizer to use. Currently implemented are ['adamw', 'sgd'] although only adamw is tested.
        scheduler: The scheduler to use. Currently implemented are ['none', 'cosine', 'step'] although none of them are tested.
        lr: The learning rate to use. Can be a float or a dict with keys 'encoder' and 'other'.
        weight_decay: The weight decay to use.
        T_max: The T_max parameter for the cosine annealing scheduler.
        step_size: The step_size parameter for the step scheduler.
        gamma: The gamma parameter for the step scheduler.
        freeze_encoder_after_epoch: The epoch after which to freeze the encoder. Set to 0 to freeze in general.
        train_last: The number of layers to train counted from the end of the model.
    """

    def __init__(self,
                 encoder: str,
                 num_classes: int,
                 pretrained_weights: Literal["none", "imagenet", "micronet", "image-micronet"],
                 optimizer: Literal["adamw", "sgd"],
                 scheduler: Literal["none", "cosine", "step"],
                 lr: Union[float, Dict[str, float]],  # dict allows different learning rates for encoder and classifier
                 weight_decay: float = None,
                 T_max: int = None,
                 step_size: int = None,
                 gamma: float = None,
                 freeze_encoder_after_epoch: int = None,  # set to 0 to freeze in general
                 train_last: int = None) -> None:
        super().__init__()

        assert not (pretrained_weights == "none" and freeze_encoder_after_epoch), "Freezing the encoder at random initialization is not useful."

        if isinstance(lr, dict):
            assert 'encoder' in lr and 'other' in lr, "If lr is a dict, it must contain learning rates for encoder and head."
            assert lr['encoder'] < lr['other'], "Remove this assertion if you know what you are doing."

        # store all arguments to the constructor to self.hparams and to tensorboard
        self.save_hyperparameters()

        self.model = self.create_classification_model(encoder, pretrained_weights)

        self.loss_func = nn.CrossEntropyLoss()

        # Metrics, we need separate classes for validation and test since they aggregate the statistics internally across batches

        # Accuracy
        self.val_acc = torchmetrics.classification.MulticlassAccuracy(
            num_classes=self.hparams.num_classes,
            average="macro"
        )

        self.test_acc = torchmetrics.classification.MulticlassAccuracy(
            num_classes=self.hparams.num_classes,
            average="macro"
        )

        # F1
        self.val_f1 = torchmetrics.classification.MulticlassF1Score(
            num_classes=self.hparams.num_classes,
            average="macro"  # micro averages are all the same as accuracy, so we use macro
        )

        self.test_f1 = torchmetrics.classification.MulticlassF1Score(
            num_classes=self.hparams.num_classes,
            average="macro"  # micro averages are all the same as accuracy, so we use macro
        )

        # Precision
        self.val_precision = torchmetrics.classification.MulticlassPrecision(
            num_classes=self.hparams.num_classes,
            average="macro"
        )

        self.test_precision = torchmetrics.classification.MulticlassPrecision(
            num_classes=self.hparams.num_classes,
            average="macro"
        )

        # Recall
        self.val_recall = torchmetrics.classification.MulticlassRecall(
            num_classes=self.hparams.num_classes,
            average="macro"
        )

        self.test_recall = torchmetrics.classification.MulticlassRecall(
            num_classes=self.hparams.num_classes,
            average="macro"
        )

        # Confusion matrix
        self.val_confmat = torchmetrics.classification.MulticlassConfusionMatrix(
            num_classes=self.hparams.num_classes
        )

        self.test_confmat = torchmetrics.classification.MulticlassConfusionMatrix(
            num_classes=self.hparams.num_classes
        )

    def create_classification_model(self, encoder: str, pretrained_weights: str) -> nn.Module:
        """Loads the encoder model and replaces the last layer with a linear layer with the correct number of classes.

        Also takes care of freezing layers if specified.
        A further expansion of this function could allow initializing the last couple of layers randomly.

        Args:
            encoder: The encoder to use. Make sure that the pretrained weights are available for it.
            pretrained_weights: The pretrained weights to use. Must be one of ['none', 'imagenet', 'micronet', 'image-micronet']. If 'none', the model is initialized randomly.
        """
        if pretrained_weights == "none":
            model = torch.hub.load('pytorch/vision:v0.6.0', encoder, weights=None)
        elif pretrained_weights == "imagenet":
            model = torch.hub.load('pytorch/vision:v0.6.0', encoder, weights='DEFAULT')  # DEFAULT usually specifies imagenet weights
        elif pretrained_weights in ["micronet", "image-micronet"]:
            model = torch.hub.load('pytorch/vision:v0.6.0', encoder, weights=None)
            state_dict = util.load_micronet_weights(encoder, pretrained_weights)
            model.load_state_dict(state_dict, strict=False)  # strict=False ignores parameters that don't match the model
        else:
            raise NotImplementedError(f"Pretrained weights {pretrained_weights} are not supported.")

        if self.hparams.train_last is not None:
            util.freeze_encoder_layers(model, self.hparams.train_last)

        # different architectures name their classification head differently
        if "resnet" in encoder:
            model.fc = nn.Linear(model.fc.in_features, self.hparams.num_classes)
        elif "vgg" in encoder:
            model.classifier = nn.Linear(model.classifier[0].in_features, self.hparams.num_classes)
        else:
            raise NotImplementedError(f"Encoder architecture {encoder} needs some work to be supported. Please update the encoder creation function where this assertion is thrown.")

        return model

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], Optional[List[torch.optim.lr_scheduler._LRScheduler]]]:
        """ Configure optimizers for encoder and classifier layers, allowing different learning rates using parameter groups."""
        if isinstance(self.hparams.lr, dict):
            encoder_group = []
            classifier_group = []
            for name, parameter_groups in self.named_parameters():
                if not ('fc' in name or 'classifier' in name):
                    encoder_group.append({"params": parameter_groups, 'lr': self.hparams.lr['encoder']})
                else:
                    classifier_group.append({"params": parameter_groups, 'lr': self.hparams.lr['other']})
            parameter_groups = encoder_group + classifier_group
        else:
            parameter_groups = [{'params': self.parameters(), 'lr': self.hparams.lr}]

        # treat optimizer as hyperparameter
        if self.hparams.optimizer == "adamw":
            optimizer = torch.optim.AdamW(parameter_groups, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(parameter_groups, weight_decay=self.hparams.weight_decay)
        else:
            raise NotImplementedError(f"Optimizer {self.hparams.optimizer} is not yet supported.")

        if self.hparams.scheduler == "none":
            return optimizer  # we can not return [optimizer], [None] or [optimizer], []
        elif self.hparams.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.T_max)
        elif self.hparams.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma)
        else:
            raise NotImplementedError(f"Scheduler {self.hparams.scheduler} is not yet supported.")

        return [optimizer], [scheduler]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Handles a single training step, i.e. forward pass and loss function computation. Also logs the loss.
        This is called by lightning and must always return the loss."""
        X, y = batch

        logits = self.forward(X)
        assert not logits.isnan().any().item(), "NaN logits"  # just to be sure

        loss = self.loss_func(logits, y)

        # log loss to tensorboard
        self.log("loss/train", loss, on_step=True, on_epoch=False, prog_bar=True)

        return loss  # must return loss and only loss, since this is passed internally to the optimization step

    def on_train_epoch_end(self):
        """A hook that is called by lighting after every training epoch.
        Here, it is used to freeze the encoder after a certain number of epochs but can be arbitrarily expanded."""
        if self.current_epoch == self.hparams.freeze_encoder_after_epoch:
            # freezing after several epochs only supports freezing the full encoder at the moment, this could be extended if necessary
            util.freeze_encoder_layers(self.model, ignore_last=0)

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Handles a single validation step, i.e. forward pass and metric computation. Also computes and logs the metrics. Can be extended to return arbitrary values."""
        X, y = batch

        logits = self.forward(X)
        loss = self.loss_func(logits, y)

        self.val_acc(logits, y)
        self.val_f1(logits, y)
        self.val_precision(logits, y)
        self.val_recall(logits, y)

        # confusion matrix does not support self.log(self.val_confmat), so we need to update it here manually
        self.val_confmat.update(logits, y)

        # log other metrics to tensorboard
        self.log("loss/validation", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("accuracy/validation", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("f1/validation", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("precision/validation", self.val_precision, on_step=False, on_epoch=True, prog_bar=False)
        self.log("recall/validation", self.val_recall, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "logits": logits, "labels": y}

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Handles a single test step, i.e. forward pass and metric computation. Also computes and logs the metrics. Can be extended to return arbitrary values."""
        X, y = batch

        logits = self.forward(X)
        loss = self.loss_func(logits, y)

        self.test_acc(logits, y)
        self.test_f1(logits, y)
        self.test_precision(logits, y)
        self.test_recall(logits, y)

        self.test_confmat.update(logits, y)

        # log metrics to tensorboard
        self.log("loss/test", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("accuracy/test", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("f1/test", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("precision/test", self.test_precision, on_step=False, on_epoch=True, prog_bar=False)
        self.log("recall/test", self.test_recall, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "logits": logits, "labels": y}  # here we can essentially return whatever we want

    def on_validation_epoch_end(self) -> None:
        """A hook that is called by lighting after every validation epoch.
        Here, it is used to compute the confusion matrix and log it. Can be arbitrarily expanded.
        """
        fig = plt.figure()
        ConfusionMatrixDisplay(self.val_confmat.compute().cpu().numpy()).plot(ax=fig.gca())
        self.logger.experiment.add_figure("confusion matrix/validation", fig, self.current_epoch)

        # reset confusion matrix MUST be called, otherwise metric as aggregated across validation runs.
        self.val_confmat.reset()

    def on_train_batch_start(self, batch: torch.Tensor, batch_idx: int) -> None:
        """A hook that is called by lighting before every training batch.
        Here, it is used to log the first input batch to ensure that the model receives sane inputs."""
        if (self.current_epoch == 0) and (batch_idx == 0):
            X, y = batch
            self.logger.experiment.add_image("input/train", torchvision.utils.make_grid(X))

    def on_validation_batch_start(self, batch: torch.Tensor, batch_idx: int) -> None:
        """A hook that is called by lighting before every validation batch.
        Here, it is used to log the first input batch to ensure that the model receives sane inputs."""
        if (self.current_epoch == 0) and (batch_idx == 0):
            X, y = batch
            self.logger.experiment.add_image("input/validation", torchvision.utils.make_grid(X))


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

    def __init__(self,
                 architecture: str,
                 encoder: str,
                 num_classes: int,
                 pretrained_weights: Literal["none", "imagenet", "micronet", "image-micronet"],
                 optimizer: Literal["adamw", "sgd"],
                 scheduler: Literal["none", "cosine", "step"],
                 lr: Union[float, Dict[str, float]],
                 T_max: int = None,
                 weight_decay: float = None,
                 momentum: int = None,
                 gamma: float = None,
                 step_size: int = None,
                 freeze_encoder_after_epoch: bool = False,
                 train_last: int = None) -> None:
        super().__init__()

        # input sanity checks
        assert not (pretrained_weights == "none" and freeze_encoder_after_epoch != 0), "Freezing the encoder at random initialization is not useful."

        if isinstance(lr, dict):
            assert 'encoder' in lr and 'other' in lr, "If lr is a dict, it must contain learning rates for encoder and classifier."
            assert lr['encoder'] < lr['other'], "Remove this assertion if you know what you are doing."

        # store arguments of constructor to self.hparams and tensorboard
        self.save_hyperparameters()

        self.model = self.create_segmentation_model()

        if self.hparams.num_classes == 1:
            self.train_iou = torchmetrics.classification.BinaryJaccardIndex()
            self.val_iou = torchmetrics.classification.BinaryJaccardIndex()
            self.test_iou = torchmetrics.classification.BinaryJaccardIndex()
            self.loss_func = nn.BCEWithLogitsLoss()  # this allows using the logits directly without applying softmax first and is thus consistent with CrossEntropyLoss
        else:
            self.train_iou = torchmetrics.classification.MulticlassJaccardIndex(num_classes=self.hparams.num_classes, average='macro')
            self.val_iou = torchmetrics.classification.MulticlassJaccardIndex(num_classes=self.hparams.num_classes, average='macro')
            self.test_iou = torchmetrics.classification.MulticlassJaccardIndex(num_classes=self.hparams.num_classes, average='macro')
            self.loss_func = nn.CrossEntropyLoss()

        # keep track of a batch of training and validation data to visualize how masks change across epochs
        self.X_t_train, self.y_t_train = None, None
        self.X_t_dev, self.y_t_dev = None, None

    def create_segmentation_model(self) -> nn.Module:
        """ Create a segmentation model with the specified encoder and backbone and initialize with specified pretrained weights.
        Also takes care of freezing encoder layers."""

        # smp supports `imagenet` out of the box, micronet must be loaded manually
        initial_weights = 'imagenet' if self.hparams.pretrained_weights == 'imagenet' else None

        model = getattr(smp, self.hparams.architecture)(
            encoder_name=self.hparams.encoder,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=initial_weights,
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=self.hparams.num_classes,  # model output channels (number of classes in your dataset)
        )

        if self.hparams.pretrained_weights in ["micronet", "image-micronet"]:
            state_dict = util.load_micronet_weights(self.hparams.encoder, self.hparams.pretrained_weights)
            model.encoder.load_state_dict(state_dict)

        if self.hparams.train_last is not None:
            util.freeze_encoder_layers(model.encoder, self.hparams.train_last)

        return model

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], Optional[List[torch.optim.lr_scheduler._LRScheduler]]]:
        """ Configure optimizers for encoder and backbone layers, allowing different learning rates for each to avoid undoing pretraing while
        sufficiently training the classifier. """

        assert self.hparams.optimizer in ["adamw", "sgd"], f"{self.hparams.optimizer} is not supported"
        assert self.hparams.scheduler in ["none", "cosine", "step"], f"{self.hparams.scheduler} is not supported"

        # create parameter groups for encoder and backbone
        if isinstance(self.hparams.lr, dict):
            encoder_group = []
            backbone_group = []
            for name, parameter_groups in self.named_parameters():
                if "encoder" in name:
                    encoder_group.append({"params": parameter_groups, 'lr': self.hparams.lr['encoder']})
                else:
                    backbone_group.append({"params": parameter_groups, 'lr': self.hparams.lr['other']})
            parameter_groups = encoder_group + backbone_group
        else:
            parameter_groups = [{"params": self.parameters(), 'lr': self.hparams.lr}]

        if self.hparams.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(parameter_groups, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'sgd':
            optimizer = torch.optim.SGD(parameter_groups, weight_decay=self.hparams.weight_decay, momentum=self.hparams.momentum)

        if self.hparams.scheduler == 'none':
            return optimizer

        if self.hparams.scheduler == 'step':
            assert self.hparams.step_size is not None and self.hparams.gamma is not None, "step_size and gamma must be specified for step LR"
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma)
        elif self.hparams.scheduler == 'cosine':
            assert self.hparams.T_max is not None, "T_max must be specified for cosine annealing"
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.T_max)

        return [optimizer], [scheduler]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        X, y = batch

        if self.hparams.num_classes > 1:
            y = y.squeeze(1).long()  # magically makes the dimensions work out, I think

        if batch_idx == 0 and self.current_epoch == 0:
            n = min(5, len(X))
            self.X_t_train, self.y_t_train = X.detach().cpu()[:n], y.detach().cpu()[:n]

        logits = self.forward(X)
        loss = self.loss_func(logits, y)

        self.train_iou(logits, y)

        self.log('loss/train', loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('iou/train', self.train_iou, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        X, y = batch

        if self.hparams.num_classes > 1:
            y = y.squeeze(1).long()

        if batch_idx == 0 and self.current_epoch == 0:
            n = min(5, len(X))
            self.X_t_dev, self.y_t_dev = X.detach().cpu()[:n], y.detach().cpu()[:n]

        logits = self.forward(X)
        loss = self.loss_func(logits, y)

        self.val_iou(logits, y)

        self.log("loss/validation", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('iou/validation', self.val_iou, on_step=False, on_epoch=True, prog_bar=True)

        return {'loss': loss, 'logits': logits, 'labels': y}

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        X, y = batch

        if self.hparams.num_classes > 1:
            y = y.squeeze(1).long()

        if batch_idx == 0 and self.current_epoch == 0:
            n = min(5, len(X))
            self.X_t_dev, self.y_t_dev = X.detach().cpu()[:n], y.detach().cpu()[:n]

        logits = self.forward(X)
        loss = self.loss_func(logits, y)

        self.test_iou(logits, y)

        self.log("loss/test", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('iou/test', self.test_iou, on_step=False, on_epoch=True, prog_bar=True)

        return {'loss': loss, 'logits': logits, 'labels': y}

    def on_train_batch_start(self, batch: torch.Tensor, batch_idx: int) -> None:
        if (self.current_epoch == 0) and (batch_idx == 0):
            X, y = batch
            self.logger.experiment.add_image("input/train_imgs", torchvision.utils.make_grid(X))
            self.logger.experiment.add_image("input/train_masks", torchvision.utils.make_grid(y))

    def on_validation_batch_start(self, batch: torch.Tensor, batch_idx: int) -> None:
        if (self.current_epoch == 0) and (batch_idx == 0):
            X, y = batch
            self.logger.experiment.add_image("input/validation_imgs", torchvision.utils.make_grid(X))
            self.logger.experiment.add_image("input/validation_masks", torchvision.utils.make_grid(y))

    def on_train_epoch_end(self) -> None:
        """We allow freezing the encoder after a certain number of epochs. Also, we log the predicted masks of the model"""
        if self.current_epoch == self.hparams.freeze_encoder_after_epoch:
            util.freeze_encoder_layers(self.model.encoder, ignore_last=0)

        # visualize predicted masks
        self.X_t_train = self.X_t_train.to(self.device)
        self.X_t_dev = self.X_t_dev.to(self.device)

        with torch.no_grad():
            if self.hparams.num_classes == 1:
                pred_train = (F.sigmoid(self.forward(self.X_t_train).detach().cpu()) > 0.5).float()
                pred_dev = (F.sigmoid(self.forward(self.X_t_dev).detach().cpu()) > 0.5).float()
            else:
                pred_train = F.softmax(self.forward(self.X_t_train, dim=1).detach().cpu()).argmax(1).float()
                pred_dev = F.softmax(self.forward(self.X_t_dev, dim=1).detach().cpu()).argmax(1).float()

        if self.current_epoch == 0:
            self.logger.experiment.add_image("predictions/label/train", torchvision.utils.make_grid(self.y_t_train), self.current_epoch)
            self.logger.experiment.add_image("predictions/image/train", torchvision.utils.make_grid(self.X_t_train), self.current_epoch)
            self.logger.experiment.add_image("predictions/label/validation", torchvision.utils.make_grid(self.y_t_dev), self.current_epoch)
            self.logger.experiment.add_image("predictions/image/validation", torchvision.utils.make_grid(self.X_t_dev), self.current_epoch)

        self.logger.experiment.add_image("predictions/prediction/train", torchvision.utils.make_grid(pred_train), self.current_epoch)
        self.logger.experiment.add_image("predictions/prediction/validation", torchvision.utils.make_grid(pred_dev), self.current_epoch)
