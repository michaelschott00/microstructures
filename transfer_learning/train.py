from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.cli import LightningCLI

from transfer_learning.data import ClassificationDataModule, SegmentationDataModule
from transfer_learning.modules import ClassificationModule, SegmentationModule


def cli_main():
    cli = LightningCLI(auto_configure_optimizers=False)


if __name__ == "__main__":
    cli_main()
