"""
This is 100% boilerplate and need not be modified unless new lightning modules or data modules are added.
Essentially, we import all modules we would like to be able to specify in the config files.
The actual instantiation, as well as everything else, is handled by the LightningCLI class
and done according to the configuration files.
"""

from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from transfer_learning.modules import ClassificationModule, SegmentationModule
from transfer_learning.data import ClassificationDataModule, SegmentationDataModule


def cli_main():
    cli = LightningCLI(auto_configure_optimizers=False)


if __name__ == '__main__':
    cli_main()
