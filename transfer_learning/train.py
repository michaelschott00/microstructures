from mldb.integrations.lightning import MLDBLightningCLI

from transfer_learning.data import ClassificationDataModule, SegmentationDataModule
from transfer_learning.modules import ClassificationModule, SegmentationModule


def cli_main():
    cli = MLDBLightningCLI(
        auto_configure_optimizers=False, parser_kwargs={"parser_mode": "omegaconf"}
    )


if __name__ == "__main__":
    cli_main()
