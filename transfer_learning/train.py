from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.loggers import TensorBoardLogger
from mldb.store import RunStore

from transfer_learning.data import ClassificationDataModule, SegmentationDataModule
from transfer_learning.modules import ClassificationModule, SegmentationModule


class MicrostructuresCLI(LightningCLI):
    def before_instantiate_classes(self) -> None:
        """Points the trainer's TensorBoardLogger at a fresh mldb run directory."""
        self.store = RunStore.from_env()
        self.run_id = self.store.create_run()
        uri = self.store.open_directory(self.run_id)

        config = self.config[self.subcommand] if self.subcommand else self.config
        config.trainer.logger = TensorBoardLogger(save_dir=uri, name="", version="")

    def after_instantiate_classes(self) -> None:
        """Exposes the run's store and run_id on the trainer so modules can reach them via `self.trainer.store`/`self.trainer.run_id`."""
        self.trainer.store = self.store
        self.trainer.run_id = self.run_id


def cli_main():
    cli = MicrostructuresCLI(
        auto_configure_optimizers=False, parser_kwargs={"parser_mode": "omegaconf"}
    )


if __name__ == "__main__":
    cli_main()
