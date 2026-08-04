from typing import Any

from jsonargparse import Namespace
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI, SaveConfigCallback
from mldb.store import RunStore

from transfer_learning.data import ClassificationDataModule, SegmentationDataModule
from transfer_learning.modules import ClassificationModule, SegmentationModule


def _flatten_hparams(config: Any, parent_key: str = "") -> dict[str, Any]:
    """Flattens a (possibly nested) jsonargparse Namespace/dict into dotted-key hparams."""
    if isinstance(config, Namespace):
        config = vars(config)
    if isinstance(config, dict):
        hparams = {}
        for key, value in config.items():
            dotted_key = f"{parent_key}.{key}" if parent_key else key
            hparams.update(_flatten_hparams(value, dotted_key))
        return hparams
    if isinstance(config, type):
        return {parent_key: config.__name__}
    if type(config).__repr__ is object.__repr__:
        return {parent_key: config.__class__.__name__}
    return {parent_key: config}


class MicrostructuresCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument(
            "--results_dir",
            type=str,
            default=None,
            help="Root directory for mldb run storage. Falls back to RunStore.from_env() if unset.",
        )

    def before_instantiate_classes(self) -> None:
        """Points the trainer's TensorBoardLogger at a fresh mldb run directory."""
        config = self.config[self.subcommand] if self.subcommand else self.config
        results_dir = config.results_dir
        self.store = RunStore(root_dir=results_dir) if results_dir is not None else RunStore.from_env()
        hparams = _flatten_hparams(config)
        self.run_id = self.store.create_run(hparams=hparams)
        uri = self.store.open_directory(self.run_id)

        config.trainer.logger = {
            "class_path": "lightning.pytorch.loggers.TensorBoardLogger",
            "init_args": {"save_dir": uri, "name": "", "version": ""},
        }

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
