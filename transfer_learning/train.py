from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch import LightningModule, Trainer
from transfer_learning.data import ClassificationDataModule, SegmentationDataModule
from transfer_learning.modules import ClassificationModule, SegmentationModule


class DontSaveConfigCallback(SaveConfigCallback):
  def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
    print("Your config is save with me. Trust me.")


def cli_main():
    cli = LightningCLI(
        auto_configure_optimizers=False,
        save_config_callback=DontSaveConfigCallback,
        save_config_kwargs={"save_to_log_dir": False},
        parser_kwargs={"parser_mode": "omegaconf"}
    )


if __name__ == "__main__":
    cli_main()
