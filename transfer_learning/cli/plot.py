#!/usr/bin/env python

import os
from datetime import datetime

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from mldb.store import RunStore

plt.rcParams["font.family"] = "DejaVu Serif"


@click.group()
@click.option(
    "--output-dir",
    "output_dir",
    default=lambda: os.environ.get("PLOTS_ROOT"),
    help="Directory where plots are saved. Defaults to the PLOTS_ROOT environment variable.",
)
@click.pass_context
def cli(ctx, output_dir):
    if output_dir is None:
        raise click.ClickException(
            "No output directory provided. Pass --output-dir or set the PLOTS_ROOT environment variable."
        )
    os.makedirs(output_dir, exist_ok=True)
    ctx.ensure_object(dict)
    ctx.obj["output_dir"] = output_dir


@cli.command()
@click.pass_context
def bar(ctx):
    """Validation accuracy by model and pretraining."""
    store = RunStore.from_env()
    db = store.get_db()
    db.attach(names=["val_metrics"], tags=["classification_1"])
    with db.connect() as con:
        df = con.sql(
            """
            select "accuracy/validation" as accuracy, "model.init_args.encoder" as model, "model.init_args.pretrained_weights" as pretraining
            from val_metrics
            """
        ).df()

    model_names = {
        "efficientnet-b4": "EfficientNet-B4",
        "inceptionv4": "Inception-v4",
        "resnet50": "ResNet-50",
        "se_resnext101_32x4d": "SE-ResNeXt-101",
        "senet154": "SENet-154",
        "vgg16_bn": "VGG-16 (BN)",
    }
    df["model"] = df["model"].map(model_names).fillna(df["model"])

    pretrain_names = {
        "none": "No Pretraining",
        "imagenet": "ImageNet",
        "micronet": "MicroNet",
        "image-micronet": "ImageNet + MicroNet",
    }
    df["pretraining"] = df["pretraining"].map(pretrain_names).fillna(df["pretraining"])

    order = (
        df.groupby("model")["accuracy"].max().sort_values(ascending=False).index.tolist()
    )

    sns.set_style("whitegrid")
    palette = sns.color_palette("colorblind")
    color_map = {
        "No Pretraining": palette[0],
        "ImageNet": palette[2],
        "MicroNet": palette[1],
        "ImageNet + MicroNet": palette[3],
    }
    hue_order = ["No Pretraining", "ImageNet", "MicroNet", "ImageNet + MicroNet"]
    g = sns.barplot(
        df,
        x="model",
        y="accuracy",
        hue="pretraining",
        palette=color_map,
        hue_order=hue_order,
        order=order,
    )
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1), frameon=True)
    g.tick_params(labelsize=12)
    g.set_title("Validation Accuracy by Model and Pretraining", fontsize=16)
    g.set_xlabel("Model", fontsize=14)
    g.set_ylabel("Validation Accuracy", fontsize=14)
    g.legend_.set_title("Pretraining")
    plt.xticks(rotation=45, ha="right")

    ymin = df["accuracy"].min() - 0.05 * (df["accuracy"].max() - df["accuracy"].min())
    g.set_ylim(bottom=ymin)

    out_path = os.path.join(
        ctx.obj["output_dir"],
        f"{datetime.now().isoformat()}_classification_results.png",
    )
    plt.savefig(out_path, bbox_inches="tight")


if __name__ == "__main__":
    cli()