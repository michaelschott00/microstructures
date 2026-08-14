#!/usr/bin/env python

import colorsys
import os
from datetime import datetime

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from mldb.store import RunStore

plt.rcParams["font.family"] = "DejaVu Serif"

MODEL_NAMES = {
    "efficientnet-b4": "EfficientNet-B4",
    "inceptionv4": "Inception-v4",
    "resnet50": "ResNet-50",
    "se_resnext101_32x4d": "SE-ResNeXt-101",
    "senet154": "SENet-154",
    "vgg16_bn": "VGG-16 (BN)",
}

PRETRAIN_NAMES = {
    "none": "No Pretraining",
    "imagenet": "ImageNet",
    "micronet": "MicroNet",
    "image-micronet": "ImageNet + MicroNet",
}

PALETTE = sns.color_palette("colorblind")

STYLE = "whitegrid"

COLOR_MAP = {
    "No Pretraining": PALETTE[0],
    "ImageNet": PALETTE[2],
    "MicroNet": PALETTE[1],
    "ImageNet + MicroNet": PALETTE[3],
}

HUE_ORDER = ["No Pretraining", "ImageNet", "MicroNet", "ImageNet + MicroNet"]

HATCH_MAP = {
    "No Pretraining": "o",
    "ImageNet": "/",
    "MicroNet": "\\",
    "ImageNet + MicroNet": "x",
}


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
@click.option(
    "--metric",
    default="accuracy/validation",
    help="Metric to plot. Defaults to accuracy/validation.",
)
@click.option(
    "--filter",
    default="data/classification_1",
    help="Value for data.init_args.data_dir filter. Defaults to data/classification_1.",
)
@click.pass_context
def bar(ctx, metric, filter):
    """Validation accuracy by model and pretraining."""
    store = RunStore.from_env()
    db = store.get_db()
    db.attach(names=["val_metrics"], hparams={"data.init_args.data_dir": [filter]})
    with db.connect() as con:
        df = con.sql(
            f"""
            select "{metric}" as accuracy, "model.init_args.encoder" as model, "model.init_args.pretrained_weights" as pretraining
            from val_metrics
            """
        ).df()

    df["model"] = df["model"].map(MODEL_NAMES).fillna(df["model"])

    df["pretraining"] = df["pretraining"].map(PRETRAIN_NAMES).fillna(df["pretraining"])

    order = (
        df.groupby("model")["accuracy"].max().sort_values(ascending=False).index.tolist()
    )

    sns.set_style(STYLE)
    g = sns.barplot(
        df,
        x="model",
        y="accuracy",
        hue="pretraining",
        palette=COLOR_MAP,
        hue_order=HUE_ORDER,
        order=order,
    )
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1), frameon=True)
    def desaturate(color, prop):
        h, l, s = colorsys.rgb_to_hls(*color[:3])
        return colorsys.hls_to_rgb(h, l, s * prop)

    color_to_hatch = {
        tuple(desaturate(COLOR_MAP[name], 0.75)): HATCH_MAP[name]
        for name in HUE_ORDER
    }
    for bar in g.patches:
        bar.set_hatch(color_to_hatch[tuple(bar.get_facecolor()[:3])])
    for handle in g.legend_.legend_handles:
        handle.set_hatch(color_to_hatch[tuple(handle.get_facecolor()[:3])])
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