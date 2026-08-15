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
CONTEXT = "poster"

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

METRIC_MAP = {
    "accuracy/validation": "Accuracy",
    "iou/validation": "IoU",
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
    "--tag",
    default="classification_1",
    help="Tag to filter for. Defaults to classification_1.",
)
@click.option(
    "--title",
    default=None,
    help="Title of the plot. Can contain {metric} placeholder that gets replaced with the metric name.",
)
@click.option(
    "--legend/--no-legend",
    default=False,
    help="Save legend as separate file (otherwise main plot has no legend).",
)
@click.pass_context
def bar(ctx, metric, tag, title, legend):
    """Validation metric by model and pretraining."""
    store = RunStore.from_env()
    db = store.get_db()
    db.attach(names=["val_metrics"], tags=[tag])
    with db.connect() as con:
        df = con.sql(
            f"""
            select "{metric}" as metric, "model.init_args.encoder" as model, "model.init_args.pretrained_weights" as pretraining
            from val_metrics
        """
        ).df()

    df["model"] = df["model"].map(MODEL_NAMES).fillna(df["model"])

    df["pretraining"] = df["pretraining"].map(PRETRAIN_NAMES).fillna(df["pretraining"])

    order = (
        df.groupby("model")["metric"].max().sort_values(ascending=False).index.tolist()
    )

    sns.set_style(STYLE)
    sns.set_context(CONTEXT) #, font_scale=1.2)
    plt.figure(figsize=(12, 6))
    g = sns.barplot(
        df,
        x="model",
        y="metric",
        hue="pretraining",
        palette=COLOR_MAP,
        hue_order=HUE_ORDER,
        order=order,
    )
    def desaturate(color, prop):
        h, l, s = colorsys.rgb_to_hls(*color[:3])
        return colorsys.hls_to_rgb(h, l, s * prop)

    color_to_hatch = {
        tuple(desaturate(COLOR_MAP[name], 0.75)): HATCH_MAP[name]
        for name in HUE_ORDER
    }
    for bar in g.patches:
        bar.set_hatch(color_to_hatch[tuple(bar.get_facecolor()[:3])])

    # Capture legend info before removing
    legend_handles = g.legend_.legend_handles if g.legend_ else []
    legend_labels = [h.get_label() for h in legend_handles] if legend_handles else []

    # Apply hatches to legend handles to match bars
    for handle in legend_handles:
        label = handle.get_label()
        if label in HATCH_MAP:
            handle.set_hatch(HATCH_MAP[label])

    metric_label = METRIC_MAP.get(metric, metric)
    if title is not None:
        final_title = title.replace("{metric}", metric_label)
    else:
        final_title = f"{metric_label} by Model and Pretraining"
    g.set_title(final_title)
    g.set_xlabel("Model")
    g.set_ylabel(metric_label)
    # Dynamic y-axis cap: round up max to nearest 0.1
    max_val = df["metric"].max()
    cap = ((int(max_val * 10) + 1) / 10)
    g.set_ylim(0, cap)
    g.set_yticks([i * 0.1 for i in range(int(cap * 10) + 1)])
    # Always remove legend from main plot
    g.legend_.remove()
    plt.xticks(rotation=45, ha="right")

    out_path = os.path.join(
        ctx.obj["output_dir"],
        f"{datetime.now().isoformat()}_bar.png",
    )
    # Save legend separately if --legend flag is set
    if legend:
        legend_path = out_path.replace('_bar.png', '_legend.png')
        fig = plt.figure(figsize=(3, 1.5))
        ax = fig.add_subplot(111)
        ax.legend(handles=legend_handles, labels=legend_labels, title="Pretraining", loc="center")
        ax.axis('off')
        fig.savefig(legend_path, bbox_inches='tight')
        plt.close(fig)
    plt.savefig(out_path, bbox_inches="tight")


if __name__ == "__main__":
    cli()