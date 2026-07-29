import os
import shutil
from collections import Counter

import click
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

SIZE = (256, 192)


def stratified_sample(df, n, seed):
    frac = n / len(df)
    parts = [
        group.sample(n=round(len(group) * frac), random_state=seed)
        for _, group in df.groupby("type")
    ]
    return pd.concat(parts)


def stratified_split(df, seed):
    train_parts, dev_parts, test_parts = [], [], []
    for _, group in df.groupby("type"):
        group = group.sample(frac=1, random_state=seed)
        n_train = round(len(group) * 0.8)
        n_dev = round(len(group) * 0.1)
        train_parts.append(group.iloc[:n_train])
        dev_parts.append(group.iloc[n_train : n_train + n_dev])
        test_parts.append(group.iloc[n_train + n_dev :])
    return pd.concat(train_parts), pd.concat(dev_parts), pd.concat(test_parts)


def prune_missing_classes(output_dir, splits):
    class_sets = {
        split: set(os.listdir(os.path.join(output_dir, split))) for split in splits
    }
    common_classes = set.intersection(*class_sets.values())

    for split, classes in class_sets.items():
        for class_name in classes - common_classes:
            shutil.rmtree(os.path.join(output_dir, split, class_name))


def split_dataset(n, seed, input_dir, output_dir, metadata_path):
    metadata = pd.read_csv(metadata_path)
    files = os.listdir(input_dir)
    metadata = metadata[metadata["image_url"].isin(files)]

    sampled = stratified_sample(metadata, n, seed)
    train, dev, test = stratified_split(sampled, seed)

    splits = [("train", train), ("dev", dev), ("test", test)]
    for split, split_df in splits:
        for _, row in split_df.iterrows():
            class_dir = os.path.join(output_dir, split, row["type"])
            os.makedirs(class_dir, exist_ok=True)
            im = Image.open(os.path.join(input_dir, row["image_url"]))
            resized = im.resize(SIZE, Image.Resampling.LANCZOS)
            resized.save(os.path.join(class_dir, row["image_url"]))

    prune_missing_classes(output_dir, [split for split, _ in splits])


@click.group()
def cli():
    pass


@cli.command()
@click.option("--n", type=int, required=True, help="Number of samples to draw")
@click.option("--seed", type=int, required=True, help="Random seed")
@click.option("--input-dir", type=str, required=True)
@click.option("--output-dir", type=str, required=True)
@click.option("--metadata-path", type=str, required=True)
def split(n, seed, input_dir, output_dir, metadata_path):
    """Stratified sample and split a dataset into train/dev/test."""
    split_dataset(n, seed, input_dir, output_dir, metadata_path)


@cli.command()
@click.option(
    "--input-dir",
    type=str,
    required=True,
    help="Root dataset directory, structured as root_dir/split/class",
)
def sizes(input_dir):
    """Plot a grid of barplots (one per split) of image sizes by class."""
    counts = Counter()
    for split in sorted(os.listdir(input_dir)):
        split_dir = os.path.join(input_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for class_name in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for filename in os.listdir(class_dir):
                with Image.open(os.path.join(class_dir, filename)) as im:
                    counts[(split, class_name, im.size)] += 1

    by_split = {}
    for (split, class_name, size), count in counts.items():
        by_split.setdefault(split, {}).setdefault(class_name, {})[size] = count

    order = {"train": 0, "dev": 1, "test": 2}
    splits = sorted(by_split, key=lambda s: (order.get(s, len(order)), s))
    fig, axes = plt.subplots(
        len(splits), 1, squeeze=False, figsize=(12, 4 * len(splits))
    )

    for ax, split in zip(axes.flat, splits):
        by_class = by_split[split]
        classes = sorted(by_class)
        sizes_ = sorted(
            {size for class_counts in by_class.values() for size in class_counts}
        )
        labels = [f"{width}x{height}" for width, height in sizes_]

        x = range(len(sizes_))
        width = 0.8 / len(classes)
        for i, class_name in enumerate(classes):
            values = [by_class[class_name].get(size, 0) for size in sizes_]
            offsets = [xi + i * width for xi in x]
            bars = ax.bar(offsets, values, width=width, label=class_name)
            ax.bar_label(bars, fontsize=7)

        ax.set_xticks([xi + width * (len(classes) - 1) / 2 for xi in x])
        ax.set_xticklabels(labels, rotation=45)
        ax.set_title(split)
        ax.set_xlabel("")
        ax.set_ylabel("count")
        ax.legend()

    fig.tight_layout()
    plt.show()


@cli.command()
@click.option("--input-dir", type=str, required=True)
@click.option("--recurse", is_flag=True, help="Recurse into subdirectories")
def sizetable(input_dir, recurse):
    """Print a table of image sizes and how often they occur."""
    counts = Counter()
    if recurse:
        for root, _, filenames in os.walk(input_dir):
            subdir = os.path.relpath(root, input_dir)
            for filename in filenames:
                with Image.open(os.path.join(root, filename)) as im:
                    counts[(subdir, *im.size)] += 1
    else:
        for filename in os.listdir(input_dir):
            with Image.open(os.path.join(input_dir, filename)) as im:
                counts[(".", *im.size)] += 1

    if recurse:
        by_subdir = {}
        for (subdir, width, height), count in counts.items():
            by_subdir.setdefault(subdir, []).append(((width, height), count))

        for subdir in sorted(by_subdir):
            click.echo(subdir)
            click.echo(f"{'width':>6} {'height':>6} {'count':>6}")
            for (width, height), count in sorted(
                by_subdir[subdir], key=lambda kv: -kv[1]
            ):
                click.echo(f"{width:>6} {height:>6} {count:>6}")
            click.echo()
    else:
        click.echo(f"{'width':>6} {'height':>6} {'count':>6}")
        for (_, width, height), count in sorted(counts.items(), key=lambda kv: -kv[1]):
            click.echo(f"{width:>6} {height:>6} {count:>6}")


if __name__ == "__main__":
    cli()
