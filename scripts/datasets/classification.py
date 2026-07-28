import os
from collections import Counter

import click
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


def split_dataset(n, seed, input_dir, output_dir, metadata_path):
    metadata = pd.read_csv(metadata_path)
    files = os.listdir(input_dir)
    metadata = metadata[metadata["image_url"].isin(files)]

    sampled = stratified_sample(metadata, n, seed)
    train, dev, test = stratified_split(sampled, seed)

    for split, split_df in [("train", train), ("dev", dev), ("test", test)]:
        for _, row in split_df.iterrows():
            class_dir = os.path.join(output_dir, split, row["type"])
            os.makedirs(class_dir, exist_ok=True)
            im = Image.open(os.path.join(input_dir, row["image_url"]))
            resized = im.resize(SIZE, Image.Resampling.LANCZOS)
            resized.save(os.path.join(class_dir, row["image_url"]))


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
@click.option("--input-dir", type=str, required=True)
@click.option("--recurse", is_flag=True, help="Recurse into subdirectories")
def sizes(input_dir, recurse):
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
        for (_, width, height), count in sorted(
            counts.items(), key=lambda kv: -kv[1]
        ):
            click.echo(f"{width:>6} {height:>6} {count:>6}")


if __name__ == "__main__":
    cli()
