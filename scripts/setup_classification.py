import argparse
import os

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n", type=int, required=True, help="number of samples to draw"
    )
    parser.add_argument("--seed", type=int, required=True, help="random seed")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--metadata_path", type=str, required=True)
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata_path)
    files = os.listdir(args.input_dir)
    metadata = metadata[metadata["image_url"].isin(files)]

    sampled = stratified_sample(metadata, args.n, args.seed)
    train, dev, test = stratified_split(sampled, args.seed)

    for split, split_df in [("train", train), ("dev", dev), ("test", test)]:
        for _, row in split_df.iterrows():
            class_dir = os.path.join(args.output_dir, split, row["type"])
            os.makedirs(class_dir, exist_ok=True)
            im = Image.open(os.path.join(args.input_dir, row["image_url"]))
            resized = im.resize(SIZE, Image.Resampling.LANCZOS)
            resized.save(os.path.join(class_dir, row["image_url"]))


if __name__ == "__main__":
    main()
