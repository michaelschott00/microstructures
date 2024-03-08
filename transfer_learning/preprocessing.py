"""
Several Data Preprocessing Steps

DO NOT USE THIS SCRIPT WITHOUT UNDERSTANDING THE CODE!
------------------------------------------------------

It is not tested, ad-hoc and not very flexible.
"""

from PIL import Image
import os
import argparse
from sklearn.model_selection import train_test_split
from warnings import warn


warn("This script is not tested, ad-hoc and not very flexible. Use with caution!")


def crop(input_dir, output_dir, crop_size, img_size):
    # create output directories
    os.makedirs(os.path.join(output_dir, "Original"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Masks"), exist_ok=True)

    # files
    original_files = os.listdir(os.path.join(input_dir, "Original"))
    mask_files = os.listdir(os.path.join(input_dir, "Masks"))

    # cropping loop
    for original_file, mask_file in zip(original_files, mask_files):
        # Open the images
        original_img = Image.open(os.path.join(input_dir, "Original", original_file))
        mask_img = Image.open(os.path.join(input_dir, "Masks", mask_file))

        # make sure image dimensions are all the same
        assert original_img.size == mask_img.size == (img_size, img_size)

        # Iterate over the images and crop them into smaller images
        for i in range(0, img_size, crop_size):
            for j in range(0, img_size, crop_size):
                # Define the coordinates of the box to crop
                box = (i, j, i + crop_size, j + crop_size)

                # Crop the image
                original_crop = original_img.crop(box)
                mask_crop = mask_img.crop(box)

                # Save the cropped image with a unique filename
                original_crop.save(os.path.join(output_dir, "Original", f"{original_file}_{i}_{j}.png"))
                mask_crop.save(os.path.join(output_dir, "Masks", f"{mask_file}_{i}_{j}.png"))


def crop_classification(input_dir, output_dir, crop_size, img_size):
    # create output directories
    os.makedirs(os.path.join(output_dir), exist_ok=True)

    # cropping loop
    for class_name in os.listdir(input_dir):
        for img_file in os.listdir(os.path.join(input_dir, class_name)):
            # Open the images
            img = Image.open(os.path.join(input_dir, class_name, img_file))

            # Iterate over the images and crop them into smaller images
            for i in range(0, img_size[0], crop_size):
                for j in range(0, img_size[1], crop_size):
                    # Define the coordinates of the box to crop
                    box = (i, j, i + crop_size, j + crop_size)

                    # Crop the image
                    cropped_img = img.crop(box)

                    # Save the cropped image with a unique filename
                    filename, ext = os.path.splitext(img_file)
                    os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)
                    cropped_img.save(os.path.join(output_dir, class_name, f"{filename}_{i}_{j}{ext}"))


def crop_new_images(input_dir, output_dir, crop_size, img_size):
    # create output directories
    os.makedirs(os.path.join(output_dir, "Original"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Masks"), exist_ok=True)

    # files
    original_files = os.listdir(os.path.join(input_dir, "Original"))
    mask_files = os.listdir(os.path.join(input_dir, "Masks"))

    # cropping loop
    for original_file, mask_file in zip(original_files, mask_files):
        # Open the images
        original_img = Image.open(os.path.join(input_dir, "Original", original_file))
        mask_img = Image.open(os.path.join(input_dir, "Masks", mask_file))

        original_img = original_img.resize((2776, 2080))

        assert original_img.size == mask_img.size == (2776, 2080), f"Original: {original_img.size}, Mask: {mask_img.size}"

        # Iterate over the images and crop them into smaller images
        for i in range(0, img_size, crop_size):
            for j in range(0, img_size, crop_size):
                # Define the coordinates of the box to crop
                box = (i, j, i + crop_size, j + crop_size)

                # Crop the image
                original_crop = original_img.crop(box)
                mask_crop = mask_img.crop(box)

                # Save the cropped image with a unique filename
                original_crop.save(os.path.join(output_dir, "Original", f"{original_file}_{i}_{j}.png"))
                mask_crop.save(os.path.join(output_dir, "Masks", f"{mask_file}_{i}_{j}.png"))


def classification_split(input_dir, output_dir, train_size, dev_size, test_size):
    folders = os.listdir(input_dir)
    for folder in folders:
        files = os.listdir(os.path.join(input_dir, folder))

        # split files into train, dev and test set
        train_files, test_files = train_test_split(files, test_size=test_size)
        train_files, dev_files = train_test_split(train_files, test_size=dev_size)

        # create output directories
        os.makedirs(os.path.join(output_dir, "train", folder), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "dev", folder), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "test", folder), exist_ok=True)

        # move files to output directories
        for file in train_files:
            os.rename(os.path.join(input_dir, folder, file), os.path.join(output_dir, "train", folder, file))
        for file in dev_files:
            os.rename(os.path.join(input_dir, folder, file), os.path.join(output_dir, "dev", folder, file))
        for file in test_files:
            os.rename(os.path.join(input_dir, folder, file), os.path.join(output_dir, "test", folder, file))

        # remove old directories
        os.rmdir(os.path.join(input_dir, folder))


def segmentation_split(input_dir, output_dir, train_size, dev_size, test_size):
    images = sorted(os.listdir(os.path.join(input_dir, "Original")))
    masks = sorted(os.listdir(os.path.join(input_dir, "Masks")))

    files = list(zip(images, masks))

    # split files into train, dev and test set
    train_files, test_files = train_test_split(files, test_size=test_size)
    train_files, dev_files = train_test_split(train_files, test_size=dev_size)

    # create output directories
    for folder in ["train", "dev", "test"]:
        os.makedirs(os.path.join(output_dir, folder, "Original"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, folder, "Masks"), exist_ok=True)

    for image, mask in train_files:
        os.rename(os.path.join(input_dir, "Original", image), os.path.join(output_dir, "train", "Original", image))
        os.rename(os.path.join(input_dir, "Masks", mask), os.path.join(output_dir, "train", "Masks", mask))
    for image, mask in dev_files:
        os.rename(os.path.join(input_dir, "Original", image), os.path.join(output_dir, "dev", "Original", image))
        os.rename(os.path.join(input_dir, "Masks", mask), os.path.join(output_dir, "dev", "Masks", mask))
    for image, mask in test_files:
        os.rename(os.path.join(input_dir, "Original", image), os.path.join(output_dir, "test", "Original", image))
        os.rename(os.path.join(input_dir, "Masks", mask), os.path.join(output_dir, "test", "Masks", mask))

    # remove old directories
    os.rmdir(os.path.join(input_dir, "Original"))
    os.rmdir(os.path.join(input_dir, "Masks"))


def main():
    # parse commandline parameters
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title="subcommands", help="choose one of the following subcommands")

    # cropping
    crop_parser = subparsers.add_parser(name="crop", help="crop images into smaller images")
    crop_parser.add_argument("--crop_size", type=int, default=256, help="size of the cropped images, one side, in pixels")
    crop_parser.add_argument("--img_size", nargs='+', type=int, help="size of the original images, one side, in pixels")
    crop_parser.add_argument("--input_dir", type=str, default=os.path.join("data", "segmentation_1", "raw"))
    crop_parser.add_argument("--output_dir", type=str, default=os.path.join("data", "segmentation_1", "tiled"))
    crop_parser.add_argument("--new", type=bool, default=False)

    # splitting
    split_parser = subparsers.add_parser(name="split", help="split images into train, dev and test sets")
    split_parser.add_argument("--train_size", type=float, default=0.8, help="size of the train set, between 0 and 1")
    split_parser.add_argument("--dev_size", type=float, default=0.1, help="size of the dev set, between 0 and 1")
    split_parser.add_argument("--test_size", type=float, default=0.1, help="size of the test set, between 0 and 1")
    split_parser.add_argument("--seed", type=int, default=42, help="seed for the random number generator")
    split_parser.add_argument("--input_dir", type=str, default=os.path.join("data", "segmentation", "raw"))
    split_parser.add_argument("--output_dir", type=str, default=os.path.join("data", "segmentation", "tiled"))

    args = parser.parse_args()

    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # sanity checks
    assert os.path.exists(args.input_dir), f"input directory {args.input_dir} does not exist"
    assert os.path.isdir(args.input_dir), f"input directory {args.input_dir} is not a directory"
    if "crop_size" in args:
        assert args.crop_size <= args.img_size[0] and args.crop_size <= args.img_size[1], f"crop size {args.crop_size} is larger than image size {args.img_size}"
        assert "tiled" not in os.listdir(args.output_dir), "data is already tiled"
    if "train_size" in args:
        assert args.train_size + args.dev_size + args.test_size == 1, "train, dev and test size must sum up to 1"
        assert not any(x in os.listdir(args.output_dir) for x in ["train", "test", "dev"]), "data is already split"

    # call the appropriate function
    if "crop_size" in args:
        if args.new:
            print('using new shit')
            crop_new_images(args.input_dir, args.output_dir, args.crop_size, args.img_size)
        elif 'segmentation' in args.input_dir:
            crop(args.input_dir, args.output_dir, args.crop_size, args.img_size)
        elif 'classification_2' in args.input_dir:
            crop_classification(args.input_dir, args.output_dir, args.crop_size, args.img_size)
    elif "train_size" in args:
        if "classification" in args.input_dir:
            classification_split(args.input_dir, args.output_dir, args.train_size, args.dev_size, args.test_size)
        elif "segmentation" in args.input_dir:
            segmentation_split(args.input_dir, args.output_dir, args.train_size, args.dev_size, args.test_size)
        else:
            raise ValueError("input directory must contain either 'classification' or 'segmentation'")


if __name__ == "__main__":
    main()
