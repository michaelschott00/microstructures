import torch
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader
import os
import cv2
from torch.utils.data import Subset
import numpy as np
import albumentations as A
import segmentation_models_pytorch as smp
from typing import Literal, Callable, Dict, Tuple, List, Type, Any


################################
#                              #
#        classification        #
#                              #
################################

class ClassificationDataset(Dataset):
    """A Dataset that loads the images and returns them together with their class label based on the folder they are stored in.

    Attributes:
        split:
            the split of the dataset to load (train, dev or test)
        root_dir:
            the directory where the dataset is stored
        preprocessing:
            a function that is applied to the raw or augmented image for preprocessing (e.g. ToTensor)
        augmentation:
            a function that is applied to the raw image for augmentation (e.g. RandomCrop)
    """

    def __init__(self,
                 split: Literal["train", "dev", "test"],
                 root_dir: str,
                 preprocessing: Callable[np.ndarray, np.ndarray] = None,
                 augmentation: Callable[np.ndarray, np.ndarray] = None) -> None:
        super().__init__()

        if split not in ['train', 'dev', 'test']:
            raise ValueError("split must be one of 'train', 'dev' or 'test'")

        assert os.path.exists(root_dir), f"root_dir '{root_dir}' does not exist"

        self.root_dir = os.path.join(root_dir, split)  # set root_dir to split to avoid always having to join it explicitly later
        assert os.path.exists(self.root_dir), f"split '{self.root_dir}' does not exist"

        self.LABELS: Dict[str, int] = {subfolder: index for index, subfolder in enumerate(self._get_subfolders(self.root_dir))}  # encode labels by enumerating the folders in the root folder
        self.preprocessing = preprocessing
        self.augmentation = augmentation
        self.file_list = self._get_filenames(self.root_dir, subfolders=self.LABELS.keys())
        assert len(self.file_list) > 0, f"no files found in {self.root_dir}"

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """Returns the image and its label at the given index after applying Albumentations Preprocessing and Augmentation as well as encoding the labels."""
        assert not isinstance(idx, slice), "slicing is not supported"

        filename = self.file_list[idx]

        # load image
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # apply augmentation transform
        if self.augmentation:
            img = self.augmentation(image=img)["image"]

        # apply preprocessing transform
        if self.preprocessing:
            img = self.preprocessing(image=img)["image"]

        # parse label from filepath and convert to int
        label = self.LABELS[os.path.basename(os.path.dirname(filename))]

        return img, label

    def _get_filenames(self, root_dir: str, subfolders: List[str]) -> List[str]:
        """Returns a list of all filenames in the given subfolders."""
        file_list = []
        for label in subfolders:
            for filename in os.listdir(os.path.join(root_dir, label)):
                file_list.append(os.path.join(root_dir, label, filename))
        return file_list

    def _get_subfolders(self, root_dir: str) -> Dict[str, int]:
        """Returns a list of all subfolders in the given root folder. That is, the folders for all classes, which also serve as class names."""
        return [name for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))]


################################
#                              #
#         segmentation         #
#                              #
################################

class SegmentationDataset(Dataset):
    """A Dataset that loads and provides access to input images as well as their corresponding masks.

    Attributes:
        MASK_DIR:
            the directory relative to root_dir/split containing the masks
        IMG_DIR:
            the directory relative to root_dir/split containing the images
        root_dir:
            the directory where the dataset is stored
        file_list:
            a list of all filenames of the masks (the corresponding image path can be restored from them)
        transform:
            an optional transformation that is applied to the raw image as well as the raw mask e.g. ToTensor
        tiled:
            whether to load patches or the full images
    """

    MASK_DIR = "Masks"
    IMG_DIR = "Original"

    def __init__(self,
                 split: Literal["train", "dev", "test"],
                 root_dir: str,
                 num_classes: int,
                 preprocessing: Callable[np.ndarray, np.ndarray] = None,
                 augmentation: Callable[np.ndarray, np.ndarray] = None,
                 tiled: bool = True) -> None:
        super().__init__()

        if split not in ['train', 'dev', 'test']:
            raise ValueError("split must be one of 'train', 'dev' or 'test'")

        assert os.path.exists(root_dir), f"root_dir '{root_dir}' does not exist"

        self.root_dir = os.path.join(root_dir, "tiled" if tiled else "raw", split)
        assert os.path.exists(self.root_dir), f"split '{self.root_dir}' does not exist"

        img_dir = os.path.join(self.root_dir, self.IMG_DIR)
        mask_dir = os.path.join(self.root_dir, self.MASK_DIR)
        assert os.path.exists(img_dir), f"img_dir '{img_dir}' does not exist"
        assert os.path.exists(mask_dir), f"mask_dir '{mask_dir}' does not exist"

        imgs = sorted([os.path.join(img_dir, img_filename) for img_filename in os.listdir(img_dir)])
        masks = sorted([os.path.join(mask_dir, mask_filename) for mask_filename in os.listdir(mask_dir)])
        self.file_list = list(zip(imgs, masks))

        self.num_classes = num_classes
        self.preprocessing = preprocessing
        self.augmentation = augmentation

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.file_list)

    def load_image(self, img_filename: str) -> np.ndarray:
        """Loads an image from the given filename.

        Args:
            img_filename: the filename of the image to load
        """
        img = cv2.imread(img_filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # read images as RGB to apply imagenet preprocessing later
        return img

    def load_mask(self, mask_filename: str) -> np.ndarray:
        """Loads a mask from the given filename, thresholds it to binary if specified, and converts the mask to 8bit integers.

        Args:
            mask_filename: the filename of the mask to load
        """
        mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)[..., np.newaxis]  # having a channel dimension makes other steps easier
        if self.num_classes == 1:
            mask = (mask > 0)  # threshold to binary mask
        return mask.astype(np.uint8)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the image and mask at the given index. Albumentations Preprocessing and Augmentation are applied to image and mask, ensuring
        that the same transformations are applied to both.

        Args:
            idx: the index of the sample to return
        """
        assert not isinstance(idx, slice), "slicing is not supported"

        img_filename, mask_filename = self.file_list[idx]

        # load image and mask
        img = self.load_image(img_filename)
        mask = self.load_mask(mask_filename)

        # apply mask transform
        if self.augmentation:
            sample = self.augmentation(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']

        # apply preprocessing transform
        if self.preprocessing:
            sample = self.preprocessing(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']

        return img, mask


################################
#                              #
#           datamodule         #
#                              #
################################

class DataModule(pl.LightningDataModule):
    """A lightning module taking care of creating the correct datasets and dataloaders with the specified parameters, and
    creating the augmentation and preprocessing pipeline.

    There is no need to ever instantiate this class manually, except for testing, since it is completely handled by lightning.

    Attributes:
        dataset_cls: the class of the dataset to use
        dataset_args: the arguments to pass to the dataset at instantiation
    """

    def __init__(self,
                 dataset_cls: Type[Dataset],
                 dataset_args: Dict[str, Any]):
        super().__init__()
        self.dataset_cls = dataset_cls
        self.dataset_args = dataset_args

    def get_train_augmentations(self) -> A.Compose:
        """Returns the augmentations to apply to the training set samples."""
        train_transform = []

        # this is technically preprocessing, but we want to apply preprocessing after augmentation, because it
        # seems more intuitive to augment the original image and then prepare it for the model by e.g. normalizing
        # however, augmentations need the images to be of fixed size
        train_transform.append(A.Resize(
            height=self.hparams.size[0],
            width=self.hparams.size[1]
        ))

        if self.hparams.crop_ratio:
            # mimic zooming
            min_edge = min(self.hparams.size)
            train_transform.append(A.RandomSizedCrop(
                min_max_height=(int(min_edge * self.hparams.crop_ratio), min_edge),
                height=self.hparams.size[0],
                width=self.hparams.size[1]
            ))

        # flipping
        if self.hparams.random_horizontal_flip:
            train_transform.append(A.HorizontalFlip())
        if self.hparams.random_vertical_flip:
            train_transform.append(A.VerticalFlip())

        # rotation
        if self.hparams.random_rotation:
            train_transform.append(A.RandomRotate90())

        # brightness
        if self.hparams.brightness_limit and self.hparams.contrast_limit:
            train_transform.append(A.RandomBrightnessContrast(
                brightness_limit=self.hparams.brightness_limit,
                contrast_limit=self.hparams.contrast_limit
            ))

        # sharpen/blur
        if self.hparams.blur_limit and self.hparams.sharpen_alpha:
            train_transform.append(A.OneOf([
                A.Blur(blur_limit=self.hparams.blur_limit),
                A.Sharpen(alpha=self.hparams.sharpen_alpha)
            ]))

        return A.Compose(train_transform)

    def get_dev_augmentations(self) -> A.Compose:
        """Returns the augmentations to apply to the validation set samples."""
        return None

    def get_test_augmentations(self) -> A.Compose:
        """Returns the augmentations to apply to the test set samples."""
        return None

    # Preprocessing
    def _to_tensor(self, x: np.ndarray, **kwargs) -> torch.Tensor:
        """Converts the images, which are initially numpy arrays, to pytorch tensors and transposes them to the shape
        pytorch's modules expect. That is, we convert from HWC to CHW."""
        return torch.tensor(x.transpose(2, 0, 1).astype('float32'))

    def get_preprocessing(self) -> A.Compose:
        """Returns the default preprocessing pipeline."""

        preprocessing_steps = []

        # this applies imagenet normalization
        if self.hparams.imagenet_preprocessing:
            preprocessing_steps += [
                A.Lambda(image=smp.encoders.get_preprocessing_fn(
                    self.hparams.encoder,
                    "imagenet"
                )),
            ]

        preprocessing_steps += [
            A.Resize(
                height=self.hparams.size[0],
                width=self.hparams.size[1]
            ),
            A.Lambda(
                image=self._to_tensor,
                mask=self._to_tensor
            )
        ]

        return A.Compose(preprocessing_steps)

    def get_train_preprocessing(self) -> A.Compose:
        """Returns the preprocessing pipeline to apply to the training set samples."""
        return self.get_preprocessing()

    def get_dev_preprocessing(self) -> A.Compose:
        """Returns the preprocessing pipeline to apply to the validation set samples."""
        return self.get_preprocessing()

    def get_test_preprocessing(self) -> A.Compose:
        """Returns the preprocessing pipeline to apply to the test set samples."""
        return self.get_preprocessing()

    # Setup

    def setup(self, stage: str) -> None:
        """Sets up the datasets for the individual stages. This is entirely handled by lightning and should not be called manually."""
        if stage == 'fit' or stage is None:
            self.train_dataset = self.dataset_cls(split='train',
                                                  root_dir=self.hparams.data_dir,
                                                  augmentation=self.get_train_augmentations(),
                                                  preprocessing=self.get_train_preprocessing(),
                                                  **self.dataset_args)
            self.dev_dataset = self.dataset_cls(split='dev',
                                                root_dir=self.hparams.data_dir,
                                                augmentation=self.get_dev_augmentations(),
                                                preprocessing=self.get_dev_preprocessing(),
                                                **self.dataset_args)
            if self.hparams.sample_size:
                self.train_dataset = Subset(self.train_dataset, np.random.choice(len(self.train_dataset), self.hparams.sample_size))

        if stage == 'validate':
            self.dev_dataset = self.dataset_cls(split='dev',
                                                root_dir=self.hparams.data_dir,
                                                augmentation=self.get_dev_augmentations(),
                                                preprocessing=self.get_dev_preprocessing(),
                                                **self.dataset_args)

        if stage == 'test' or stage is None:
            self.test_dataset = self.dataset_cls(split='test',
                                                 root_dir=self.hparams.data_dir,
                                                 augmentation=self.get_test_augmentations(),
                                                 preprocessing=self.get_test_preprocessing(),
                                                 **self.dataset_args)

        if stage == "predict":
            raise NotImplementedError("Prediction is not implemented")

    def train_dataloader(self) -> DataLoader:
        """Returns the dataloader for the training set. Handled by lightning and should not be called manually."""
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Returns the dataloader for the validation set. Handled by lightning and should not be called manually."""
        return DataLoader(self.dev_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers)

    def test_dataloader(self) -> DataLoader:
        """Returns the dataloader for the test set. Handled by lightning and should not be called manually."""
        return DataLoader(self.test_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers)


class ClassificationDataModule(DataModule):
    """Same as DataModule but adapted for the classification task

    Hyperparameters can be specified in config files or on the commandline.
    This class is used by lightning and should not be called manually.

    Args:
        data_dir: path to the data directory
        num_workers: number of workers for the dataloader
        imagenet_preprocessing: whether to apply imagenet preprocessing
        size: size of the images to crop/interpolate to
        encoder: name of the encoder to get the correct preprocessing function
        crop_ratio: ratio of the image to crop to
        random_horizontal_flip: whether to apply random horizontal flipping
        random_vertical_flip: whether to apply random vertical flipping
        random_rotation: whether to apply random rotation
        brightness_limit: brightness limit for random brightness
        contrast_limit: contrast limit for random contrast
        blur_limit: blur limit for random blur
        sharpen_alpha: alpha range for random sharpen
        batch_size: batch size
        sample_size: sample size (NOT a proportion but the absolute size, i.e. if you specify 100, your train set will have 100 samples)
    """

    def __init__(self,  # TODO: update
                 # filesystem stuff
                 data_dir: str,

                 # hardware related
                 num_workers: int = 0,

                 # preprocessing
                 imagenet_preprocessing: bool = True,
                 size: List[int] = None,
                 encoder: str = None,

                 # data augmentation
                 crop_ratio: float = None,
                 random_horizontal_flip: bool = False,
                 random_vertical_flip: bool = False,
                 random_rotation: bool = False,
                 brightness_limit: float = None,
                 contrast_limit: float = None,
                 blur_limit: int = None,
                 sharpen_alpha: List[float] = None,

                 # batching and sampling
                 batch_size: int = 32,
                 sample_size: int = None) -> None:
        super().__init__(dataset_cls=ClassificationDataset, dataset_args={})

        # store hyperparameters to access them through self.hparams, superclass uses this to access all hprams
        self.save_hyperparameters()


class SegmentationDataModule(DataModule):
    """Same as data module, but adapted for the segmentation task.

    Hyperparameters can be specified in config files or on the commandline.
    This class is used by lightning and should not be called manually.

    Args:
        data_dir: path to the data directory
        num_workers: number of workers for the dataloader
        imagenet_preprocessing: whether to apply imagenet preprocessing
        size: size of the images to crop/interpolate to
        encoder: name of the encoder to get the correct preprocessing function
        crop_ratio: ratio of the image to crop to
        random_horizontal_flip: whether to apply random horizontal flipping
        random_vertical_flip: whether to apply random vertical flipping
        random_rotation: whether to apply random rotation
        brightness_limit: brightness limit for random brightness
        contrast_limit: contrast limit for random contrast
        blur_limit: blur limit for random blur
        sharpen_alpha: alpha range for random sharpen
        batch_size: batch size
        sample_size: sample size (NOT a proportion but the absolute size, i.e. if you specify 100, your train set will have 100 samples)
        tiled: whether to use tiled images
    """

    def __init__(self,  # TODO: update
                 # filesystem stuff
                 data_dir: str,

                 num_classes: int = 1,

                 # hardware related
                 num_workers: int = 0,

                 # preprocessing
                 imagenet_preprocessing: bool = True,
                 size: List[int] = None,
                 encoder: str = None,

                 # data augmentation
                 crop_ratio: float = None,
                 random_horizontal_flip: bool = False,
                 random_vertical_flip: bool = False,
                 random_rotation: bool = False,
                 brightness_limit: float = None,
                 contrast_limit: float = None,
                 blur_limit: int = None,
                 sharpen_alpha: List[float] = None,

                 # batching and sampling
                 sample_size: int = None,
                 batch_size: int = 32,

                 # tiling
                 tiled: bool = True):
        super().__init__(
            dataset_cls=SegmentationDataset,
            dataset_args={"tiled": tiled, "num_classes": num_classes}
        )

        # store hyperparameters to access them through self.hparams, superclass uses this to access all hprams
        self.save_hyperparameters()
