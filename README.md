# Transfer Learning

## References

- [A deep learning approach for complex microstructure inference](https://rdcu.be/dAD2j)
- [Microstructure segmentation with deep learning encoders pre-trained on a large microscopy dataset](https://rdcu.be/dAD2f)
- [MicroNet](https://github.com/nasa/pretrained-microscopy-models)
- [lightning](https://lightning.ai/docs/pytorch/stable/)

## Overview

This project is about applying transfer learning to the task of classifying and segmenting microstructures in microscopy images of steel. Unfortunately, the datasets are not publicaly available, so the code is not directly executable. However, the code can be adapted to other datasets.
Currently it supports training, validation and testing for [VGG](https://arxiv.org/abs/1409.1556) and [ResNet](https://arxiv.org/abs/1512.03385) encoders for classification.
In the case of segmentation, different backbones and more encoders are available out of the box.
See [segmentation models pytorch](https://github.com/qubvel/segmentation_models.pytorch) for a list of available encoders and backbones.

## Usage

### Setup

Install required packages with

```sh
pip install -r requirements.txt
```

### Overview

The entire project is controlled from the command line. The entry point, which calls [`transfer_learning/train.py`](transfer_learning/train.py) is

```sh
python -m transfer_learning.train
```

To get an overview of all commands available, run

```sh
python -m transfer_learning.train --help
```

### Training

The access point for training is

```sh
python -m transfer_learning.train train
```

To get an overview of available options, run

```sh
python -m transfer_learning.train fit --help
```

These options need not be set from the command line, but can be specified in yaml configuration files.
A couple examples, i.e. those we used for the project, can be found in [configs](./configs).

A training run can be done by specifying configuration files `$CONFIG_1, ..., $CONFIG_N` from [configs](./configs):

```sh
python -m transfer_learning.train fit --config $CONFIG_1 ... --config $CONFIG_N
```

**Note**: Configuration files can overlap in the options they set.
For example `$CONFIG_1` could set `batch_size=32` and `$CONFIG_2` could set
`batch_size=64`. In this case we would get `batch_size=64`. In general, the files overwrite
each other from left to right, such that the option is chosen, which is
specified in the configuration file furthest to the right.

#### Classification Example

As an example, to train a network for the classification task using [VGG-16](https://arxiv.org/abs/1409.1556) with [BatchNorm](https://arxiv.org/abs/1502.03167) pretrained
on [ImageNet](https://www.image-net.org/) and [MicroNet](https://github.com/nasa/pretrained-microscopy-models), optimized with [AdamW](https://arxiv.org/abs/1711.05101) and using data augmentation, the script needs to be
called as follows:

```sh
python -m transfer_learning.train fit \
    --config configs/base.yaml \
    --config configs/task/classification_1.yaml \
    --config configs/models/classification/vgg16_bn.yaml \
    --config configs/optimization/adamw_basic.yaml \
    --config configs/pretraining/image-micronet.yaml \
    --config configs/augmentation/microscope.yaml
```

**Note**: The order matters here. We set different learning rates in
the pretraining configuration file. So it must appear after the optimizer file.
Otherwise the optimizer configuration file overwrites the different learning rates with
a default value.

#### Segmentation Example

A training run for segmentation could look as follows:

```sh
python -m transfer_learning.train fit \
    --config configs/base.yaml \
    --config configs/task/segmentation_1.yaml \
    --config configs/models/classification/vanilla-vgg16_bn.yaml \
    --config configs/optimization/adamw_basic.yaml \
    --config configs/pretraining/image-micronet.yaml \
    --config configs/augmentation/microscope.yaml
```

#### Setting Parameters from the CLI

It is also possible to set/overwrite individual hyperparameters from the commandline. For example
to specify `batch_size` manually in the above example, regardless of what value it had in any of the
configuration files, the following would work:

```sh
python -m transfer_learning.train fit \
    --config configs/base.yaml \
    --config configs/task/segmentation_1.yaml \
    --config configs/models/classification/vanilla-vgg16_bn.yaml \
    --config configs/optimization/adamw_basic.yaml \
    --config configs/pretraining/image-micronet.yaml \
    --config configs/augmentation/microscope.yaml \
    --data.init_args.batch_size 128
```

The structure for these options follows the structure in the yaml files. That is

```yaml
data:
    init_args:
        batch_size: 128
```

translates to

```sh
--data.init_args.batch_size 128
```

It is also possible to create new configuration files. The existing ones can serve as orientation.

### Validation/Testing

The script saves a checkpoint for the best model according to validation accuracy/iou.
These can be found in the `lightning_logs` folder after the training run. It also saves
all the hyperparameters that were used to train the model to a configuration file
in the same folder. Evaluating such a model on the validation- or test set can then
be done by running

```sh
python -m transfer_learning.train validate --config $CONFIG --ckpt_path $CKPT_PATH
```

or 

```sh
python -m transfer_learning.train test --config $CONFIG --ckpt_path $CKPT_PATH
```

where `$CONFIG` is the path to the configuration file and `$CKPT_PATH` is the path to the checkpoint file.

**Note**: The configuration file MUST be specified before the checkpoint. Otherwise
the model will be initialized randomly.

### Prediction

This was not part of the project, so it is not implemented. There is, however, a way to implement

```sh
python -m transfer_learning.train predict
```

The following pages

- https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#prediction-loop
- https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.BasePredictionWriter.html#basepredictionwriter
- https://github.com/Lightning-AI/lightning/discussions/10509

might be useful.
Other than that, stored checkpoints in the lightning_logs folder still contain the usual state_dicts.
Those can be extracted and used to initialize a pytorch model from scratch entirely without lightning.
See https://lightning.ai/docs/pytorch/stable/deploy/production_intermediate.html for a way to do this.

### Results

During training, validation and testing progress is logged to tensorboard. To view it, run

```sh
tensorboard --logdir lightning_logs
```

Additionally, as mentioned before, a configuration file for restoring all hyperparameters as well as a checkpoint
containing all information for restoring the module's state are logged to the corresponding directory in `lightning_logs`.
All hyperparameters are also logged to an additional `hparams.yaml` file in the same folder.

## Further Reading on Configuration Files

Configuration files are in yaml format, and there are plenty of examples in [configs](./configs).
The documentation for configuration files in general can be found here:
- https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced.html
