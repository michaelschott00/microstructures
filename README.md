# Transfer Learning for Microstructure Classification and Segmentation

## Results

### MicroNet pretraining increases performance, particularly in the low data regime

Combining ImageNet- and MicroNet pretraining and utilizing domain-motivated data augmentation together with fine-tuning techniques, we

- improved F1-Score by 0.05 and
- improved IoU by 0.12

over randomly initialized VGG baselines. We further

- increased F1-Score by ~6% in the low data regime (605 samples)

compared to ImageNet pretraining.

### Full data results

| Encoder | Pretraining       | F1-Score | Encoder  | Pretraining       | IoU  |
| ------- | ------------      | -------- | -------  | ------------      | ---  |
| VGG16   | None              | 0.84     | VGG-16   | None              | 0.53 |
| VGG16   | ImageNet          | 0.88     | VGG-16   | ImageNet          | 0.60 |
| VGG16   | MicroNet          | 0.83     | VGG-16   | MicroNet          | 0.50 |
| VGG16   | ImageNet-Micronet | 0.89     | VGG-16   | ImageNet-Micronet | 0.60 |
|         |                   |          | ResNet50 | ImageNet          | 0.62 |
|         |                   |          | ResNet50 | MicroNet          | 0.46 |
|         |                   |          | ResNet50 | ImageNet-MicroNet | 0.65 |

### Low-data results

- [ ] Re-run evaluation (waiting for github pipeline)

## Usage

### Setup

Install required packages with

```sh
pip install -r requirements.txt
```

### Classification Example

The following command trains a network for microstructure classification using [VGG-16](https://arxiv.org/abs/1409.1556) with [BatchNorm](https://arxiv.org/abs/1502.03167), [ImageNet](https://www.image-net.org/) and [MicroNet](https://github.com/nasa/pretrained-microscopy-models) pretraining using the [AdamW](https://arxiv.org/abs/1711.05101) optimizer and data augmentation:

```sh
python -m transfer_learning.train fit \
    --config configs/base.yaml \
    --config configs/task/classification_1.yaml \
    --config configs/models/classification/vgg16_bn.yaml \
    --config configs/optimization/adamw_basic.yaml \
    --config configs/pretraining/image-micronet.yaml \
    --config configs/augmentation/microscope.yaml
```

### Segmentation Example

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

### Setting Parameters from the CLI

Individual hyperparameters can be overwritten from the command line. For example
to specify `batch_size` manually, use

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

`$CONFIG` is the path to the configuration file and `$CKPT_PATH` is the path to the checkpoint file.

**Note**: The configuration file MUST be specified before the checkpoint. Otherwise
the model will be initialized randomly.

### Viewing logs and results

During training, validation and testing progress is logged to tensorboard. To view it, run

```sh
tensorboard --logdir lightning_logs
```

## References

- [A deep learning approach for complex microstructure inference](https://rdcu.be/dAD2j)
- [Microstructure segmentation with deep learning encoders pre-trained on a large microscopy dataset](https://rdcu.be/dAD2f)
- [MicroNet](https://github.com/nasa/pretrained-microscopy-models)
