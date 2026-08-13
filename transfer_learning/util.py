from os import cpu_count
from typing import Callable, List, Literal, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.hub import load_state_dict_from_url


def get_num_cpu_workers() -> int:
    result = cpu_count()
    if result is None:
        result = 1
    return result


def freeze_encoder_layers(model: nn.Module, ignore_last: int = 0) -> None:
    """Freezes the encoder layers of a model.

    Args:
        model: The model whose encoder layers to freeze.
        ignore_last: The number of layers to ignore from the end of the model.
    """
    parameters = list(model.parameters())
    for i, param in enumerate(parameters):
        if i < len(parameters) - ignore_last:
            param.requires_grad = False


def check_state_dict_sanity(state_dict: dict) -> None:
    """Some weights in a downloaded state_dict might be NaNs. This function checks for that.

    Args:
        state_dict: The state_dict to check.
    """
    for layer, weights in state_dict.items():
        if weights.isnan().any().item():
            raise ValueError(f"weights for layer {layer} contain NaNs")


def get_pretrained_microscopynet_url(
    encoder, encoder_weights, version=1.1, self_supervision=""
):
    """
    Get the url to download the specified pretrained encoder.

    Copied from https://github.com/nasa/pretrained-microscopy-models/blob/9b7c4abc1321e81eca7a68d548e5371676fa74fa/pretrained_microscopy_models/util.py#L27 to avoid version conflicts with smp.

    Args:
        encoder (str): pretrained encoder model name (e.g. resnet50)
        encoder_weights (str): pretraining dataset, either 'micronet' or
            'imagenet-micronet' with the latter indicating the encoder
            was first pretrained on imagenet and then finetuned on microscopynet
        version (float): model version to use, defaults to latest.
            Current options are 1.0 or 1.1.
        self_supervision (str): self-supervision method used. If self-supervision
            was not used set to '' (which is default).

    Returns:
        str: url to download the pretrained model
    """

    # there is an error with the name for resnext101_32x8d so catch and return
    # (currently there is only version 1.0 for this model so don't need to check version.)
    if encoder == "resnext101_32x8d":
        return "https://nasa-public-data.s3.amazonaws.com/microscopy_segmentation_models/resnext101_pretrained_microscopynet_v1.0.pth.tar"

    # only resnet50/micronet has version 1.1 so I'm not going to overcomplicate this right now.
    if encoder != "resnet50" or encoder_weights != "micronet":
        version = 1.0

    # setup self-supervision
    if self_supervision != "":
        version = 1.0
        self_supervision = "_" + self_supervision

    # correct for name change for URL
    if encoder_weights == "micronet":
        encoder_weights = "microscopynet"
    elif encoder_weights == "image-micronet":
        encoder_weights = "imagenet-microscopynet"
    else:
        raise ValueError("encoder_weights must be 'micronet' or 'image-micronet'")

    # get url
    url_base = (
        "https://nasa-public-data.s3.amazonaws.com/microscopy_segmentation_models/"
    )
    url_end = "_v%s.pth.tar" % str(version)
    return (
        url_base + f"{encoder}{self_supervision}_pretrained_{encoder_weights}" + url_end
    )


def load_micronet_weights(
    encoder: str, pretrained_weights: Literal["micronet", "image-micronet"]
) -> dict:
    """Loads micronet weigths for the given encoder and pretrained_weights specification.

    Args:
        encoder: The encoder for which to load the weights.
        pretrained_weights: The pretrained weights to load.
    """
    if pretrained_weights not in ["micronet", "image-micronet"]:
        raise ValueError(
            "pretrained_weights must be one of ['micronet', 'image-micronet']"
        )
    url = get_pretrained_microscopynet_url(encoder, pretrained_weights)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = load_state_dict_from_url(url, map_location=device)
    check_state_dict_sanity(state_dict)
    return state_dict


################################
#                              #
#   tiled full-image inference #
#                              #
################################

# Adapted from pretrained_microscopy_models.segmentation_training.segmentation_models_inference.
# Instead of stitching activated/thresholded predictions together, we stitch the raw logits so
# that the result can be plugged directly into the same loss/metric calls used for cropped tiles.


def _extract_tiles(
    image: torch.Tensor, tile_size: Tuple[int, int], stride: Tuple[int, int]
) -> torch.Tensor:
    """Splits a padded image into overlapping tiles.

    Args:
        image: image tensor of shape (C, H, W), already padded so that H and W are exact multiples of tile_size.
        tile_size: the (height, width) of each tile.
        stride: the (height, width) step between the start of consecutive tiles.

    Returns:
        A tensor of shape (n_tiles_h, n_tiles_w, C, tile_height, tile_width).
    """
    tile_height, tile_width = tile_size
    stride_height, stride_width = stride

    tiles = image.unfold(1, tile_height, stride_height).unfold(
        2, tile_width, stride_width
    )
    return tiles.permute(1, 2, 0, 3, 4).contiguous()


def tile_and_predict(
    predict_fn: Callable[[torch.Tensor], torch.Tensor],
    image: torch.Tensor,
    tile_size: List[int] | Tuple[int, int],
    num_classes: int,
    batch_size: int = 8,
) -> torch.Tensor:
    """Runs inference on a full-sized image by tiling it, predicting each tile, and stitching the
    per-tile logits back together into a single full-sized prediction.

    Tiles overlap by half of `tile_size` in each dimension; only the center of each tile's
    prediction is kept when stitching, which avoids border artifacts from the model seeing
    incomplete context at the edge of a tile.

    Args:
        predict_fn: a function that takes a batch of tiles (N, C, tile_height, tile_width) and
            returns logits of shape (N, num_classes, tile_height, tile_width).
        image: the full-sized, already-preprocessed image tensor of shape (C, H, W).
        tile_size: the (height, width) of the tiles the model was trained on.
        num_classes: the number of output channels `predict_fn` produces.
        batch_size: how many tiles to feed to `predict_fn` at once.

    Returns:
        The stitched logits of shape (num_classes, H, W), matching the input image's size.
    """
    tile_height, tile_width = tile_size
    stride_height, stride_width = tile_height // 2, tile_width // 2
    _, height, width = image.shape

    # reflect-pad by a quarter tile on each side so that every pixel of the original image ends
    # up in the center region of at least one tile
    image = F.pad(
        image.unsqueeze(0),
        (
            stride_width // 2,
            stride_width // 2,
            stride_height // 2,
            stride_height // 2,
        ),
        mode="reflect",
    ).squeeze(0)

    # pad further so height and width are exact multiples of the tile size, otherwise the last
    # row/column of tiles would be lost
    _, padded_height, padded_width = image.shape
    pad_height = (tile_height - padded_height % tile_height) % tile_height
    pad_width = (tile_width - padded_width % tile_width) % tile_width
    image = F.pad(image.unsqueeze(0), (0, pad_width, 0, pad_height)).squeeze(0)

    tiles = _extract_tiles(
        image, (tile_height, tile_width), (stride_height, stride_width)
    )
    n_tiles_h, n_tiles_w = tiles.shape[:2]
    tiles = tiles.reshape(-1, *tiles.shape[2:])

    outputs = []
    with torch.no_grad():
        for i in range(0, tiles.shape[0], batch_size):
            logits = predict_fn(tiles[i : i + batch_size])
            outputs.append(logits.detach())
    outputs = torch.cat(outputs, dim=0)
    outputs = outputs.reshape(
        n_tiles_h, n_tiles_w, num_classes, tile_height, tile_width
    )

    # keep only the center of each tile's prediction so that the stitched tiles tile the image
    # without overlap
    outputs = outputs[
        :,
        :,
        :,
        stride_height // 2 : stride_height // 2 + stride_height,
        stride_width // 2 : stride_width // 2 + stride_width,
    ]

    # stitch the grid of tiles back into a single image
    outputs = outputs.permute(2, 0, 3, 1, 4).contiguous()
    outputs = outputs.reshape(
        num_classes, n_tiles_h * stride_height, n_tiles_w * stride_width
    )

    # remove the padding added above, cropping back to the original image size
    return outputs[:, :height, :width]
