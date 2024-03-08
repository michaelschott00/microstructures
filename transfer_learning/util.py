from torch.utils import model_zoo
import pretrained_microscopy_models as pmm
from torch import nn
from typing import Literal


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


def load_micronet_weights(encoder: str, pretrained_weights: Literal["micronet", "image-micronet"]) -> dict:
    """Loads micronet weigths for the given encoder and pretrained_weights specification.

    Args:
        encoder: The encoder for which to load the weights.
        pretrained_weights: The pretrained weights to load.
    """
    if pretrained_weights not in ["micronet", "image-micronet"]:
        raise ValueError("pretrained_weights must be one of ['micronet', 'image-micronet']")
    url = pmm.util.get_pretrained_microscopynet_url(encoder, pretrained_weights)
    state_dict = model_zoo.load_url(url)
    check_state_dict_sanity(state_dict)
    return state_dict
