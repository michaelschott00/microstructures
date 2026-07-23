import pytest
import torch
from torch import nn

from transfer_learning import util


def test_freeze_encoder_layers_freezes_all_by_default():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 2))

    util.freeze_encoder_layers(model, ignore_last=0)

    assert all(not p.requires_grad for p in model.parameters())


def test_freeze_encoder_layers_keeps_last_n_trainable():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 2))
    parameters = list(model.parameters())
    n_ignore = 2  # last linear layer has a weight and a bias -> 2 parameter tensors

    util.freeze_encoder_layers(model, ignore_last=n_ignore)

    frozen = [p.requires_grad for p in parameters[: len(parameters) - n_ignore]]
    trainable = [p.requires_grad for p in parameters[len(parameters) - n_ignore :]]
    assert not any(frozen)
    assert all(trainable)


def test_check_state_dict_sanity_passes_for_clean_weights():
    state_dict = {"layer.weight": torch.zeros(2, 2)}
    util.check_state_dict_sanity(state_dict)  # should not raise


def test_check_state_dict_sanity_raises_for_nan_weights():
    state_dict = {"layer.weight": torch.tensor([float("nan"), 1.0])}
    with pytest.raises(ValueError, match="NaNs"):
        util.check_state_dict_sanity(state_dict)


def test_load_micronet_weights_rejects_invalid_pretrained_weights():
    with pytest.raises(ValueError):
        util.load_micronet_weights("resnet18", "imagenet")
