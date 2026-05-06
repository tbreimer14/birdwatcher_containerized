import pytest
import torch
from src.model import HookedResNet18


def test_sanity_initialization_and_freezing():
    model = HookedResNet18()
    assert model.model.fc.out_features == 2

    frozen_params = [
        name for name, param in model.named_parameters() if not param.requires_grad
    ]
    active_params = [
        name for name, param in model.named_parameters() if param.requires_grad
    ]

    assert "model.conv1.weight" in frozen_params
    assert "model.layer1.0.conv1.weight" in frozen_params
    assert "model.layer4.0.conv1.weight" in active_params
    assert "model.fc.weight" in active_params


def test_forward_hook_captures_activations():
    model = HookedResNet18()
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)

    assert model.get_activations() is None
    output = model(dummy_input)
    activations = model.get_activations()

    assert output.shape == (1, 2)
    assert activations is not None
    assert activations.shape == (1, 256, 14, 14)


def test_backward_hook_captures_gradients():
    model = HookedResNet18()
    model.train()
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    loss = output.sum()

    assert model.get_gradients() is None
    loss.backward()
    gradients = model.get_gradients()

    assert gradients is not None
    assert gradients.shape == (1, 256, 14, 14)
