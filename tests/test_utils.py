import pytest
import torch
import torch.optim as optim
import numpy as np
from PIL import Image
from src.model import HookedResNet18
from src.utils import (
    generate_grad_cam,
    overlay_heatmap,
    process_canvas_mask,
    train_step,
)


def test_sanity_utils_execution():
    dummy_activations = torch.randn(1, 256, 14, 14)
    dummy_gradients = torch.randn(1, 256, 14, 14)

    cam = generate_grad_cam(dummy_activations, dummy_gradients)
    assert isinstance(cam, np.ndarray)
    assert cam.shape == (14, 14)

    dummy_image = Image.new("RGB", (224, 224), color="white")
    overlayed = overlay_heatmap(dummy_image, cam)
    assert isinstance(overlayed, Image.Image)
    assert overlayed.size == (224, 224)

    dummy_canvas = np.zeros((224, 224, 4), dtype=np.uint8)
    mask = process_canvas_mask(dummy_canvas, target_size=(14, 14))
    assert isinstance(mask, torch.Tensor)
    assert mask.shape == (1, 1, 14, 14)


def test_grad_cam_logic():
    activations = torch.zeros(1, 2, 14, 14)
    activations[0, 0, 0, 0] = 10.0
    activations[0, 1, 13, 13] = 5.0

    gradients = torch.zeros(1, 2, 14, 14)
    gradients[0, 0, 0, 0] = 1.0
    gradients[0, 1, 13, 13] = -1.0

    cam = generate_grad_cam(activations, gradients)

    assert cam[13, 13] == 0.0
    assert cam[0, 0] == pytest.approx(1.0, abs=1e-5)
    assert np.min(cam) >= 0.0
    assert np.max(cam) <= 1.0


def test_process_canvas_mask_spatial_mapping():
    canvas = np.zeros((224, 224, 4), dtype=np.uint8)
    canvas[0:112, :, 3] = 255

    mask_tensor = process_canvas_mask(canvas, target_size=(14, 14))

    assert torch.all(mask_tensor[0, 0, 0:7, :] == 1.0)
    assert torch.all(mask_tensor[0, 0, 7:14, :] == 0.0)


def test_sanity_train_step_execution():
    model = HookedResNet18()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    image_tensor = torch.randn(1, 3, 224, 224)
    label = torch.tensor([0])
    user_mask = torch.ones(1, 1, 14, 14)

    # layer3 is the active layer, so we check it for updates
    pre_update_weight = model.model.layer3[0].conv1.weight.clone()

    loss_total, loss_ce, loss_attention = train_step(
        model, optimizer, image_tensor, label, user_mask, lambda_weight=1.0
    )

    assert isinstance(loss_total, float)
    assert isinstance(loss_ce, float)
    assert isinstance(loss_attention, float)

    post_update_weight = model.model.layer3[0].conv1.weight
    assert not torch.equal(pre_update_weight, post_update_weight)


def test_attention_penalty_logic():
    model = HookedResNet18()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    image_tensor = torch.randn(1, 3, 224, 224)
    label = torch.tensor([1])
    user_mask = torch.ones(1, 1, 14, 14)

    _, _, loss_attention = train_step(
        model, optimizer, image_tensor, label, user_mask, lambda_weight=1.0
    )
    assert loss_attention == 0.0


def test_gradient_flow_isolation():
    model = HookedResNet18()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    image_tensor = torch.randn(1, 3, 224, 224)
    label = torch.tensor([0])
    user_mask = torch.zeros(1, 1, 14, 14)

    # Clone weights to verify isolation
    pre_conv1_weight = model.model.conv1.weight.clone()
    pre_layer3_weight = model.model.layer3[0].conv1.weight.clone()
    pre_layer4_weight = model.model.layer4[0].conv1.weight.clone()
    pre_fc_weight = model.model.fc.weight.clone()

    train_step(model, optimizer, image_tensor, label, user_mask)

    # conv1, layer4, and fc should be frozen
    assert torch.equal(pre_conv1_weight, model.model.conv1.weight)
    assert torch.equal(pre_layer4_weight, model.model.layer4[0].conv1.weight)
    assert torch.equal(pre_fc_weight, model.model.fc.weight)

    # layer3 is active and should change
    assert not torch.equal(pre_layer3_weight, model.model.layer3[0].conv1.weight)
