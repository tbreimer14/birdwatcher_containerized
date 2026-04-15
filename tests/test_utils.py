# tests/test_utils.py
import pytest
import torch
import torch.optim as optim
import numpy as np
from PIL import Image
from src.model import HookedResNet18
from src.utils import generate_grad_cam, overlay_heatmap, process_canvas_mask, train_step

def test_sanity_utils_execution():
    """
    Sanity Check: All utility functions should process dummy tensors/arrays 
    without crashing and return the strictly expected data types and shapes.
    """
    # Dummy outputs matching a ResNet18 layer4
    dummy_activations = torch.randn(1, 512, 7, 7)
    dummy_gradients = torch.randn(1, 512, 7, 7)

    cam = generate_grad_cam(dummy_activations, dummy_gradients)
    assert isinstance(cam, np.ndarray)
    assert cam.shape == (7, 7)

    dummy_image = Image.new("RGB", (224, 224), color="white")
    overlayed = overlay_heatmap(dummy_image, cam)
    assert isinstance(overlayed, Image.Image)
    assert overlayed.size == (224, 224)

    # Dummy empty Streamlit canvas (RGBA)
    dummy_canvas = np.zeros((224, 224, 4), dtype=np.uint8)
    mask = process_canvas_mask(dummy_canvas, target_size=(7, 7))
    assert isinstance(mask, torch.Tensor)
    assert mask.shape == (1, 1, 7, 7)

def test_grad_cam_logic():
    """
    Non-Trivial Test 1: Verify the mathematical correctness of Grad-CAM.
    Checks that the ReLU function correctly drops negative influences and 
    that the output tensor is safely normalized between 0.0 and 1.0.
    """
    # Create an activation map where only the top-left and bottom-right are active
    activations = torch.zeros(1, 1, 7, 7)
    activations[0, 0, 0, 0] = 10.0
    activations[0, 0, 6, 6] = 5.0

    # Reward the top-left (positive gradient), penalize the bottom-right (negative gradient)
    gradients = torch.zeros(1, 1, 7, 7)
    gradients[0, 0, 0, 0] = 1.0
    gradients[0, 0, 6, 6] = -1.0

    cam = generate_grad_cam(activations, gradients)

    # The penalized zone should be exactly 0.0 due to ReLU
    assert cam[6, 6] == 0.0

    # The rewarded zone should be the maximum value (normalized to 1.0)
    assert cam[0, 0] == 1.0

    # Ensure global bounds are respected
    assert np.min(cam) >= 0.0
    assert np.max(cam) <= 1.0

def test_process_canvas_mask_spatial_mapping():
    """
    Non-Trivial Test 2: Verify the translation from a high-res Streamlit RGBA 
    canvas down to a 7x7 tensor preserves the correct spatial location.
    """
    canvas = np.zeros((224, 224, 4), dtype=np.uint8)

    # Paint the entire top half of the canvas (set alpha channel to 255)
    canvas[0:112, :, 3] = 255

    mask_tensor = process_canvas_mask(canvas, target_size=(7, 7))

    # The resulting tensor should contain 1.0s in the top half (rows 0, 1, 2)
    assert torch.all(mask_tensor[0, 0, 0:3, :] == 1.0)

    # The resulting tensor should contain 0.0s in the bottom half (rows 4, 5, 6)
    assert torch.all(mask_tensor[0, 0, 4:7, :] == 0.0)

def test_sanity_train_step_execution():
    """
    Sanity Check: The training step should execute without errors, return 
    valid loss floats, and successfully update the active weights of the model.
    """
    model = HookedResNet18()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    image_tensor = torch.randn(1, 3, 224, 224)
    label = torch.tensor([0])
    user_mask = torch.ones(1, 1, 7, 7)

    pre_update_weight = model.model.fc.weight.clone()

    loss_total, loss_ce, loss_attention = train_step(
        model, optimizer, image_tensor, label, user_mask, lambda_weight=1.0
    )

    assert isinstance(loss_total, float)
    assert isinstance(loss_ce, float)
    assert isinstance(loss_attention, float)

    post_update_weight = model.model.fc.weight
    assert not torch.equal(pre_update_weight, post_update_weight)

def test_attention_penalty_logic():
    """
    Non-Trivial Test 1: Verify the attention penalty math is correct. 
    If the user's mask covers the entire image, the penalty zone is empty.
    """
    model = HookedResNet18()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    image_tensor = torch.randn(1, 3, 224, 224)
    label = torch.tensor([1])
    user_mask = torch.ones(1, 1, 7, 7)

    _, _, loss_attention = train_step(
        model, optimizer, image_tensor, label, user_mask, lambda_weight=1.0
    )
    assert loss_attention == 0.0

def test_gradient_flow_isolation():
    """
    Non-Trivial Test 2: Verify that the combined backpropagation strictly 
    respects the frozen layers and only updates layer4 and the fc layer.
    """
    model = HookedResNet18()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    image_tensor = torch.randn(1, 3, 224, 224)
    label = torch.tensor([0])
    user_mask = torch.zeros(1, 1, 7, 7)

    pre_conv1_weight = model.model.conv1.weight.clone()
    pre_layer4_weight = model.model.layer4[0].conv1.weight.clone()

    train_step(model, optimizer, image_tensor, label, user_mask)

    post_conv1_weight = model.model.conv1.weight
    post_layer4_weight = model.model.layer4[0].conv1.weight

    assert torch.equal(pre_conv1_weight, post_conv1_weight)
    assert not torch.equal(pre_layer4_weight, post_layer4_weight)
