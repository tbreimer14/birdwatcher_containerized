# tests/test_model.py
import pytest
import torch
from src.model import HookedResNet18

def test_sanity_initialization_and_freezing():
    """
    Sanity Check: The model should initialize, output exactly 2 classes,
    and correctly freeze early layers to optimize for fast HCI interactions.
    """
    model = HookedResNet18()
    
    # 1. Check structural modification
    assert model.model.fc.out_features == 2
    
    # 2. Check gradient tracking (Freezing logic)
    frozen_params = [name for name, param in model.named_parameters() if not param.requires_grad]
    active_params = [name for name, param in model.named_parameters() if param.requires_grad]
    
    assert "model.conv1.weight" in frozen_params
    assert "model.layer1.0.conv1.weight" in frozen_params
    assert "model.layer4.0.conv1.weight" in active_params
    assert "model.fc.weight" in active_params

def test_forward_hook_captures_activations():
    """
    Non-Trivial Test 1: Verify the forward hook successfully intercepts and 
    stores the 7x7 spatial feature maps from layer4 during inference.
    """
    model = HookedResNet18()
    model.eval() 
    
    # Standard input tensor (Batch=1, Channels=3, H=224, W=224)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    assert model.get_activations() is None
    
    # Trigger the forward pass
    output = model(dummy_input)
    activations = model.get_activations()
    
    assert output.shape == (1, 2)
    
    # layer4 spatial activations in ResNet18 should be 512 channels, 7x7 spatial
    assert activations is not None
    assert activations.shape == (1, 512, 7, 7)

def test_backward_hook_captures_gradients():
    """
    Non-Trivial Test 2: Verify the backward hook successfully captures the 
    gradients flowing back through layer4, necessary for Grad-CAM weighting.
    """
    model = HookedResNet18()
    model.train() 
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Forward pass
    output = model(dummy_input)
    
    # Create a dummy scalar loss and trigger backpropagation
    loss = output.sum()
    
    assert model.get_gradients() is None
    
    loss.backward()
    gradients = model.get_gradients()
    
    # Gradients should match the exact dimensional shape of the activations
    assert gradients is not None
    assert gradients.shape == (1, 512, 7, 7)