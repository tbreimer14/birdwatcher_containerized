# src/model.py
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class HookedResNet18(nn.Module):
    """
    A modified ResNet18 optimized for Explanatory Interactive Learning.
    Exposes layer4 spatial activations and gradients for Grad-CAM.
    """

    def __init__(self):
        super().__init__()

        # pre-trained resnet18
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # modify final layer for binary classification
        # idea: save database of actual classified birds
        # and do the binary classification later
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)

        # storage for hooks
        self.activations = None
        self.gradients = None

        self._freeze_layers()
        self._register_hooks()

    def _freeze_layers(self):
        for name, param in self.model.named_parameters():
            if "layer3" not in name and "layer4" not in name and "fc" not in name:
                param.requires_grad = False

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            # grad_output is a tuple
            # we want the gradient with respect to the output tensor
            self.gradients = grad_output[0]

        # target final convolution block before pooling and flattening
        target_layer = self.model.layer3
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def forward(self, x):
        return self.model(x)

    def get_activations(self):
        return self.activations

    def get_gradients(self):
        return self.gradients
