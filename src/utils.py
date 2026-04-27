# src/utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def generate_grad_cam(activations, gradients):
    """
    Generates a Grad-CAM heatmap from the model's spatial activations and gradients.

    Args:
        activations (torch.Tensor): The feature maps from layer4. Shape: (1, C, H, W)
        gradients (torch.Tensor): The gradients flowing into layer4. Shape: (1, C, H, W)

    Returns:
        numpy.ndarray: A 2D heatmap normalized between 0.0 and 1.0.
    """
    # global average pooling on gradients to get importance weights in each channel
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

    # weight the activations by their important and sum accross channels
    cam = torch.sum(weights * activations, dim=1, keepdim=True)

    # ReLU to only visualize positive influence
    cam = F.relu(cam)

    # normalize the heatmap (between 0 and 1) for visual renderingz
    cam = cam - torch.min(cam)
    cam = cam / (torch.max(cam) + 1e-8)

    # squeeze out batch/channel dims and convert to numpy
    return cam.squeeze().detach().cpu().numpy()


def overlay_heatmap(base_image, heatmap, alpha=0.5):
    """
    Overlays a 2D mathematical heatmap onto a standard PIL image.

    Args:
        base_image (PIL.Image): The original unmodified image.
        heatmap (numpy.ndarray): The 2D array [0, 1] from generate_grad_cam.
        alpha (float): Transparency of the heatmap overlay.

    Returns:
        PIL.Image: The blended composite image.
    """
    # resize the heatmap for the high-res image
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(
        base_image.size, resample=Image.Resampling.BICUBIC
    )

    # blue = low attention; red = high attention
    cmap = plt.get_cmap("jet")
    heatmap_colored = cmap(np.array(heatmap_img) / 255.0)

    # drop alpha channel and convert to PIL image
    heatmap_colored = np.uint8(255 * heatmap_colored[:, :, :3])
    heatmap_colored_img = Image.fromarray(heatmap_colored)

    # blend the original image and the heatmap
    blended = Image.blend(base_image.convert("RGB"), heatmap_colored_img, alpha)
    return blended


def process_canvas_mask(canvas_rgba, target_size=(7, 7)):
    """
    Translates the user's Streamlit canvas drawing into a PyTorch tensor
    matching the spatial dimensions of the model's feature maps.

    Args:
        canvas_rgba (numpy.ndarray): The raw Streamlit canvas output (H, W, 4).
        target_size (tuple): The spatial dimensions of layer4 (typically 7x7).

    Returns:
        torch.Tensor: A binary mask of shape (1, 1, H, W). 1 = drawn, 0 = empty.
    """
    # extract the alpha channel (index 3 = transparency/drawing)
    alpha_channel = canvas_rgba[:, :, 3]

    # convert to binary matrix (1 where user painted, 0 otherwise)
    binary_mask = (alpha_channel > 0).astype(np.float32)

    # convert to pytorch tensor with batch and channel dimensions
    mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0).unsqueeze(0)

    # downsample high-res mask to match low-res layer4 dimensions
    resized_mask = F.adaptive_avg_pool2d(mask_tensor, target_size)

    # re-binarize after interpolation
    final_mask = (resized_mask > 0.5).float()

    return final_mask


def train_step(model, optimizer, image_tensor, label, user_mask, lambda_weight=1.0):
    """
    Executes a single interactive fine-tuning step. Penalizes the model
    for focusing outside the human-defined bounding box.
    """
    model.train()

    # 0. Prevent Gradient Compensation by freezing downstream layers temporarily
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = False

    optimizer.zero_grad()

    # 1. Forward Pass & Standard Loss
    outputs = model(image_tensor)
    criterion = nn.CrossEntropyLoss()
    loss_ce = criterion(outputs, label)

    # 2. Attention Penalty Loss
    activations = model.get_activations()
    penalty_zone = 1.0 - user_mask
    # Calculate the relative ratio of background activations to total activations
    penalized_sum = torch.sum(activations * penalty_zone)
    total_sum = torch.sum(activations) + 1e-8

    loss_attention = penalized_sum / total_sum

    # 3. Combined Optimization
    loss_total = loss_ce + (lambda_weight * loss_attention)

    # 4. Backpropagation
    loss_total.backward()
    optimizer.step()

    # 5. Unfreeze downstream layers for standard baseline evaluation
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True

    return loss_total.item(), loss_ce.item(), loss_attention.item()


def evaluate_model(model, dataset, batch_size=32):

    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            # Handle both 2-item and 3-item tuples safely
            if len(batch) == 3:
                tensor_images, _, labels = batch
            else:
                tensor_images, labels = batch

            outputs = model(tensor_images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total == 0:
        return 0.0
    return 100.0 * correct / total
