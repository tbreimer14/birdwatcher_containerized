import pytest
import os
import torch
from PIL import Image
from unittest.mock import patch
from src.data_loader import TrickWaterbirdsDataset


def mock_huggingface_dataset(*args, **kwargs):
    return [
        {"image": Image.new("RGB", (300, 300), color="brown"), "label": 0, "place": 0},
        {"image": Image.new("RGB", (300, 300), color="blue"), "label": 0, "place": 1},
        {"image": Image.new("RGB", (300, 300), color="cyan"), "label": 1, "place": 1},
        {"image": Image.new("RGB", (400, 200), color="green"), "label": 1, "place": 0},
    ]


@patch("src.data_loader.load_dataset", side_effect=mock_huggingface_dataset)
def test_sanity_initialization_and_caching(mock_load, tmp_path):
    test_data_dir = tmp_path / "test_data"
    dataset = TrickWaterbirdsDataset(split="train", data_dir=str(test_data_dir))

    assert os.path.exists(test_data_dir)
    mock_load.assert_called_once_with(
        "grodino/waterbirds", split="train", cache_dir=str(test_data_dir)
    )
    assert len(dataset) > 0


@patch("src.data_loader.load_dataset", side_effect=mock_huggingface_dataset)
def test_spurious_correlation_filtering(mock_load):
    dataset = TrickWaterbirdsDataset(split="train")
    assert len(dataset) == 2
    for item in dataset.filtered_data:
        assert item["label"] != item["place"]


@patch("src.data_loader.load_dataset", side_effect=mock_huggingface_dataset)
def test_dual_output_transforms(mock_load):
    dataset = TrickWaterbirdsDataset(split="train")
    tensor_image, raw_pil_image, label = dataset[0]

    assert isinstance(tensor_image, torch.Tensor)
    assert isinstance(raw_pil_image, Image.Image)
    assert isinstance(label, int)
    assert tensor_image.shape == (3, 224, 224)
    assert raw_pil_image.size == (224, 224)
