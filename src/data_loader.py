# data_loader.py
import os
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset


class TrickWaterbirdsDataset(Dataset):
    """
    A custom PyTorch Dataset that loads the waterbirds dataset locally and isolates
    images where the subject label conflicts with the background label.
    """

    def __init__(self, split="train", data_dir="./data", return_pil=True, filter=True):
        os.makedirs(data_dir, exist_ok=True)
        self.return_pil = return_pil

        # load datasets and cache
        self.dataset = load_dataset(
            "grodino/waterbirds", split=split, cache_dir=data_dir
        )

        # filter for spurious correlation
        if filter:
            self.filtered_data = [
                item for item in self.dataset if item["label"] != item["place"]
            ]
        else:
            self.filtered_data = list(self.dataset)

        # transform for the front end
        self.base_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
            ]
        )

        # transform with normalization
        self.tensor_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.filtered_data)

    def __getitem__(self, idx):
        item = self.filtered_data[idx]

        image = item["image"].convert("RGB")
        label = item["label"]

        raw_pil_image = self.base_transform(image)
        tensor_image = self.tensor_transform(raw_pil_image)

        if self.return_pil:
            return tensor_image, raw_pil_image, label
        return tensor_image, label
