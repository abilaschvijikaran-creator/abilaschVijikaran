"""Section 2: feature creation.

Every image goes through the same transformation chain before it reaches the model:
1. Resize to 224x224, the size ResNet50 expects
2. Convert to a tensor which are numbers Pytorch can use
3. Normalise with the ImageNet mean and standard deviation the model was trained with

Run from the project2 folder to check the pipeline on one cleaned image:
    python src/feature_creation.py
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd

from config import CLEANED_CSV, CLEANED_IMAGES_DIR, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


# Define Image Transformation Pipeline
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])


def transform_image(image):
# Turn a file path or PIL image into a normalised 3x224x224 tensor
    if isinstance(image, str):
        image = Image.open(image)
    return image_transform(image.convert("RGB"))


class CleanedImageDataset(Dataset):
    def __init__(self, csv_path=CLEANED_CSV, image_dir=CLEANED_IMAGES_DIR):
        self.filenames = pd.read_csv(csv_path)["filename"].tolist()
        self.image_dir = image_dir

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        return transform_image(f"{self.image_dir}/{self.filenames[idx]}")


if __name__ == "__main__":
    dataset = CleanedImageDataset()
    first = f"{dataset.image_dir}/{dataset.filenames[0]}"
    original = Image.open(first)
    tensor = dataset[0]
    print(f"Images in dataset: {len(dataset)}")
    print(f"Original size: {original.size}")
    print(f"Transformed shape: {tuple(tensor.shape)}")
    print(f"Tensor value range: [{tensor.min():.4f}, {tensor.max():.4f}]")
