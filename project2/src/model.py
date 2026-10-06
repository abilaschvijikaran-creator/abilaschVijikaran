"""Section 3: the recommendation model.

A pre-trained ResNet50 with its final classification layer removed, so it outputs a 2,048-number embedding for each image instead of a class label. Embeddings for the whole catalogue are computed once and saved.

Run from the project2 folder (a GPU makes this much faster):
    python src/model.py
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from config import BATCH_SIZE, CLEANED_CSV, CLEANED_IMAGES_DIR, EMBEDDINGS_FILE
from feature_creation import CleanedImageDataset


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(device=None):
    """Load ResNet50 without its final layer, ready to produce embeddings."""
    device = device or get_device()
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    model = model.to(device)
    return model


def generate_embeddings(model, csv_path=CLEANED_CSV, image_dir=CLEANED_IMAGES_DIR,
                        batch_size=BATCH_SIZE, device=None):
# Return one 2,048-number embedding per cleaned image, in CSV order
    device = device or get_device()
    loader = DataLoader(CleanedImageDataset(csv_path, image_dir), batch_size=batch_size, shuffle=False)

    embeddings_list = []

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            embedding = model(batch.to(device))
            embedding = embedding.squeeze(-1).squeeze(-1)
            embeddings_list.append(embedding.cpu())

            if idx % 50 == 0:
                print(f"Processed: {idx * batch_size}/{len(loader.dataset)}")

    embeddings = torch.cat(embeddings_list, dim=0)
    return embeddings


if __name__ == "__main__":
    device = get_device()
    model = load_model(device)
    print(f"Model loaded on {device}")

    embeddings = generate_embeddings(model, device=device)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Each image has {embeddings.shape[1]} dimensional embedding")

    torch.save(embeddings, EMBEDDINGS_FILE)
    print("Embeddings saved")
