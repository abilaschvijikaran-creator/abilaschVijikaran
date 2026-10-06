"""Section 4: output functions.

- compare_two_images: cosine similarity score between any two images
- find_similar_images: the N most similar catalogue images to an input image
- show_recommendations: the input image next to its matches, each with its score

Run from the project2 folder with any clothing photo:
    python src/recommend.py path/to/photo.jpg
"""

import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from config import CLEANED_CSV, CLEANED_IMAGES_DIR, EMBEDDINGS_FILE, SAMPLE_OUTPUT_FILE, TOP_N
from feature_creation import transform_image
from model import get_device, load_model


class Recommender:
    def __init__(self, embeddings_file=EMBEDDINGS_FILE, csv_path=CLEANED_CSV,
                 image_dir=CLEANED_IMAGES_DIR, model=None):
        self.device = get_device()
        self.model = model if model is not None else load_model(self.device)
        self.embeddings = torch.load(embeddings_file, map_location="cpu").flatten(1)
        self.filenames = pd.read_csv(csv_path)["filename"].tolist()
        self.image_dir = image_dir

    def get_embedding(self, image):
        """Turn a file path or PIL image into a (1, 2048) embedding."""
        tensor = transform_image(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model(tensor).flatten(1).cpu()

    def compare_two_images(self, image_a, image_b):
        """Cosine similarity between two images: 1.0 means identical."""
        return F.cosine_similarity(self.get_embedding(image_a), self.get_embedding(image_b)).item()

    def find_similar_images(self, image, n=TOP_N):
        """Score the image against every saved embedding and keep the closest n."""
        scores = F.cosine_similarity(self.get_embedding(image), self.embeddings)
        top_scores, top_indices = torch.topk(scores, n)
        return [self.filenames[i] for i in top_indices.tolist()], top_scores.tolist()

    def show_recommendations(self, image_path, n=TOP_N, save_path=SAMPLE_OUTPUT_FILE):
        """Plot the input image next to its n matches and save the figure."""
        filenames, scores = self.find_similar_images(image_path, n)
        fig, axes = plt.subplots(1, n + 1, figsize=(3 * (n + 1), 4))

        axes[0].imshow(Image.open(image_path).convert("RGB"))
        axes[0].set_title("Input image")
        for ax, fname, score in zip(axes[1:], filenames, scores):
            ax.imshow(Image.open(f"{self.image_dir}/{fname}").convert("RGB"))
            ax.set_title(f"Score: {score:.3f}")
        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return filenames, scores


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python src/recommend.py path/to/photo.jpg")

    recommender = Recommender()
    filenames, scores = recommender.show_recommendations(sys.argv[1])
    print(f"Top {len(filenames)} matches:")
    for fname, score in zip(filenames, scores):
        print(f"  {score:.3f}  {fname}")
    print(f"Figure saved to {SAMPLE_OUTPUT_FILE}")
