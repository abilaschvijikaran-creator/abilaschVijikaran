import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from torch import nn
from torch.nn.functional import cosine_similarity
from PIL import Image
import pandas as pd
import zipfile
import os
from huggingface_hub import hf_hub_download

DATA_REPO = "abz2781/clothing-recommender-data"

@st.cache_resource
def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model

@st.cache_resource
def load_data():
    embeddings = torch.load(hf_hub_download(DATA_REPO, "embeddings.pt", repo_type="dataset"), map_location="cpu")
    embeddings = embeddings.reshape(embeddings.shape[0], -1)
    filenames = pd.read_csv(hf_hub_download(DATA_REPO, "filenames.csv", repo_type="dataset"))
    return embeddings, filenames

@st.cache_resource
def load_thumbnails():
    thumb_zip = hf_hub_download(DATA_REPO, "thumbnails.zip", repo_type="dataset")
    with zipfile.ZipFile(thumb_zip, "r") as zip_ref:
        zip_ref.extractall("/tmp/thumbnails")
    return "/tmp/thumbnails"

def get_embedding(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(img_tensor)
    return embedding.reshape(-1)

def find_similar(query_embedding, all_embeddings, n=5):
    # Compare the query vector against every stored vector, then keep the top n
    similarities = cosine_similarity(all_embeddings, query_embedding.unsqueeze(0), dim=1)
    top_scores, top_indices = torch.topk(similarities, n)
    return top_indices.tolist(), top_scores.tolist()

st.title("Clothing Recommender")
st.write("Upload a clothing image to find similar items from our database of 23,023 products")

model = load_model()
embeddings, filenames = load_data()
thumb_dir = load_thumbnails()

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, caption="Your Image", use_column_width=True)

    query_embedding = get_embedding(image, model)
    top_indices, top_scores = find_similar(query_embedding, embeddings, n=5)

    st.write("### Top 5 Similar Products")

    cols = st.columns(5)
    for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
        with cols[i]:
            filename = filenames.iloc[idx]["filename"]
            thumb_path = os.path.join(thumb_dir, filename)
            if os.path.exists(thumb_path):
                st.image(Image.open(thumb_path), use_column_width=True)
            st.metric("Similarity", f"{score:.3f}")
            st.caption(filename)
