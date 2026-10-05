# Project 2: Visual Product Recommendation System

Upload a photo of a piece of clothing and get back the 5 most visually similar products from a catalogue of 23,023 women's rental clothing images.

**Live app:** https://abilaschvijikaran-b7brzhodzpjrgjkozgy2ch.streamlit.app

## What's in this folder

| File | What it is |
|---|---|
| `project2.ipynb` ( Full pipeline: data cleaning, feature creation, model and embeddings, output functions, model outputs )
| `Documentation.md` ( One-page model summary and limitations )
| `clothing-recommender/streamlit_app.py` (the web app code)
| `clothing-recommender/requirements.txt` ( packages for the web app )

## How it works

1. **Data cleaning:** 33,565 images from the Vibrent Clothes Rental dataset (Kaggle) were checked. 10,533 exact duplicates and 9 images under 50 pixels were removed, leaving 23,023.
2. **Feature creation:** each image is resized to 224×224, converted to a tensor and normalised with ImageNet mean and standard deviation.
3. **Model:** a pre-trained ResNet50 with its final layer removed turns each image into a 2,048-number embedding. All catalogue embeddings are saved once.
4. **Recommendation:** an input image is embedded the same way and compared with every saved embedding using cosine similarity. The 5 highest scores are returned.

## How to run the notebook

1. Open `project2.ipynb` in Google Colab.
2. Run the cells in order, top to bottom. The notebook mounts Google Drive and downloads the dataset from Kaggle with `kagglehub`.
3. The cleaning check and the embedding step take the longest because they go through every image.
4. If a session disconnects, the checkpoint cell in the Output Functions section reloads the saved embeddings from Drive, so the earlier sections don't need to run again.

## How the app is deployed

- Hosted on Streamlit Community Cloud with Python 3.11.
- The embeddings, file names and thumbnails are stored in the public Hugging Face dataset `abz2781/clothing-recommender-data`. The app downloads them when it starts.
- `requirements.txt` links CPU-only PyTorch 2.1.2 wheels built for Linux and Python 3.11, to match Streamlit Cloud. To run the app on another system, install `torch` and `torchvision` for that system instead, then run `streamlit run streamlit_app.py` from the `clothing-recommender` folder.
