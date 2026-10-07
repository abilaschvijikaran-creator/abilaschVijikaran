# Project 2: Visual Product Recommendation System

Upload a photo of a piece of clothing and get back the 5 most visually similar products from a catalogue of 23,023 women's rental clothing images.

**Live app:** https://abilaschvijikaran-b7brzhodzpjrgjkozgy2ch.streamlit.app

## Project Status

- Data pipeline: Complete (23,023 cleaned images)
- Model: Pre-trained ResNet50, embeddings created
- Live app: Running on Streamlit Cloud

## Project structure

```
project2/
├── notebooks/
│   └── project2.ipynb
├── src/
│   ├── config.py          
│   ├── data_processing.py      
│   ├── feature_creation.py     
│   ├── model.py               
│   └── recommend.py            
├── clothing-recommender/
│   ├── streamlit_app.py        
│   └── requirements.txt        
├── requirements.txt            
├── run_pipeline.py
├── Documentation.md            
└── README.md
```

The scripts in `src/` are the main, reproducible project code. The notebook is the record of the pipeline being run and tested in Colab, with its outputs visible. `run_pipeline.py` executes all stages sequentially.

## How it works

1. **Data cleaning:** 33,565 images from the Vibrent Clothes Rental dataset (Kaggle) were checked. 10,533 exact duplicates (same MD5 file hash) and 9 images under 50 pixels were removed, leaving 23,023. No files were unreadable.
2. **Feature creation:** each image is resized to 224×224, converted to a tensor and normalised with ImageNet mean and standard deviation.
3. **Model:** a pre-trained ResNet50 with its final layer removed turns each image into a 2,048-number embedding. All catalogue embeddings are saved once.
4. **Recommendation:** an input image is embedded the same way and compared with every saved embedding using cosine similarity. The 5 highest scores are returned.

## Quick Start (Local Machine)

To run the pipeline locally:

```bash
# Clone and navigate
git clone https://github.com/abilaschvijikaran-creator/abilaschVijikaran.git
cd abilaschVijikaran/project2

# Install dependencies
pip install -r requirements.txt

# Run all stages at once
python run_pipeline.py

# Or run individual stages
python src/data_processing.py      
python src/feature_creation.py     
python src/model.py                
python src/recommend.py photo.jpg
```

## How to Reproduce (Full Setup)

The scripts read and write to a local or cloud folder. By default, they use Google Drive (`/content/drive/MyDrive/project2`). To use a different location, change `PROJECT_DIR` in `src/config.py`.

### Option 1: Google Colab (Original method)

1. Open a new Colab notebook, preferably with a GPU runtime.
2. Mount Google Drive and download the dataset into `MyDrive/project2/data` (the first cells of `notebooks/project2.ipynb` do this with `kagglehub`).
3. Clone this repository and install dependencies:
   ```
   !git clone https://github.com/abilaschvijikaran-creator/abilaschVijikaran.git
   %cd abilaschVijikaran/project2
   !pip install -r requirements.txt
   ```
4. Run the pipeline scripts as shown above, or open `notebooks/project2.ipynb` in Colab and run the cells from top to bottom.

If a session disconnects, the first cell of section 4 reloads the saved embeddings from Drive, so sections 1 to 3 don't need to run again.

### Option 2: Local Machine

1. Clone the repository and navigate to the project folder.
2. Download the Vibrent Clothes Rental dataset from [Kaggle](https://www.kaggle.com/datasets/abilaschvijikaran/vibrent-clothes-rental) and extract it to `data/images/`.
3. Install dependencies: `pip install -r requirements.txt`
4. Run the pipeline scripts in order (see Quick Start above).
5. Update `PROJECT_DIR` in `src/config.py` if your data is in a different location.

## Technical Specifications

- **Model**: ResNet50 (ImageNet1K_V1 weights)
- **Embedding dimension**: 2,048
- **Image size**: 224×224
- **Similarity metric**: Cosine similarity
- **Catalogue size**: 23,023 images
- **Recommendations returned**: Top 5
- **Data source**: Vibrent Clothes Rental (Kaggle)

## How the App is Deployed

- Hosted on Streamlit Community Cloud with Python 3.11.
- The embeddings, file names and thumbnails are stored in the public Hugging Face dataset `abz2781/clothing-recommender-data`. The app downloads them when it starts.
- `clothing-recommender/requirements.txt` links CPU-only PyTorch 2.1.2 wheels built for Linux and Python 3.11, to match Streamlit Cloud. To run the app on another system, install `torch` and `torchvision` for that system instead, then run `streamlit run streamlit_app.py` from the `clothing-recommender` folder.

## Performance

- Test case (T-shirt): Similarity scores 0.899–0.913
- Target threshold: 0.7 
- Model performs well on visual similarity matching
