# Project 2: Visual Product Recommendation System

Upload a photo of a piece of clothing and get back the 5 most visually similar products from a catalogue of 23,023 women's rental clothing images.

**Live app:** https://abilaschvijikaran-b7brzhodzpjrgjkozgy2ch.streamlit.app

## Project Status

- Data pipeline: Complete (23,023 cleaned images)
- Model: Pre-trained ResNet50, embeddings created
- Live app: Running on Streamlit Cloud

## Project structure

This repository holds two projects: `project/` is Project 1 (medical text summarisation) and `project2/` is this project.

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
├── sample_output.png          
├── Documentation.md           
└── README.md
```

The scripts in `src/` are the main, reproducible project code. The notebook is the record of the pipeline being run and tested in Colab, with its outputs visible. `run_pipeline.py` runs all four stages in order.

## How it works

1. **Data cleaning:** 33,565 images from the Vibrent Clothes Rental dataset (Kaggle) were checked. 10,533 exact duplicates (same MD5 file hash) and 9 images under 50 pixels were removed, leaving 23,023. No files were unreadable.
2. **Feature creation:** each image is resized to 224×224, converted to a tensor and normalised with ImageNet mean and standard deviation.
3. **Model:** a pre-trained ResNet50 with its final layer removed turns each image into a 2,048-number embedding. All catalogue embeddings are saved once.
4. **Recommendation:** an input image is embedded the same way and compared with every saved embedding using cosine similarity. The 5 highest scores are returned.

## How to reproduce it

All data paths come from `PROJECT_DIR` in `src/config.py`. By default this is the Google Drive folder `/content/drive/MyDrive/project2`. The raw images must be in `PROJECT_DIR/data/images` before the pipeline runs.

### Option 1: Google Colab (recommended)

1. Open a new Colab notebook, preferably with a GPU runtime (Runtime > Change runtime type).
2. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount("/content/drive")
   ```
3. Download the [Vibrent Clothes Rental dataset](https://www.kaggle.com/datasets/kaborg15/vibrent-clothes-rental-dataset) and copy it to Drive (about 6 GB):
   ```python
   import kagglehub, shutil
   path = kagglehub.dataset_download("kaborg15/vibrent-clothes-rental-dataset")
   shutil.copytree(path, "/content/drive/MyDrive/project2/data", dirs_exist_ok=True)
   ```
4. Clone this repository and install the packages:
   ```
   !git clone https://github.com/abilaschvijikaran-creator/abilaschVijikaran.git
   %cd abilaschVijikaran/project2
   !pip install -r requirements.txt
   ```
5. Run the full pipeline:
   ```
   !python run_pipeline.py
   ```

To follow the same steps with explanations and outputs, open `notebooks/project2.ipynb` in Colab and run the cells from top to bottom. If a session disconnects, the first cell of section 4 reloads the saved embeddings from Drive, so sections 1 to 3 don't need to run again.

### Option 2: Local machine

1. Clone the repository and install the packages:
   ```bash
   git clone https://github.com/abilaschvijikaran-creator/abilaschVijikaran.git
   cd abilaschVijikaran/project2
   pip install -r requirements.txt
   ```
2. Download the [Vibrent Clothes Rental dataset](https://www.kaggle.com/datasets/kaborg15/vibrent-clothes-rental-dataset) and extract it so the images are in `project2/data/images`.
3. In `src/config.py`, change `PROJECT_DIR` to the full path of your `project2` folder.
4. Run the full pipeline:
   ```bash
   python run_pipeline.py
   ```

### Running the pipeline

`python run_pipeline.py` runs the four stages in order and stops if one fails. Stage 4 tests with the first cleaned catalogue image. To test with your own photo instead, run `python run_pipeline.py path/to/photo.jpg`.

Each stage can also be run on its own:

```bash
python src/data_processing.py       # cleans the images and prints what was removed
python src/feature_creation.py      # checks the transform on one image
python src/model.py                 # creates and saves the 23,023 embeddings
python src/recommend.py photo.jpg   # prints the top 5 matches and saves sample_output.png
```

Creating the embeddings (stage 3) is the slowest step, especially without a GPU.

## Technical Specifications

- **Model**: ResNet50 (ImageNet1K_V1 weights)
- **Embedding dimension**: 2,048
- **Image size**: 224×224
- **Similarity metric**: Cosine similarity
- **Catalogue size**: 23,023 images
- **Recommendations returned**: Top 5
- **Data source**: [Vibrent Clothes Rental (Kaggle)](https://www.kaggle.com/datasets/kaborg15/vibrent-clothes-rental-dataset)

## How the App is Deployed

- Hosted on Streamlit Community Cloud with Python 3.11.
- The embeddings, file names and thumbnails are stored in the public Hugging Face dataset `abz2781/clothing-recommender-data`. The app downloads them when it starts.
- `clothing-recommender/requirements.txt` links CPU-only PyTorch 2.1.2 wheels built for Linux and Python 3.11, to match Streamlit Cloud. To run the app on another system, install `torch` and `torchvision` for that system instead, then run `streamlit run streamlit_app.py` from the `clothing-recommender` folder.

## Performance

- Test case (T-shirt): similarity scores 0.899–0.913 (see `sample_output.png`)
- Target threshold: 0.7
- Only one test image was scored in detail, so this shows the system works rather than being a full evaluation.
