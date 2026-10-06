# Shared settings for the project scripts.
# All paths point at the Google Drive folder used in the Colab notebook.
# Change PROJECT_DIR if your data lives somewhere else.
PROJECT_DIR = "/content/drive/MyDrive/project2"

# Section 1: data cleaning
RAW_IMAGES_DIR = f"{PROJECT_DIR}/data/images"
CLEANED_IMAGES_DIR = f"{PROJECT_DIR}/cleaned_images_v2"
CLEANED_CSV = f"{PROJECT_DIR}/cleaned_images_v2.csv"
MIN_SIZE = 50

# Section 2: feature creation
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Section 3: model
EMBEDDINGS_FILE = f"{PROJECT_DIR}/embeddings_v2.pt"
BATCH_SIZE = 32

# Section 4: outputs
TOP_N = 5
SAMPLE_OUTPUT_FILE = f"{PROJECT_DIR}/sample_output.png"
