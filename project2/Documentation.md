# Visual Product Recommendation System

## Model Summary

This project recommends visually similar clothing using a pre-trained ResNet50 model as an unsupervised feature extractor. No model training or fine-tuning was done; the pre-trained weights are used as they are. The raw data came from the Vibrent Clothes Rental dataset on Kaggle. Of the 33,565 images copied to Google Drive, cleaning removed 10,542: 10,533 exact duplicates (matching MD5 file hash), 9 images under 50 pixels on either side, and 0 unreadable files. That left 23,023 images. For feature creation, each image passes through a transformation chain that resizes it to 224×224, converts it to a tensor and normalises it with ImageNet mean and standard deviation. The final classification layer of ResNet50 was removed, so the model outputs a 2,048-number embedding for each image. These embeddings were computed once for the whole catalogue and saved. To make a recommendation, the input image is embedded the same way and compared with every saved embedding using cosine similarity, and the 5 highest-scoring images are returned. On the test image below, all five matches scored between 0.899 and 0.913, above the 0.7 target. The model is deployed as a Streamlit web app: https://abilaschvijikaran-b7brzhodzpjrgjkozgy2ch.streamlit.app

**At a glance:** ResNet50 (IMAGENET1K_V1) · 2,048-dimensional embeddings · 224×224 input · cosine similarity · 23,023 catalogue images · top 5 results.

The pipeline is split into four scripts in `src/` (data cleaning, feature creation, model, recommendations), with shared settings in `src/config.py`, and `run_pipeline.py` runs all four in order. `notebooks/project2.ipynb` shows the same pipeline run in Colab with its outputs. Setup and reproduction steps are in `README.md`.

<img width="1762" height="394" alt="sample_output" src="https://github.com/user-attachments/assets/b9a17508-c041-4dd3-a164-4ce14225e575" />


## Limitations and Future Improvements

The catalogue only contains women's rental clothing, so recommendations for other categories are limited to whatever looks closest in that catalogue. The model was trained on general ImageNet photos rather than fashion, and it matches garment shape and pose more strongly than colour: a black T-shirt input returned grey and blue T-shirts among its top results. Duplicate removal only catches files that are byte-for-byte identical, so near-duplicates of the same product can still appear twice in the results. Only one test image was scored in detail, so the 0.899–0.913 range shows the system works but isn't a full evaluation. Future improvements would be a fashion-aware model such as CLIP or a Vision Transformer, fine-tuning on fashion images, adding a colour feature alongside the embedding, removing near-duplicates by grouping images whose embeddings are almost identical, and testing on a wider set of input images.
