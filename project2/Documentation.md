## Visual Product Recommendation System
### Model summary

This project recommends visually similar clothing using a pre-trained ResNet50 model as an unsupervised feature extractor. The raw data came from the vibrant clothes rental dataset on Kaggle. Of the 33565 images copied to my google drive, the cleaning process removed 10542 that consisted of 10533 exact duplicates, 9 images were under 50 pixels on either side and there were no unreadable files. Which left 23023 images. For feature creation each image passes through a transformation chain that resizes it to 224 x 224 then converts it to a tensor and normalizes it with ImageNet mean and standard deviation. The final classification layer of ResNet50 was removed so the model outputs a 2048 number embedding for each image. These embeddings were computed once for the whole catalog and saved. To make a recommendation the input image is embedded the same way and compared with every saved embedding using cosine similarity and the 5 highest scoring images are returned. On the test image below all five matches scored between 0.899 and 0.913 which is above the 0.7 target. The model is deployed as a Streamlit web app: https://abilaschvijikaran-b7brzhodzpjrgjkozgy2ch.streamlit.app/ 

<img width="1762" height="394" alt="sample_output" src="https://github.com/user-attachments/assets/b789bdc2-41fa-4518-9960-f2ccc5da4a18" />


## Limitations and Future improvements

The catalog only contains women's rental clothing which means recommendations for other categories are limited to whatever looks closest in that catalog. The model was trained on general ImageNet photos rather than fashion and it matches garment shape and pose more strongly than color for example a black t-shirt input returned grey and blue t-shirt among its top results. Duplicate removal only catches files that are byte-for-byte identical so near duplicates of the same product can still appear twice in the results. Future improvements would be a fashion aware model like CLIP or Vision transformer, fine-tuning on fashion images, adding a color feature with the embedding and removing near duplicates by grouping images whose embeddings are almost identical.
