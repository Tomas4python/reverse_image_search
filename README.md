# Reverse Image Search (RIS) Application

## Project Overview
The goal of this project was to develop a Reverse Image Search (RIS) application. 
This single-notebook project provides a complete workflow for:

- Downloading image dataset 'Caltech101'.
- Initializing models with EfficientNetB0 and EfficientNetV2 for comparison.
- Extracting image features using these pretrained CNN's.
- Performing similarity searches with K-Nearest Neighbors (KNN) and Cosine Similarity.
- Visualizing data with PCA and t-SNE for comparative analysis.
- Implementing a Flask web application to interact with the RIS model.

## Technologies Used
- **TensorFlow v2.10 and v2.16.1**: For GPU and CPU compatibility and performance.
- **EfficientNetB0 and EfficientNetV2**: CNN models for feature extraction.
- **Scikit-Learn**: For KNN and cosine similarity computations.
- **PCA and t-SNE**: For visualization and comparative analysis.
- **Flask**: For the web application backend.

## Project Files Structure
- `pp8_ris_efficientnetb0.ipynb` - Notebook with EfficientNetB0 code.
- `pp8_ris_efficientnetv2.ipynb` - Notebook with EfficientNetV2 code.
- `features_.npy` - Extracted features from the Caltech101 dataset.
- `labels_.npy` - Labels extracted from the Caltech101 dataset.
- `model.h5` - Saved fully trained model.
- `/new_images/` - Directory for independent test images.
- `/flask_app/app.py` - Flask web application code.
- `requirements.txt` is the file that lists all the Python packages and their versions
required to run the project on GPU, includes Python 3.10, TensorFlow 2.10, CUDA Toolkit
11.2 and cuDNN 8.1.0.

## Challenges
- **TensorFlow Compatibility**: Initial challenges with model saving in TensorFlow v2.10 led
to the use of TensorFlow v2.16.1, which, while resolving the saving issue, required running the
model on CPU, increasing processing time more than 3 times (on my computer).

## Conclusions
- **Model Performance**: EfficientNetB0 performed well in feature extraction, though EfficientNetV2
showed superior accuracy in identifying consecutive images of the same individual.
- **Visualization Tools**: PCA highlighted two distinct classes and other looked mixed, whereas t-SNE
provided more differentiated class separations.
- **Similarity Measures**: Cosine similarity proved slightly more accurate than KNN in maintaining
consistency among the top returned images.

## Flask Application
The Flask application integrates the RIS model and feature data, providing a interface for uploading
and searching images. The application performs as expected.

## Screenshots
Upload dialog screenshot: <br>
![Upload dialog](/flask_app/screenshot_1.png) <br>
Results window screenshot: <br>
![Upload dialog](/flask_app/screenshot_2.png)

