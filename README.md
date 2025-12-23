# DINOv3 Image Search

This project utilizes the [DINOv3](https://github.com/facebookresearch/dinov3) model to generate high-quality image embeddings and provides a web interface for image-to-image similarity search.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Prepare Data**:
    Place your images in the `source/test` directory (or configure `SOURCE_FOLDER` in `generate_embeddings.py`).

## Usage

### 1. Generate Embeddings
First, generate the vector embeddings for your image dataset. This creates an `embeddings.csv` file.

```bash
python generate_embeddings.py
```

### 2. Run Search Application
Launch the Gradio web interface to search through your dataset using a query image.

```bash
python app.py
```
*   **Interface**: Upload an image to find the top 10 most similar images from your generated embeddings.
*   **Access**: Open the local URL provided in the terminal (usually `http://127.0.0.1:7860`).
