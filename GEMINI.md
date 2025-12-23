# DINOv3 Image Embedding Generator

This project is a Python-based utility designed to generate high-quality image embeddings using the [DINOv3 Huge+ model](https://github.com/facebookresearch/dinov3) (`dinov3-vith16plus-pretrain-lvd1689m`) from Facebook Research. It leverages Hugging Face Transformers for model loading and inference, optimized for NVIDIA GPUs (CUDA) with automatic mixed precision.

## Project Structure

- **`generate_embeddings.py`**: The core script. It loads the pre-trained DINOv3 model via Hugging Face, scans the `source/test` directory for images, processes them in batches, and saves the resulting embedding vectors to a CSV file.
- **`requirements.txt`**: Lists the necessary Python dependencies (`torch`, `transformers`, `accelerate`, `pillow`, `pandas`, `tqdm`).
- **`source/`**: Directory containing the input images.
- **`embeddings.csv`**: The output file where file paths and their corresponding embedding vectors are stored.

## Setup & Installation

1.  **Environment**: This project is configured for **Python 3.12**.
2.  **Dependencies**: Install the required packages using pip:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Ensure you have a version of PyTorch installed that supports your CUDA version for GPU acceleration.*

## Usage

To generate embeddings, run the main script:

```bash
python generate_embeddings.py
```

### Configuration

You can modify the following constants at the top of `generate_embeddings.py` to adjust the script's behavior:

- `SOURCE_FOLDER`: Path to the directory containing images (default: `"./source/test"`).
- `OUTPUT_FILE`: Name of the CSV file to save results (default: `"embeddings.csv"`).
- `BATCH_SIZE`: Number of images to process at once (default: `16`). Adjust this based on your GPU VRAM.
- `MODEL_NAME`: The Hugging Face model identifier (default: `"facebook/dinov3-vith16plus-pretrain-lvd1689m"`).

### Features

- **Resume Capability**: If `embeddings.csv` already exists, the script will skip images that have already been processed.
- **DINOv3 Integration**: Uses the latest DINOv3 architecture for state-of-the-art visual features.
- **FP16 & Device Optimization**: Automatically utilizes GPU (CUDA) and Half-Precision (FP16) via the `accelerate` and `transformers` libraries.
- **Recursive Scanning**: The script searches for images recursively within the source folder.

## Output Format

The `embeddings.csv` file contains two columns:
1.  `file_path`: The relative path to the image file.
2.  `embedding`: The computed feature vector (as a string list).