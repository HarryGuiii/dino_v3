import gradio as gr
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import pandas as pd
import json
import os
import torch.nn.functional as F

# --- CONFIGURATION ---
MODEL_NAME = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
EMBEDDINGS_FILE = "embeddings.csv"
TOP_K = 10

# --- GLOBAL VARIABLES ---
model = None
processor = None
df = None
embeddings_tensor = None

def load_resources():
    global model, processor, df, embeddings_tensor
    
    print("Loading resources...")
    
    # 1. Load Model & Processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16)
        model.eval()
        print("Model loaded.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise e

    # 2. Load Embeddings
    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"Embeddings file not found: {EMBEDDINGS_FILE}")
        raise FileNotFoundError(f"{EMBEDDINGS_FILE} not found. Run generate_embeddings.py first.")
    
    print(f"Loading embeddings from {EMBEDDINGS_FILE}...")
    df = pd.read_csv(EMBEDDINGS_FILE)
    
    # Convert string embeddings to lists then tensors
    # literal_eval is safer than json.loads for python string repr, but json.loads works if it's strict json
    # str([1, 2]) in python matches json format mostly.
    try:
        # We use json.loads assuming the format is standard JSON-like list
        tqdm_pandas = False
        try:
            from tqdm import tqdm
            tqdm.pandas()
            tqdm_pandas = True
        except ImportError:
            pass
            
        if tqdm_pandas:
             df['embedding'] = df['embedding'].progress_apply(json.loads)
        else:
             df['embedding'] = df['embedding'].apply(json.loads)
             
        embeddings_list = df['embedding'].tolist()
        embeddings_tensor = torch.tensor(embeddings_list).to(device) # Keep on GPU for fast search if possible
        
        # Normalize stored embeddings for cosine similarity (if not already, but dinov3 might not be normalized)
        # Cosine similarity = (A . B) / (|A| * |B|)
        # If we normalize A and B, it becomes just A . B
        embeddings_tensor = F.normalize(embeddings_tensor, p=2, dim=1)
        
        print(f"Loaded {len(df)} embeddings.")
        
    except Exception as e:
        print(f"Failed to parse embeddings: {e}")
        raise e

def search(input_image):
    global model, processor, df, embeddings_tensor
    
    if input_image is None:
        return []

    if model is None:
        load_resources()

    device = model.device
    
    # 1. Preprocess Input Image
    try:
        # Ensure RGB
        if input_image.mode != "RGB":
            input_image = input_image.convert("RGB")
            
        inputs = processor(images=input_image, return_tensors="pt").to(device).to(torch.bfloat16)
        
        # 2. Generate Embedding
        with torch.no_grad():
            outputs = model(**inputs)
            # Pooler output is usually the embedding for classification/retrieval
            query_embedding = outputs.pooler_output # Shape: (1, Hidden_Dim)
            
        # 3. Compute Similarity
        # Normalize query
        query_embedding = F.normalize(query_embedding.float(), p=2, dim=1)
        
        # Cosine Similarity: Dot product of normalized vectors
        # embeddings_tensor is already normalized and on device (if we moved it there)
        # If embeddings_tensor is on CPU, we might want to move query to CPU or embeddings to GPU
        # Ideally embeddings are on GPU if VRAM allows.
        
        if embeddings_tensor.device != query_embedding.device:
            embeddings_tensor = embeddings_tensor.to(query_embedding.device)

        similarities = torch.matmul(query_embedding, embeddings_tensor.T).squeeze(0) # (N,)
        
        # 4. Get Top K
        top_k_scores, top_k_indices = torch.topk(similarities, TOP_K)
        
        # 5. Retrieve Images
        results = []
        for score, idx in zip(top_k_scores.cpu().tolist(), top_k_indices.cpu().tolist()):
            file_path = df.iloc[idx]['file_path']
            
            # Resolve relative path if necessary
            # The csv contains paths relative to where generate_embeddings.py was run (root)
            # So if we run this app from root, it should be fine.
            if os.path.exists(file_path):
                results.append((file_path, f"Similarity: {score:.4f}"))
            else:
                results.append((None, f"File missing: {file_path} ({score:.4f})"))
                
        return results

    except Exception as e:
        print(f"Error during search: {e}")
        return []

# --- APP UI ---
def build_app():
    # Load resources immediately on startup
    load_resources()
    
    with gr.Blocks(title="DINOv3 Image Search") as app:
        gr.Markdown("# DINOv3 Image-to-Image Search")
        gr.Markdown("Upload an image to find the top 10 most similar images from the dataset.")
        
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(type="pil", label="Upload Query Image")
                search_btn = gr.Button("Search", variant="primary")
            
            with gr.Column():
                gallery = gr.Gallery(label="Top 10 Results", columns=2, height=800)
        
        search_btn.click(fn=search, inputs=input_img, outputs=gallery)
        
    return app

if __name__ == "__main__":
    app = build_app()
    app.launch(share=False) # Set share=True if you want a public link
