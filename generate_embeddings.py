import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm
import math

# --- CONFIGURATION ---
SOURCE_FOLDER = "./source/test"
OUTPUT_FILE = "embeddings.csv"          # Output file name
BATCH_SIZE = 16                         # Start with 16 for RTX 4080
# Model: DINOv3 ViT-H+ (Huge+) - 840M parameters
MODEL_NAME = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
# ---------------------

def main():
    print("Starting the script...")

    # 1. Setup Device
    # We let accelerate/transformers handle device mapping, but we check for CUDA for logging
    if not torch.cuda.is_available():
        print("WARNING: CUDA not found. Running on CPU will be extremely slow.")
    else:
        print(f"✅ Running on {torch.cuda.get_device_name(0)}")

    # 2. Load DINOv3 Model
    print(f"🚀 Loading DINOv3 Model ({MODEL_NAME})...")
    try:
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.bfloat16)
        print("✅ Model loaded in BFloat16 (BF16)")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 4. Scan Folder for Images
    print(f"📂 Scanning folder: {SOURCE_FOLDER}")
    all_files = [
        os.path.join(dp, f) 
        for dp, dn, filenames in os.walk(SOURCE_FOLDER) 
        for f in filenames 
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
    ]
    print(f"Found {len(all_files)} images.")

    # 5. Check for existing progress to Resume
    processed_paths = set()
    if os.path.exists(OUTPUT_FILE):
        print("Found existing output file. Checking for duplicates...")
        try:
            # Read just the first column (file paths) to save memory
            existing_df = pd.read_csv(OUTPUT_FILE, usecols=['file_path'])
            processed_paths = set(existing_df['file_path'].tolist())
            print(f"Skipping {len(processed_paths)} already processed images.")
        except Exception as e:
            print(f"Could not read existing file (might be empty or corrupt): {e}")

    # Filter out already processed images
    files_to_process = [f for f in all_files if f not in processed_paths]
    
    if not files_to_process:
        print("🎉 All images already processed!")
        return

    # 6. Batch Processing Loop
    num_batches = math.ceil(len(files_to_process) / BATCH_SIZE)
    
    print("🔥 Starting Inference...")
    
    # Open file in append mode
    # If new file, write header. If exists, skip header.
    write_header = not os.path.exists(OUTPUT_FILE)
    
    for i in tqdm(range(num_batches), desc="Processing Batches"):
        batch_files = files_to_process[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
        
        valid_images = []
        valid_files = []

        # Pre-process images
        for file_path in batch_files:
            try:
                img = Image.open(file_path).convert('RGB')
                valid_images.append(img)
                valid_files.append(file_path)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        if not valid_images:
            continue

        # Prepare inputs using the processor
        # 'pt' = PyTorch tensors
        try:
            inputs = processor(images=valid_images, return_tensors="pt").to(model.device).to(torch.bfloat16)
        except Exception as e:
            print(f"Error processing batch: {e}")
            continue

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            # DINOv3 "pooler_output" contains the representation
            embeddings = outputs.pooler_output # Shape: (Batch, Hidden_Dim)
            
        # Move to CPU and convert to list
        embeddings_list = embeddings.float().cpu().numpy().tolist()

        # Save to CSV immediately (Append mode)
        data = {
            'file_path': valid_files,
            'embedding': [str(e) for e in embeddings_list] 
        }
        df_batch = pd.DataFrame(data)
        df_batch.to_csv(OUTPUT_FILE, mode='a', header=write_header, index=False)
        write_header = False # Only write header once

    print(f"✅ Done! Embeddings saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()