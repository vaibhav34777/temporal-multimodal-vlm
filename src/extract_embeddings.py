import os
import pandas as pd
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np
from tqdm import tqdm

def extract_embeddings(csv_path, frames_dir, output_dir, model_name="facebook/dinov2-small"):
    os.makedirs(output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device}...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    df = pd.read_csv(csv_path)
    
    print(f"Extracting embeddings for {len(df)} videos in {csv_path}...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        video_id = str(row['video_id'])
        vid_dir = os.path.join(frames_dir, video_id)
        
        if not os.path.exists(vid_dir):
            continue
            
        frame_files = sorted([f for f in os.listdir(vid_dir) if f.endswith('.jpg')])
        
        embeddings = []
        with torch.no_grad():
            for frame_file in frame_files:
                img_path = os.path.join(vid_dir, frame_file)
                image = Image.open(img_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(device)
                
                outputs = model(**inputs)
                emb = outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
                embeddings.append(emb)
                
        if len(embeddings) > 0:
            embeddings = np.array(embeddings)
            np.save(os.path.join(output_dir, f"{video_id}.npy"), embeddings)
            
    print(f"Done! Saved embeddings to {output_dir}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, default='/kaggle/working/train_subset.csv')
    parser.add_argument('--val_csv', type=str, default='/kaggle/working/val_subset.csv')
    parser.add_argument('--frames_dir', type=str, default='/kaggle/working/frames')
    parser.add_argument('--output_train', type=str, default='/kaggle/working/embeddings/train')
    parser.add_argument('--output_val', type=str, default='/kaggle/working/embeddings/val')
    parser.add_argument('--model', type=str, default='facebook/dinov2-small')
    args, _ = parser.parse_known_args()

    if os.path.exists(args.train_csv):
        extract_embeddings(args.train_csv, args.frames_dir, args.output_train, args.model)
    if os.path.exists(args.val_csv):
        extract_embeddings(args.val_csv, args.frames_dir, args.output_val, args.model)