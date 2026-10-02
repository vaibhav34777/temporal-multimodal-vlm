import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class EmbeddingDataset(Dataset):
    """
    A fast PyTorch dataset that loads pre-extracted DINOv2 embeddings from numpy files.
    """
    def __init__(self, csv_file, embeddings_dir):
        """
        csv_file: Path to train_subset.csv or val_subset.csv
        embeddings_dir: Directory where extracted .npy embeddings are stored
        """
        self.data = pd.read_csv(csv_file)
        self.embeddings_dir = embeddings_dir
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_id = str(row['video_id'])
        label = row['class_id']
        
        # Load the saved embedding numpy array: shape (num_frames, embedding_dim)
        emb_path = os.path.join(self.embeddings_dir, f"{video_id}.npy")
        embedding = np.load(emb_path)
        
        return torch.tensor(embedding, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
