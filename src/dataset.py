import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    def __init__(self, csv_file, embeddings_dir):
        self.data = pd.read_csv(csv_file)
        self.embeddings_dir = embeddings_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_id = str(row['video_id'])
        label = row['class_id']
        emb_path = os.path.join(self.embeddings_dir, f"{video_id}.npy")
        embedding = np.load(emb_path)
        return torch.tensor(embedding, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
