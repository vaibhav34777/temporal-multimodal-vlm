import os
import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from dataset import EmbeddingDataset
from models import BaselineMLP, TemporalGRU

CLASS_NAMES = [
    "Pushing [something] from left to right",
    "Pushing [something] from right to left",
    "Moving [something] up",
    "Moving [something] down",
    "Tearing [something] into two pieces"
]

def evaluate_per_class(model, loader, device, model_name="Model"):
    model.eval()
    class_correct = [0] * len(CLASS_NAMES)
    class_total = [0] * len(CLASS_NAMES)
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total_samples += labels.size(0)
            total_correct += predicted.eq(labels).sum().item()
            
            for i in range(labels.size(0)):
                lbl = labels[i].item()
                if lbl < len(CLASS_NAMES):
                    class_total[lbl] += 1
                    if predicted[i].item() == lbl:
                        class_correct[lbl] += 1
                        
    overall_acc = 100. * total_correct / total_samples if total_samples > 0 else 0.0
    print(f"\n================ Per-Class Evaluation: {model_name} ================")
    print(f"{'Class Name':<45} | {'Samples':<8} | {'Accuracy'}")
    print("-" * 70)
    for i, name in enumerate(CLASS_NAMES):
        tot = class_total[i]
        corr = class_correct[i]
        acc = (100. * corr / tot) if tot > 0 else 0.0
        print(f"{name:<45} | {tot:<8} | {acc:.2f}%")
    print("-" * 70)
    print(f"{'Overall Accuracy':<45} | {total_samples:<8} | {overall_acc:.2f}%\n")
    return overall_acc

def evaluate_progressive(model, loader, device):
    model.eval()
    results = {}
    print("\n--- Progressive Observation Experiment (GRU) ---")
    for num_frames in range(1, 9):
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                truncated_inputs = inputs[:, :num_frames, :]
                outputs = model(truncated_inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        acc = 100. * correct / total
        results[num_frames] = acc
        print(f"Frames used: {num_frames} -> Accuracy: {acc:.2f}%")
    return results

def evaluate_shuffled(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            batch_size, seq_len, emb_dim = inputs.shape
            shuffled_inputs = torch.zeros_like(inputs)
            for i in range(batch_size):
                indices = torch.randperm(seq_len)
                shuffled_inputs[i] = inputs[i, indices, :]
                
            outputs = model(shuffled_inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    acc = 100. * correct / total
    print("\n--- Temporal Order Shuffling Experiment (GRU) ---")
    print(f"Accuracy with Shuffled Frames: {acc:.2f}%\n")
    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_csv', type=str, default='/kaggle/working/val_subset.csv')
    parser.add_argument('--emb_dir_val', type=str, default='/kaggle/working/embeddings/val')
    parser.add_argument('--model_dir', type=str, default='/kaggle/working/models')
    args, _ = parser.parse_known_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_dataset = EmbeddingDataset(args.val_csv, args.emb_dir_val)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    mlp_path = os.path.join(args.model_dir, "BaselineMLP.pth")
    if not os.path.exists(mlp_path):
        mlp_path = "models/BaselineMLP.pth"
        
    gru_path = os.path.join(args.model_dir, "TemporalGRU.pth")
    if not os.path.exists(gru_path):
        gru_path = "models/TemporalGRU.pth"
        
    mlp = BaselineMLP(input_dim=384, num_classes=5).to(device)
    if os.path.exists(mlp_path):
        mlp.load_state_dict(torch.load(mlp_path, map_location=device))
        evaluate_per_class(mlp, val_loader, device, "Baseline MLP (Mean Pooling)")
        
    gru = TemporalGRU(input_dim=384, hidden_dim=256, num_classes=5, num_layers=1).to(device)
    if os.path.exists(gru_path):
        gru.load_state_dict(torch.load(gru_path, map_location=device))
        evaluate_per_class(gru, val_loader, device, "Temporal GRU (8 Frames)")
        evaluate_progressive(gru, val_loader, device)
        evaluate_shuffled(gru, val_loader, device)
