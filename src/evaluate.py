import os
import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from dataset import EmbeddingDataset
from models import SingleFrameBaseline, TemporalGRU, TemporalTransformer

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


def evaluate_progressive(model, loader, device, model_name="Model"):
    model.eval()
    results = {}
    print(f"\n--- Progressive Observation Experiment ({model_name}) ---")
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
        print(f"  Frames used: {num_frames} -> Accuracy: {acc:.2f}%")
    return results


def evaluate_shuffled(model, loader, device, model_name="Model"):
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
    print(f"\n--- Temporal Order Shuffling Experiment ({model_name}) ---")
    print(f"  Accuracy with Shuffled Frames: {acc:.2f}%\n")
    return acc


def load_model(model, path, device):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_csv', type=str, default='/kaggle/working/val_subset.csv')
    parser.add_argument('--emb_dir_val', type=str, default='/kaggle/working/embeddings/val')
    parser.add_argument('--model_dir', type=str, default='/kaggle/working/models')
    args, _ = parser.parse_known_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_dataset = EmbeddingDataset(args.val_csv, args.emb_dir_val)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    single_frame = SingleFrameBaseline(input_dim=384, num_classes=5).to(device)
    if load_model(single_frame, os.path.join(args.model_dir, "SingleFrameBaseline.pth"), device):
        evaluate_per_class(single_frame, val_loader, device, "Single-Frame Baseline (Center Frame)")

    gru = TemporalGRU(input_dim=384, hidden_dim=256, num_classes=5, num_layers=1).to(device)
    if load_model(gru, os.path.join(args.model_dir, "TemporalGRU.pth"), device):
        evaluate_per_class(gru, val_loader, device, "Temporal GRU (8 Frames)")
        evaluate_progressive(gru, val_loader, device, "GRU")
        evaluate_shuffled(gru, val_loader, device, "GRU")

    transformer_pe = TemporalTransformer(input_dim=384, num_heads=4, num_layers=2, ff_dim=512,
                                          num_classes=5, dropout=0.4, use_positional_encoding=True).to(device)
    if load_model(transformer_pe, os.path.join(args.model_dir, "TemporalTransformerWithPE.pth"), device):
        evaluate_per_class(transformer_pe, val_loader, device, "Temporal Transformer (With PE)")
        evaluate_progressive(transformer_pe, val_loader, device, "Transformer+PE")
        evaluate_shuffled(transformer_pe, val_loader, device, "Transformer+PE")

    transformer_nope = TemporalTransformer(input_dim=384, num_heads=4, num_layers=2, ff_dim=512,
                                            num_classes=5, dropout=0.4, use_positional_encoding=False).to(device)
    if load_model(transformer_nope, os.path.join(args.model_dir, "TemporalTransformerNoPE.pth"), device):
        evaluate_per_class(transformer_nope, val_loader, device, "Temporal Transformer (No PE - Order-Free)")
        evaluate_shuffled(transformer_nope, val_loader, device, "Transformer NoPE")
