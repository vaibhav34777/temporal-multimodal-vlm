import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from dataset import EmbeddingDataset
from models import BaselineMLP, TemporalGRU

CLASS_NAMES = [
    "Pushing [something] from left to right",
    "Pushing [something] from right to left",
    "Moving [something] up",
    "Moving [something] down",
    "Tearing [something] into two pieces"
]

def find_case_examples(mlp, gru, loader, df, device):
    mlp.eval()
    gru.eval()
    
    cases = {
        "gru_win": None,
        "both_correct": None,
        "both_fail": None,
        "mlp_win": None
    }
    
    idx_counter = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            mlp_preds = mlp(inputs).max(1)[1]
            gru_preds = gru(inputs).max(1)[1]
            
            for i in range(labels.size(0)):
                true_lbl = labels[i].item()
                mlp_p = mlp_preds[i].item()
                gru_p = gru_preds[i].item()
                
                vid_id = str(df.iloc[idx_counter]["video_id"])
                idx_counter += 1
                
                mlp_corr = (mlp_p == true_lbl)
                gru_corr = (gru_p == true_lbl)
                
                record = {
                    "video_id": vid_id,
                    "true_class": CLASS_NAMES[true_lbl],
                    "mlp_pred": CLASS_NAMES[mlp_p],
                    "gru_pred": CLASS_NAMES[gru_p],
                    "mlp_correct": mlp_corr,
                    "gru_correct": gru_corr
                }
                
                if gru_corr and not mlp_corr and cases["gru_win"] is None:
                    cases["gru_win"] = record
                elif gru_corr and mlp_corr and cases["both_correct"] is None:
                    cases["both_correct"] = record
                elif not gru_corr and not mlp_corr and cases["both_fail"] is None:
                    cases["both_fail"] = record
                elif not gru_corr and mlp_corr and cases["mlp_win"] is None:
                    cases["mlp_win"] = record
                    
            if all(v is not None for v in cases.values()):
                break
                
    return cases

def plot_qualitative_grid(cases, frames_dir, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    case_order = [
        ("Case 1: Temporal Context Resolves Direction (GRU Correct, Baseline Failed)", cases["gru_win"]),
        ("Case 2: Strong Static Spatial Cues (Both Models Correct)", cases["both_correct"]),
        ("Case 3: Severe Ambiguity / Blur (Both Models Failed)", cases["both_fail"]),
        ("Case 4: Single-Frame Shortcut / Spurious Clue (Baseline Correct, GRU Failed)", cases["mlp_win"])
    ]
    
    valid_cases = [c for c in case_order if c[1] is not None]
    num_rows = len(valid_cases)
    if num_rows == 0:
        return
        
    fig, axes = plt.subplots(num_rows, 8, figsize=(22, 3.6 * num_rows))
    if num_rows == 1:
        axes = np.expand_dims(axes, 0)
        
    for r_idx, (title, case_data) in enumerate(valid_cases):
        vid_id = case_data["video_id"]
        vid_folder = os.path.join(frames_dir, vid_id)
        frame_files = sorted([f for f in os.listdir(vid_folder) if f.endswith('.jpg')]) if os.path.exists(vid_folder) else []
        
        row_header = (
            f"{title}\n"
            f"Video ID: {vid_id} | Ground Truth: {case_data['true_class']}\n"
            f"Baseline Pred: {case_data['mlp_pred']} [{'CORRECT' if case_data['mlp_correct'] else 'WRONG'}] | "
            f"GRU Pred: {case_data['gru_pred']} [{'CORRECT' if case_data['gru_correct'] else 'WRONG'}]"
        )
        
        for c_idx in range(8):
            ax = axes[r_idx, c_idx]
            if c_idx < len(frame_files):
                img_path = os.path.join(vid_folder, frame_files[c_idx])
                img = Image.open(img_path)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "Frame N/A", ha='center', va='center')
                
            ax.axis('off')
            ax.set_title(f"F{c_idx+1}", fontsize=8)
            
            if c_idx == 0:
                ax.text(0.0, 1.35, row_header, transform=ax.transAxes,
                        fontsize=9.5, fontweight='bold', va='bottom', ha='left',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='#F0F4F8', edgecolor='#CBD5E0'))

    plt.subplots_adjust(hspace=0.6, wspace=0.08, top=0.92, bottom=0.04)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Qualitative success/failure figure saved to: {save_path}")

if __name__ == "__main__":
    val_csv = "/kaggle/working/val_subset.csv"
    if not os.path.exists(val_csv):
        val_csv = "data/val_subset.csv"
        
    emb_dir_val = "/kaggle/working/embeddings/val"
    if not os.path.exists(emb_dir_val):
        emb_dir_val = "embeddings/val"
        
    frames_dir = "/kaggle/working/frames"
    if not os.path.exists(frames_dir):
        frames_dir = "frames"
        
    mlp_path = "/kaggle/working/models/BaselineMLP.pth"
    if not os.path.exists(mlp_path):
        mlp_path = "models/BaselineMLP.pth"
        
    gru_path = "/kaggle/working/models/TemporalGRU.pth"
    if not os.path.exists(gru_path):
        gru_path = "models/TemporalGRU.pth"
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if os.path.exists(val_csv) and os.path.exists(emb_dir_val):
        df_val = pd.read_csv(val_csv)
        val_dataset = EmbeddingDataset(val_csv, emb_dir_val)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        mlp = BaselineMLP(input_dim=384, num_classes=5).to(device)
        gru = TemporalGRU(input_dim=384, hidden_dim=256, num_classes=5, num_layers=1).to(device)
        
        if os.path.exists(mlp_path) and os.path.exists(gru_path):
            mlp.load_state_dict(torch.load(mlp_path, map_location=device))
            gru.load_state_dict(torch.load(gru_path, map_location=device))
            
            cases = find_case_examples(mlp, gru, val_loader, df_val, device)
            
            out_img = "/kaggle/working/visualizations/model_success_failure.png"
            if not os.path.exists("/kaggle/working"):
                out_img = "visualizations/model_success_failure.png"
                
            plot_qualitative_grid(cases, frames_dir, out_img)
