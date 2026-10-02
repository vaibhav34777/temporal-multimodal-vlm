import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

OUTPUT_DIR = "/kaggle/working/visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_CSV = "/kaggle/working/train_subset.csv"
VAL_CSV = "/kaggle/working/val_subset.csv"
FRAMES_DIR = "/kaggle/working/frames"
EMB_DIR_VAL = "/kaggle/working/embeddings/val"

CLASS_NAMES = [
    "Pushing [something] from left to right",
    "Pushing [something] from right to left",
    "Moving [something] up",
    "Moving [something] down",
    "Tearing [something] into two pieces"
]

def plot_frame_strip():
    df = pd.read_csv(VAL_CSV)
    fig, axes = plt.subplots(len(CLASS_NAMES), 8, figsize=(20, 12))
    fig.suptitle("Temporal Frame Strips per Action Class", fontsize=16, fontweight='bold', y=1.01)
    
    for row_idx, cls in enumerate(CLASS_NAMES):
        subset = df[df['class_name'] == cls]
        if len(subset) == 0:
            continue
        video_id = str(subset.iloc[0]['video_id'])
        vid_dir = os.path.join(FRAMES_DIR, video_id)
        frame_files = sorted([f for f in os.listdir(vid_dir) if f.endswith('.jpg')])
        
        for col_idx, fname in enumerate(frame_files[:8]):
            img = Image.open(os.path.join(vid_dir, fname))
            axes[row_idx, col_idx].imshow(img)
            axes[row_idx, col_idx].axis('off')
            if col_idx == 0:
                short_name = cls.replace('[something]', 'obj').replace('[', '').replace(']', '')
                axes[row_idx, col_idx].set_ylabel(short_name, fontsize=8, rotation=0, labelpad=80, va='center')
            axes[row_idx, col_idx].set_title(f"F{col_idx+1}", fontsize=7)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "frame_strips.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

def plot_cosine_similarity():
    df = pd.read_csv(VAL_CSV)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("DINOv2 Embedding Cosine Similarity Across Frames", fontsize=14, fontweight='bold')

    directional_class = "Pushing [something] from left to right"
    control_class = "Tearing [something] into two pieces"

    for ax, cls, title_suffix in zip(axes, [directional_class, control_class], ["Directional Class", "Control Class"]):
        subset = df[df['class_name'] == cls].head(10)
        all_sims = []
        for _, row in subset.iterrows():
            emb_path = os.path.join(EMB_DIR_VAL, f"{row['video_id']}.npy")
            if not os.path.exists(emb_path):
                continue
            emb = np.load(emb_path)
            sims = [cosine_similarity(emb[i:i+1], emb[i+1:i+2])[0][0] for i in range(len(emb)-1)]
            all_sims.append(sims)

        all_sims = np.array(all_sims)
        mean_sims = all_sims.mean(axis=0)
        std_sims = all_sims.std(axis=0)
        x = np.arange(1, len(mean_sims)+1)

        ax.plot(x, mean_sims, marker='o', color='steelblue', linewidth=2, label='Mean cosine sim')
        ax.fill_between(x, mean_sims - std_sims, mean_sims + std_sims, alpha=0.2, color='steelblue')
        ax.set_xlabel("Frame Transition (Frame N to N+1)", fontsize=11)
        ax.set_ylabel("Cosine Similarity", fontsize=11)
        ax.set_title(f"{title_suffix}\n({cls.replace('[something]', 'obj')})", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f"F{i} to F{i+1}" for i in range(1, len(mean_sims)+1)], rotation=30, fontsize=8)
        ax.set_ylim(0.6, 1.05)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "cosine_similarity.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

def plot_progressive_observation():
    frames_used = [1, 2, 3, 4, 5, 6, 7, 8]
    accuracies =  [32.57, 36.22, 40.55, 46.24, 60.59, 69.70, 74.26, 77.22]
    baseline_acc = 59.45

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(frames_used, accuracies, marker='o', color='steelblue', linewidth=2.5, markersize=8, label='Temporal GRU (N frames)')
    ax.axhline(y=baseline_acc, color='coral', linestyle='--', linewidth=2, label=f'Baseline MLP (single frame): {baseline_acc}%')
    ax.axhline(y=20, color='gray', linestyle=':', linewidth=1.5, label='Random chance: 20%')

    for x, y in zip(frames_used, accuracies):
        ax.annotate(f"{y}%", (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

    ax.set_xlabel("Number of Frames Observed", fontsize=12)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=12)
    ax.set_title("Progressive Observation: GRU Accuracy vs. Number of Frames", fontsize=13, fontweight='bold')
    ax.set_xticks(frames_used)
    ax.set_ylim(10, 90)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "progressive_observation.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

def plot_shuffling_experiment():
    conditions = ['Baseline MLP\n(Single Frame)', 'GRU\n(Ordered Sequence)', 'GRU\n(Shuffled Sequence)']
    accuracies = [59.45, 77.22, 57.40]
    colors = ['coral', 'steelblue', 'mediumpurple']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(conditions, accuracies, color=colors, width=0.5, edgecolor='black', linewidth=0.8)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.8, f"{acc}%",
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.axhline(y=20, color='gray', linestyle=':', linewidth=1.5, label='Random chance: 20%')
    ax.set_ylabel("Validation Accuracy (%)", fontsize=12)
    ax.set_title("Temporal Order Matters: Shuffling Experiment", fontsize=13, fontweight='bold')
    ax.set_ylim(0, 95)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.legend(fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "shuffling_experiment.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

def plot_dataset_statistics():
    df_train = pd.read_csv(TRAIN_CSV)
    df_val = pd.read_csv(VAL_CSV)

    short_names = [
        "Push L to R", "Push R to L",
        "Move Up", "Move Down", "Tearing"
    ]
    train_counts = df_train['class_name'].value_counts().reindex(CLASS_NAMES).values
    val_counts = df_val['class_name'].value_counts().reindex(CLASS_NAMES).values

    x = np.arange(len(short_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, train_counts, width, label='Train', color='steelblue', edgecolor='black', linewidth=0.7)
    ax.bar(x + width/2, val_counts, width, label='Validation', color='coral', edgecolor='black', linewidth=0.7)

    ax.set_xlabel("Action Class", fontsize=12)
    ax.set_ylabel("Number of Videos", fontsize=12)
    ax.set_title("Dataset Statistics: Samples per Class", fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "dataset_statistics.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

print("Generating all visualizations...")
plot_dataset_statistics()
plot_frame_strip()
plot_cosine_similarity()
plot_progressive_observation()
plot_shuffling_experiment()
print(f"\nAll visualizations saved to {OUTPUT_DIR}")
