import os
import json
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

OUTPUT_DIR = "/kaggle/working/visualizations"
if not os.path.exists("/kaggle/working"):
    OUTPUT_DIR = "visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULTS_DIR = "/kaggle/working/results"
if not os.path.exists("/kaggle/working"):
    RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_CSV = "/kaggle/working/train_subset.csv"
if not os.path.exists(TRAIN_CSV):
    TRAIN_CSV = "data/train_subset.csv"

VAL_CSV = "/kaggle/working/val_subset.csv"
if not os.path.exists(VAL_CSV):
    VAL_CSV = "data/val_subset.csv"

FRAMES_DIR = "/kaggle/working/frames"
if not os.path.exists(FRAMES_DIR):
    FRAMES_DIR = "frames"

EMB_DIR_VAL = "/kaggle/working/embeddings/val"
if not os.path.exists(EMB_DIR_VAL):
    EMB_DIR_VAL = "embeddings/val"

CLASS_NAMES = [
    "Pushing [something] from left to right",
    "Pushing [something] from right to left",
    "Moving [something] up",
    "Moving [something] down",
    "Tearing [something] into two pieces"
]

SHORT_NAMES = ["Push L→R", "Push R→L", "Move Up", "Move Down", "Tearing"]


def plot_frame_strip():
    if not os.path.exists(VAL_CSV) or not os.path.exists(FRAMES_DIR):
        return
    df = pd.read_csv(VAL_CSV)
    fig, axes = plt.subplots(len(CLASS_NAMES), 8, figsize=(20, 12))
    fig.suptitle("Temporal Frame Strips per Action Class", fontsize=16, fontweight='bold', y=1.01)

    for row_idx, cls in enumerate(CLASS_NAMES):
        subset = df[df['class_name'] == cls]
        if len(subset) == 0:
            continue
        video_id = str(subset.iloc[0]['video_id'])
        vid_dir = os.path.join(FRAMES_DIR, video_id)
        if not os.path.exists(vid_dir):
            continue
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
    if not os.path.exists(VAL_CSV) or not os.path.exists(EMB_DIR_VAL):
        return
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

        if len(all_sims) == 0:
            continue
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
        ax.set_xticklabels([f"F{i}→F{i+1}" for i in range(1, len(mean_sims)+1)], rotation=30, fontsize=8)
        ax.set_ylim(0.6, 1.05)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "cosine_similarity.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_progressive_observation(
    gru_accs=None,
    transformer_pe_accs=None,
    single_frame_acc=None
):
    frames_used = [1, 2, 3, 4, 5, 6, 7, 8]

    if gru_accs is None:
        gru_accs = [31.89, 36.67, 42.82, 47.61, 60.82, 67.20, 72.44, 75.40]
    if transformer_pe_accs is None:
        transformer_pe_accs = [44.19, 44.65, 47.84, 52.39, 62.64, 70.16, 74.26, 77.22]
    if single_frame_acc is None:
        single_frame_acc = 55.81

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(frames_used, gru_accs, marker='o', color='coral', linewidth=2.5, markersize=8, label='Temporal GRU')
    ax.plot(frames_used, transformer_pe_accs, marker='s', color='mediumpurple', linewidth=2.5, markersize=8, label='Temporal Transformer (With PE)')

    ax.axhline(y=single_frame_acc, color='darkgreen', linestyle='-.', linewidth=2,
               label=f'Single-Frame Baseline: {single_frame_acc:.2f}%')
    ax.axhline(y=20, color='gray', linestyle=':', linewidth=1.5, label='Random chance: 20%')

    for x, y in zip(frames_used, gru_accs):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=8, fontweight='bold', color='coral')
    for x, y in zip(frames_used, transformer_pe_accs):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, -15),
                    ha='center', fontsize=8, fontweight='bold', color='mediumpurple')

    ax.set_xlabel("Number of Frames Observed", fontsize=12)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=12)
    ax.set_title("Progressive Observation: Accuracy vs. Number of Frames", fontsize=13, fontweight='bold')
    ax.set_xticks(frames_used)
    ax.set_ylim(15, 92)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "progressive_observation.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_shuffling_experiment(
    single_frame_acc=None,
    gru_ordered_acc=None,
    gru_shuffled_acc=None,
    transformer_pe_ordered_acc=None,
    transformer_pe_shuffled_acc=None,
    transformer_nope_ordered_acc=None,
    transformer_nope_shuffled_acc=None,
    vlm_ordered_acc=None,
    vlm_shuffled_acc=None
):
    if single_frame_acc is None:
        single_frame_acc = 55.81
    if gru_ordered_acc is None:
        gru_ordered_acc = 75.40
    if gru_shuffled_acc is None:
        gru_shuffled_acc = 55.13
    if transformer_pe_ordered_acc is None:
        transformer_pe_ordered_acc = 77.22
    if transformer_pe_shuffled_acc is None:
        transformer_pe_shuffled_acc = 56.26
    if transformer_nope_ordered_acc is None:
        transformer_nope_ordered_acc = 61.96
    if transformer_nope_shuffled_acc is None:
        transformer_nope_shuffled_acc = 61.96
    if vlm_ordered_acc is None:
        vlm_ordered_acc = 68.00
    if vlm_shuffled_acc is None:
        vlm_shuffled_acc = 44.00

    labels = [
        'Single-Frame\nBaseline',
        'GRU\n(Ordered)',
        'GRU\n(Shuffled)',
        'Transformer\n+PE (Ordered)',
        'Transformer\n+PE (Shuffled)',
        'Transformer\nNoPE (Ordered)',
        'Transformer\nNoPE (Shuffled)',
        'VLM\n(Ordered)',
        'VLM\n(Shuffled)',
    ]
    accs = [
        single_frame_acc,
        gru_ordered_acc,
        gru_shuffled_acc,
        transformer_pe_ordered_acc,
        transformer_pe_shuffled_acc,
        transformer_nope_ordered_acc,
        transformer_nope_shuffled_acc,
        vlm_ordered_acc,
        vlm_shuffled_acc,
    ]
    bar_colors = [
        'darkgreen',
        'coral', 'lightcoral',
        'mediumpurple', 'plum',
        'goldenrod', 'moccasin',
        'teal', 'lightseagreen',
    ]

    fig, ax = plt.subplots(figsize=(15, 5))
    bars = ax.bar(labels, accs, color=bar_colors, width=0.55, edgecolor='black', linewidth=0.8)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.8,
                f"{acc:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.axhline(y=20, color='gray', linestyle=':', linewidth=1.5, label='Random chance: 20%')
    ax.set_ylabel("Validation Accuracy (%)", fontsize=12)
    ax.set_title("Model Comparison: Ordered vs. Shuffled Frames (All Models + VLM)", fontsize=13, fontweight='bold')
    ax.set_ylim(0, 97)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.legend(fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "shuffling_experiment.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_per_class_accuracy():
    single_frame_accs = [51.55, 61.70, 32.95, 35.62, 94.25]
    gru_accs          = [76.29, 79.79, 55.68, 64.38, 98.85]
    transformer_pe    = [64.95, 77.66, 72.73, 71.23, 100.00]
    transformer_nope  = [59.79, 68.09, 39.77, 39.73, 98.85]

    x = np.arange(len(SHORT_NAMES))
    width = 0.2

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - 1.5*width, single_frame_accs, width, label='Single-Frame Baseline', color='darkgreen', edgecolor='black', linewidth=0.7)
    ax.bar(x - 0.5*width, gru_accs,          width, label='Temporal GRU',          color='coral',     edgecolor='black', linewidth=0.7)
    ax.bar(x + 0.5*width, transformer_pe,    width, label='Transformer + PE',       color='mediumpurple', edgecolor='black', linewidth=0.7)
    ax.bar(x + 1.5*width, transformer_nope,  width, label='Transformer NoPE',       color='goldenrod', edgecolor='black', linewidth=0.7)

    ax.set_xlabel("Action Class", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Per-Class Accuracy Across All Models", fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_NAMES, fontsize=10)
    ax.set_ylim(0, 112)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "per_class_accuracy.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_vlm_per_class(vlm_results_path):
    if not os.path.exists(vlm_results_path):
        return
    with open(vlm_results_path) as f:
        vlm_data = json.load(f)
    records = vlm_data.get("records", [])
    if not records:
        return

    ordered_by_class = {name: [] for name in CLASS_NAMES}
    shuffled_by_class = {name: [] for name in CLASS_NAMES}

    for rec in records:
        cls = rec.get("class_name")
        if cls in ordered_by_class:
            ordered_by_class[cls].append(rec.get("ordered_correct", 0))
            shuffled_by_class[cls].append(rec.get("shuffled_correct", 0))

    ordered_accs = []
    shuffled_accs = []
    for cls in CLASS_NAMES:
        o = ordered_by_class[cls]
        s = shuffled_by_class[cls]
        ordered_accs.append(100.0 * sum(o) / len(o) if o else 0.0)
        shuffled_accs.append(100.0 * sum(s) / len(s) if s else 0.0)

    x = np.arange(len(SHORT_NAMES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    bars_o = ax.bar(x - width/2, ordered_accs,  width, label='Ordered Frames',  color='teal',         edgecolor='black', linewidth=0.7)
    bars_s = ax.bar(x + width/2, shuffled_accs, width, label='Shuffled Frames', color='lightseagreen', edgecolor='black', linewidth=0.7)

    for bar, acc in zip(bars_o, ordered_accs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.8,
                f"{acc:.0f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar, acc in zip(bars_s, shuffled_accs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.8,
                f"{acc:.0f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

    model_label = vlm_data.get("model", "GPT-4o")
    overall_ordered  = vlm_data.get("ordered_acc", 0)
    overall_shuffled = vlm_data.get("shuffled_acc", 0)
    ax.axhline(y=20, color='gray', linestyle=':', linewidth=1.5, label='Random chance: 20%')
    ax.set_xlabel("Action Class", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(
        f"VLM ({model_label}) Per-Class Accuracy — Ordered {overall_ordered:.1f}% vs Shuffled {overall_shuffled:.1f}%",
        fontsize=12, fontweight='bold'
    )
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_NAMES, fontsize=10)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "vlm_per_class_accuracy.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_dataset_statistics():
    if not os.path.exists(TRAIN_CSV) or not os.path.exists(VAL_CSV):
        return
    df_train = pd.read_csv(TRAIN_CSV)
    df_val = pd.read_csv(VAL_CSV)

    train_counts = df_train['class_name'].value_counts().reindex(CLASS_NAMES).fillna(0).values
    val_counts = df_val['class_name'].value_counts().reindex(CLASS_NAMES).fillna(0).values

    x = np.arange(len(SHORT_NAMES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, train_counts, width, label='Train',      color='steelblue', edgecolor='black', linewidth=0.7)
    ax.bar(x + width/2, val_counts,   width, label='Validation', color='coral',     edgecolor='black', linewidth=0.7)

    ax.set_xlabel("Action Class", fontsize=12)
    ax.set_ylabel("Number of Videos", fontsize=12)
    ax.set_title("Dataset Statistics: Samples per Class", fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_NAMES, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "dataset_statistics.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def zip_visualizations(output_dir, zip_name="visualizations.zip"):
    zip_path = os.path.join(os.path.dirname(output_dir), zip_name)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(output_dir):
            fpath = os.path.join(output_dir, fname)
            if os.path.isfile(fpath):
                zf.write(fpath, arcname=fname)
    print(f"Zipped all visualizations to: {zip_path}")
    return zip_path


if __name__ == "__main__":
    vlm_results_path = os.path.join(OUTPUT_DIR, "vlm_accuracy_results.json")

    print("Generating all visualizations...")
    plot_dataset_statistics()
    plot_frame_strip()
    plot_cosine_similarity()
    plot_progressive_observation()
    plot_shuffling_experiment()
    plot_per_class_accuracy()
    plot_vlm_per_class(vlm_results_path)
    print(f"All visualizations saved to {OUTPUT_DIR}")

    zip_visualizations(OUTPUT_DIR)
