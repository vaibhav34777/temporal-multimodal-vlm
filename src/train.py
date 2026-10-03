import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import EmbeddingDataset
from models import SingleFrameBaseline, TemporalGRU, TemporalTransformer


def train_model(model, train_loader, val_loader, model_name, epochs=40, lr=1e-3, warmup_epochs=5, model_dir="/kaggle/working/models"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs - warmup_epochs),
        eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    best_acc = 0.0
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, f"{model_name}.pth")

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": []
    }

    print(f"\n================ Training {model_name} ================")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        epoch_train_loss = train_loss / total
        epoch_train_acc = 100. * correct / total

        epoch_val_loss, epoch_val_acc = evaluate(model, val_loader, device, criterion)

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        history["lr"].append(current_lr)

        print(f"Epoch {epoch+1:02d}/{epochs} | LR: {current_lr:.6f} | Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            torch.save(model.state_dict(), save_path)

    print(f"--> Best Validation Accuracy for {model_name}: {best_acc:.2f}%\n")
    return history


def evaluate(model, loader, device, criterion=None):
    model.eval()
    loss_val = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if criterion is not None:
                loss = criterion(outputs, labels)
                loss_val += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = loss_val / total if total > 0 else 0.0
    acc = 100. * correct / total if total > 0 else 0.0
    return avg_loss, acc


def plot_training_curves(histories, save_dir="/kaggle/working/visualizations"):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(list(histories.values())[0]["train_loss"]) + 1)

    colors = {
        "SingleFrameBaseline": "darkgreen",
        "TemporalGRU": "coral",
        "TemporalTransformerWithPE": "mediumpurple",
        "TemporalTransformerNoPE": "goldenrod",
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for name, hist in histories.items():
        c = colors.get(name, "black")
        axes[0].plot(epochs, hist["train_loss"], label=f"{name} (Train)", color=c, linestyle="--", alpha=0.6)
        axes[0].plot(epochs, hist["val_loss"], label=f"{name} (Val)", color=c)
        axes[1].plot(epochs, hist["train_acc"], label=f"{name} (Train)", color=c, linestyle="--", alpha=0.6)
        axes[1].plot(epochs, hist["val_acc"], label=f"{name} (Val)", color=c)

    axes[0].set_title("Training and Validation Loss", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epochs", fontsize=11)
    axes[0].set_ylabel("Cross Entropy Loss", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Training and Validation Accuracy", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epochs", fontsize=11)
    axes[1].set_ylabel("Accuracy (%)", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    out_file = os.path.join(save_dir, "training_curves.png")
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, default='/kaggle/working/train_subset.csv')
    parser.add_argument('--val_csv', type=str, default='/kaggle/working/val_subset.csv')
    parser.add_argument('--emb_dir_train', type=str, default='/kaggle/working/embeddings/train')
    parser.add_argument('--emb_dir_val', type=str, default='/kaggle/working/embeddings/val')
    parser.add_argument('--model_dir', type=str, default='/kaggle/working/models')
    parser.add_argument('--vis_dir', type=str, default='/kaggle/working/results')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    args, _ = parser.parse_known_args()

    train_dataset = EmbeddingDataset(args.train_csv, args.emb_dir_train)
    val_dataset = EmbeddingDataset(args.val_csv, args.emb_dir_val)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    histories = {}

    single_frame = SingleFrameBaseline(input_dim=384, num_classes=5, dropout=0.4)
    histories["SingleFrameBaseline"] = train_model(
        single_frame, train_loader, val_loader, "SingleFrameBaseline",
        epochs=args.epochs, lr=args.lr, warmup_epochs=args.warmup_epochs, model_dir=args.model_dir
    )

    gru = TemporalGRU(input_dim=384, hidden_dim=256, num_classes=5, num_layers=1, dropout=0.4)
    histories["TemporalGRU"] = train_model(
        gru, train_loader, val_loader, "TemporalGRU",
        epochs=args.epochs, lr=args.lr, warmup_epochs=args.warmup_epochs, model_dir=args.model_dir
    )

    transformer_pe = TemporalTransformer(input_dim=384, num_heads=4, num_layers=2, ff_dim=512,
                                         num_classes=5, dropout=0.4, use_positional_encoding=True)
    histories["TemporalTransformerWithPE"] = train_model(
        transformer_pe, train_loader, val_loader, "TemporalTransformerWithPE",
        epochs=args.epochs, lr=args.lr, warmup_epochs=args.warmup_epochs, model_dir=args.model_dir
    )

    transformer_nope = TemporalTransformer(input_dim=384, num_heads=4, num_layers=2, ff_dim=512,
                                            num_classes=5, dropout=0.4, use_positional_encoding=False)
    histories["TemporalTransformerNoPE"] = train_model(
        transformer_nope, train_loader, val_loader, "TemporalTransformerNoPE",
        epochs=args.epochs, lr=args.lr, warmup_epochs=args.warmup_epochs, model_dir=args.model_dir
    )

    plot_training_curves(histories, save_dir=args.vis_dir)
