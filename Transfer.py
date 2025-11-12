# ================================================
# Unit 3 Assignment - Transfer Learning × Auto-ML
# CNN (ResNet-18) vs ViT (DeiT-Tiny)
# Dataset A = CIFAR-10 (natural images)
# Dataset B = FashionMNIST (non-natural, domain-shifted)
# ================================================

import os
os.environ["TORCH_HOME"] = "./weights"
os.environ["XDG_CACHE_HOME"] = "./weights"
os.environ["HF_HOME"] = "./weights"
import matplotlib
matplotlib.use("Agg")  # headless backend for servers
# ================================================================

import json
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import timm

from sklearn.metrics import classification_report, confusion_matrix

# ----------------- 0 -----------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


DIRS = {
    "data": "./data",
    "weights": "./weights",
    "out": "./outputs",
    "figs": "./outputs/figs",
    "ckpt": "./outputs/checkpoints",
    "logs": "./outputs/logs",
}
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)



IMG_SIZE = 224  
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transform(dataset_name, train=True, use_colorjitter=True):
    ops = []

    if dataset_name == 'B':

        ops.append(transforms.Grayscale(num_output_channels=3))

    if train:
      
        train_ops = [transforms.Resize((IMG_SIZE, IMG_SIZE)),
                     transforms.RandomHorizontalFlip()]
        if use_colorjitter:
            train_ops.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
        ops.extend(train_ops)
    else:
        ops.append(transforms.Resize((IMG_SIZE, IMG_SIZE)))

    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose(ops)

data_root = DIRS["data"]


base_train_A = datasets.CIFAR10(root=data_root, train=True, download=True, transform=None)
base_train_B = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=None)
test_base_A = datasets.CIFAR10(root=data_root, train=False, download=True, transform=None)
test_base_B = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=None)

NUM_CLASSES_A = 10 
NUM_CLASSES_B = 10  


np.random.seed(42)

indices_A = np.random.permutation(len(base_train_A))
train_size_A = int(0.9 * len(base_train_A))
train_idx_A = indices_A[:train_size_A]
val_idx_A = indices_A[train_size_A:]

indices_B = np.random.permutation(len(base_train_B))
train_size_B = int(0.9 * len(base_train_B))
train_idx_B = indices_B[:train_size_B]
val_idx_B = indices_B[train_size_B:]

def create_loaders(dataset_name, batch_size, use_colorjitter=True, num_workers=4):
    if dataset_name == 'A':
        train_transform = get_transform('A', train=True, use_colorjitter=use_colorjitter)
        eval_transform = get_transform('A', train=False)

        full_train_for_train = datasets.CIFAR10(root=data_root, train=True, download=False, transform=train_transform)
        full_train_for_val = datasets.CIFAR10(root=data_root, train=True, download=False, transform=eval_transform)
        full_test = datasets.CIFAR10(root=data_root, train=False, download=False, transform=eval_transform)

        train_indices = train_idx_A
        val_indices = val_idx_A
    else:  # 'B'
        train_transform = get_transform('B', train=True, use_colorjitter=use_colorjitter)
        eval_transform = get_transform('B', train=False)

        full_train_for_train = datasets.FashionMNIST(root=data_root, train=True, download=False, transform=train_transform)
        full_train_for_val = datasets.FashionMNIST(root=data_root, train=True, download=False, transform=eval_transform)
        full_test = datasets.FashionMNIST(root=data_root, train=False, download=False, transform=eval_transform)

        train_indices = train_idx_B
        val_indices = val_idx_B

    train_ds = Subset(full_train_for_train, train_indices)
    val_ds = Subset(full_train_for_val, val_indices)
    test_ds = full_test

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    print(f"[Data] Dataset {dataset_name}: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    return train_loader, val_loader, test_loader



def model_param_millions(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

def create_cnn(num_classes):
    """ResNet-18  CNN backbone"""
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    except AttributeError:
        model = models.resnet18(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    print(f"[Model] ResNet-18 params: {model_param_millions(model):.2f}M")
    return model

def create_vit(num_classes):
    """DeiT-Tiny ViT backbone"""
    model = timm.create_model("deit_tiny_patch16_224", pretrained=True, num_classes=num_classes)
    
    try:
        patch = model.patch_embed.patch_size if hasattr(model, "patch_embed") else (16, 16)
        embed = model.embed_dim if hasattr(model, "embed_dim") else 192
        layers = len(model.blocks) if hasattr(model, "blocks") else 12
        heads = model.num_heads if hasattr(model, "num_heads") else 3
        print(f"[Model] DeiT-Tiny: patch={patch}, embed_dim={embed}, layers={layers}, heads={heads}, params={model_param_millions(model):.2f}M")
    except Exception as e:
        print("[Model] DeiT-Tiny summary failed:", e)
        print(f"[Model] DeiT-Tiny params: {model_param_millions(model):.2f}M")
    return model

def set_trainable_layers(model, backbone_type, regime):
    """
    regime: 'linear', 'partial', 'full'
    """
    if regime == 'linear':
        for p in model.parameters():
            p.requires_grad = False
        if backbone_type == 'cnn':
            for p in model.fc.parameters():
                p.requires_grad = True
        elif backbone_type == 'vit':
            for p in model.head.parameters():
                p.requires_grad = True

    elif regime == 'partial':
        for p in model.parameters():
            p.requires_grad = False
        if backbone_type == 'cnn':
            for p in model.layer4.parameters():
                p.requires_grad = True
            for p in model.fc.parameters():
                p.requires_grad = True
        elif backbone_type == 'vit':
            for p in model.blocks[-1].parameters():
                p.requires_grad = True
            for p in model.head.parameters():
                p.requires_grad = True

    elif regime == 'full':
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown regime: {regime}")
    return model

# ----------------- 3 -----------------

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model, dataloader, criterion, device, return_preds=False):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_y, all_pred = [], []

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if return_preds:
            all_y.append(labels.detach().cpu().numpy())
            all_pred.append(preds.detach().cpu().numpy())

    loss_avg = running_loss / total
    acc = correct / total
    if return_preds:
        y_true = np.concatenate(all_y, axis=0)
        y_pred = np.concatenate(all_pred, axis=0)
        return loss_avg, acc, y_true, y_pred
    return loss_avg, acc

# ----------------- 4. Auto-ML ----------------

SEARCH_SPACE = {
    "lr": [1e-3, 5e-4],
    "epochs": [6, 10, 15],      
    "batch_size": [16, 32, 64], 
}

def auto_ml_for_backbone_and_dataset(backbone_type, dataset_name, num_classes, use_colorjitter=True):
   
    best_cfg = None
    best_val_acc = 0.0

    for lr in SEARCH_SPACE["lr"]:
        for epochs in SEARCH_SPACE["epochs"]:
            for batch_size in SEARCH_SPACE["batch_size"]:
                print("\n" + "=" * 70)
                print(f"[AutoML] backbone={backbone_type}, dataset={dataset_name}, lr={lr}, epochs={epochs}, batch_size={batch_size}")

                train_loader, val_loader, _ = create_loaders(dataset_name, batch_size, use_colorjitter=use_colorjitter)

           
                model = create_cnn(num_classes) if backbone_type == "cnn" else create_vit(num_classes)
                model = model.to(device)
                set_trainable_layers(model, backbone_type, regime="full")

                criterion = nn.CrossEntropyLoss()
                optimizer = optim.AdamW(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=lr, weight_decay=1e-4
                )

                best_val_acc_this_cfg = 0.0
                t0 = time.time()
                for epoch in range(epochs):
                    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
                    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                    print(f"  Epoch {epoch+1}/{epochs} | train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
                    best_val_acc_this_cfg = max(best_val_acc_this_cfg, val_acc)
                t1 = time.time()
                print(f"  --> Best val_acc for this cfg = {best_val_acc_this_cfg:.4f} | time={t1-t0:.1f}s")

                if best_val_acc_this_cfg > best_val_acc:
                    best_val_acc = best_val_acc_this_cfg
                    best_cfg = {"lr": lr, "epochs": epochs, "batch_size": batch_size}

    print("\n" + "#" * 70)
    print(f"[AutoML DONE] Best cfg for backbone={backbone_type}, dataset={dataset_name}: {best_cfg}, best_val_acc={best_val_acc:.4f}")
    return best_cfg

# ----------------- 5-----------------

def train_with_history(backbone_type, dataset_name, num_classes, regime, cfg, use_colorjitter=True):
    batch_size, epochs, lr = cfg["batch_size"], cfg["epochs"], cfg["lr"]

    train_loader, val_loader, test_loader = create_loaders(dataset_name, batch_size, use_colorjitter=use_colorjitter)
    model = create_cnn(num_classes) if backbone_type == "cnn" else create_vit(num_classes)
    model = model.to(device)
    set_trainable_layers(model, backbone_type, regime)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)

    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    print("\n" + "-" * 70)
    print(f"[Train] backbone={backbone_type}, dataset={dataset_name}, regime={regime}, cfg={cfg}")

    t0 = time.time()
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss); train_accs.append(train_acc)
        val_losses.append(val_loss);     val_accs.append(val_acc)

        print(f"  Epoch {epoch+1}/{epochs} | train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device, return_preds=True)
    t1 = time.time()
    elapsed = t1 - t0


    ckpt_path = os.path.join(DIRS["ckpt"], f"ckpt_{backbone_type}_{dataset_name}_{regime}.pth")
    torch.save(model.state_dict(), ckpt_path)


    fig1 = os.path.join(DIRS["figs"], f"val_loss_{backbone_type}_{dataset_name}_{regime}.png")
    fig2 = os.path.join(DIRS["figs"], f"val_acc_{backbone_type}_{dataset_name}_{regime}.png")

    epochs_axis = range(1, len(val_losses) + 1)
    plt.figure(figsize=(7,5)); plt.plot(epochs_axis, val_losses, label=f"{backbone_type.upper()}-{regime}")
    plt.xlabel("Epoch"); plt.ylabel("Validation Loss"); plt.title(f"Validation Loss (Dataset {dataset_name})"); plt.grid(True); plt.legend(); plt.tight_layout(); plt.savefig(fig1); plt.close()

    plt.figure(figsize=(7,5)); plt.plot(epochs_axis, val_accs, label=f"{backbone_type.upper()}-{regime}")
    plt.xlabel("Epoch"); plt.ylabel("Validation Accuracy"); plt.title(f"Validation Accuracy (Dataset {dataset_name})"); plt.grid(True); plt.legend(); plt.tight_layout(); plt.savefig(fig2); plt.close()


    cls_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    history = {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "val_losses": val_losses,
        "val_accs": val_accs,
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "cfg": cfg,
        "regime": regime,
        "backbone": backbone_type,
        "dataset": dataset_name,
        "time_sec": elapsed,
        "checkpoint": ckpt_path,
        "val_loss_fig": fig1,
        "val_acc_fig": fig2,
        "cls_report": cls_report,
        "confusion_matrix": cm.tolist(),
    }
    return model, history

# ----------------- 6 -----------------

def main():
    backbones = ["cnn", "vit"]
    datasets_ids = ["A", "B"]
    num_classes_map = {"A": NUM_CLASSES_A, "B": NUM_CLASSES_B}

    best_cfg_map = {}
    results = []       
    histories = {}   

    start_time_all = time.time()


    for backbone_type in backbones:
        for ds_id in datasets_ids:
            print("\n" + "=" * 80)
            print(f"*** Auto-ML search for backbone={backbone_type}, dataset={ds_id} ***")
            best_cfg = auto_ml_for_backbone_and_dataset(
                backbone_type, ds_id, num_classes_map[ds_id], use_colorjitter=True
            )
            best_cfg_map[(backbone_type, ds_id)] = best_cfg


    for backbone_type in backbones:
        for ds_id in datasets_ids:
            cfg = best_cfg_map[(backbone_type, ds_id)]
            for regime in ["linear", "partial", "full"]:
                _, history = train_with_history(
                    backbone_type, ds_id, num_classes_map[ds_id], regime, cfg, use_colorjitter=True
                )
                best_val_acc = float(max(history["val_accs"]))
                results.append({
                    "backbone": backbone_type,
                    "dataset": ds_id,
                    "regime": regime,
                    "cfg": cfg,
                    "best_val_acc": best_val_acc,
                    "test_acc": float(history["test_acc"]),
                    "time_sec": float(history["time_sec"]),
                })
                histories[(backbone_type, ds_id, regime)] = history

    end_time_all = time.time()
    total_seconds = end_time_all - start_time_all
    print("\nTotal running time (seconds):", total_seconds)


    print("\n" + "#" * 80)
    print("Summary of results (best validation accuracy & test accuracy):")
    for r in results:
        print(f"Backbone={r['backbone']:>3}, Dataset={r['dataset']}, Regime={r['regime']:<7} | "
              f"ValAcc(max)={r['best_val_acc']:.4f}, TestAcc={r['test_acc']:.4f} | "
              f"Config={r['cfg']} | time={r['time_sec']:.1f}s")

    winners = {}
    for backbone_type in backbones:
        for ds_id in datasets_ids:
            subset = [r for r in results if r["backbone"] == backbone_type and r["dataset"] == ds_id]
            best_r = max(subset, key=lambda x: x["best_val_acc"])
            winners[(backbone_type, ds_id)] = best_r
            print(f"\n[Winner] Backbone={backbone_type}, Dataset={ds_id} -> Regime={best_r['regime']}, ValAcc={best_r['best_val_acc']:.4f}, TestAcc={best_r['test_acc']:.4f}, cfg={best_r['cfg']}")


    out_json = os.path.join(DIRS["out"], "results.json")
    with open(out_json, "w") as f:
        json.dump({
            "results": results,
            "best_cfg_map": {f"{k[0]}_{k[1]}": v for k, v in best_cfg_map.items()},
            "total_seconds": total_seconds
        }, f, indent=2)
    print(f"\nSaved results to {out_json}")


    summary_md = os.path.join(DIRS["out"], "summary.md")
    with open(summary_md, "w") as f:
        f.write("# Transfer Learning × Auto-ML Study (CNN vs ViT, A vs B)\n\n")
        f.write(f"- **Total Run Time**: {total_seconds/60:.1f} minutes\n")
        f.write(f"- **Hardware**: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
        f.write("\n---\n\n")

        f.write("## 1) Model Descriptions\n")
        f.write("- **CNN**: ResNet-18 (ImageNet-1K supervised pretraining)\n")
        f.write("- **ViT**: DeiT-Tiny (patch=16×16, embed_dim=192, layers=12, heads=3, ImageNet-1K supervised)\n")
        f.write("\n---\n\n")

        f.write("## 2) Experimental Settings\n")
        f.write("- Dataset A: CIFAR-10 (natural images); Dataset B: Fashion-MNIST (grayscale → 3-ch)\n")
        f.write("- Splits: train 90%, val 10% (from train), test 100% of official test\n")
        f.write("- Augmentations: Resize(224), RandomHorizontalFlip, ColorJitter(0.2)\n")
        f.write(f"- Auto-ML Search Space: lr={SEARCH_SPACE['lr']}, epochs={SEARCH_SPACE['epochs']}, batch_size={SEARCH_SPACE['batch_size']}\n")
        f.write("- Regimes: Linear (freeze all, head only), Partial (last block + head), Full (all layers)\n")
        f.write("- Loss: Cross-Entropy; Optimizer: AdamW (wd=1e-4)\n")
        f.write("\n---\n\n")


        f.write("## 3) Results Overview\n")
        df = pd.DataFrame(results)
        f.write(df.sort_values(by=["dataset","backbone","regime"]).to_markdown(index=False))
        f.write("\n\n")

        # Winners
        f.write("### Winners per (Backbone, Dataset)\n")
        for k, v in winners.items():
            f.write(f"- **{k[0].upper()} on {k[1]}** → Regime: **{v['regime']}**, Best ValAcc: **{v['best_val_acc']:.4f}**, TestAcc: **{v['test_acc']:.4f}**, cfg={v['cfg']}\n")
        f.write("\n---\n\n")


        f.write("## 4) Training Curves (Validation)\n")
        for ds_id in datasets_ids:
            for backbone_type in backbones:
                win = winners[(backbone_type, ds_id)]
                hist = histories[(backbone_type, ds_id, win["regime"])]
                f.write(f"### {backbone_type.upper()} on {ds_id} (Regime: {win['regime']})\n")
                f.write(f"- Loss curve: ![]({os.path.relpath(hist['val_loss_fig'], start=DIRS['out'])})\n")
                f.write(f"- Acc curve: ![]({os.path.relpath(hist['val_acc_fig'], start=DIRS['out'])})\n\n")


        f.write("\n## 5) Per-Class Metrics & Confusion Matrices (Winners)\n")
        for ds_id in datasets_ids:
            for backbone_type in backbones:
                win = winners[(backbone_type, ds_id)]
                hist = histories[(backbone_type, ds_id, win["regime"])]
                f.write(f"### {backbone_type.upper()} on {ds_id} (Regime: {win['regime']})\n")
                # classification report
                rep_df = pd.DataFrame(hist["cls_report"]).T
                f.write(rep_df.to_markdown())
                f.write("\n\n")
                # confusion matrix
                cm_df = pd.DataFrame(hist["confusion_matrix"])
                f.write("Confusion Matrix:\n\n")
                f.write(cm_df.to_markdown(index=False, headers=False))
                f.write("\n\n")


        f.write("## 6) Analysis (Guided)\n")
        f.write("- **Domain Similarity**: Expect CNN to do better on Dataset A (closer to ImageNet), while ViT may be competitive/robust on Dataset B.\n")
        f.write("- **Freezing Depth**: Typically Full > Partial > Linear; discuss exceptions if any.\n")
        f.write("- **Auto-ML Trends**: Note preferred LR/epochs/batch patterns per model & dataset.\n")
        f.write("- **Runtime**: Compare wall-clock time across models; CNNs often train faster.\n")
        f.write("\n---\n\n")

        f.write("## 7) Reproducibility\n")
        f.write("- Fixed split (seed=42), deterministic seeds set for torch/np/random.\n")
        f.write("- Report includes configs & checkpoints for winners.\n")

    print(f"Saved summary report to {summary_md}")
    print("\nAll done. Use the printed tables & curves and the markdown report to write your 2–3 page submission.")

if __name__ == "__main__":
    main()
