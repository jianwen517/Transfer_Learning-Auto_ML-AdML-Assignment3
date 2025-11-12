# Transfer Learning × Auto-ML Study (CNN vs ViT, A vs B)

- **Total Run Time**: 932.6 minutes
- **Hardware**: NVIDIA A10

---

## 1) Model Descriptions
- **CNN**: ResNet-18 (ImageNet-1K supervised pretraining)
- **ViT**: DeiT-Tiny (patch=16×16, embed_dim=192, layers=12, heads=3, ImageNet-1K supervised)

---

## 2) Experimental Settings
- Dataset A: CIFAR-10 (natural images); Dataset B: Fashion-MNIST (grayscale → 3-ch)
- Splits: train 90%, val 10% (from train), test 100% of official test
- Augmentations: Resize(224), RandomHorizontalFlip, ColorJitter(0.2)
- Auto-ML Search Space: lr=[0.001, 0.0005], epochs=[6, 10, 15], batch_size=[16, 32, 64]
- Regimes: Linear (freeze all, head only), Partial (last block + head), Full (all layers)
- Loss: Cross-Entropy; Optimizer: AdamW (wd=1e-4)

---

## 3) Results Overview
