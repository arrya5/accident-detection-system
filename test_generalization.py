"""
External Dataset Generalization Test
Downloads a brand new Kaggle dataset and tests the model on it.
The model has NEVER seen these images — true generalization test.
"""
import kagglehub
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pathlib import Path
import os

# ── download dataset ────────────────────────────────────────────────────────
print("=" * 60)
print("DOWNLOADING NEW DATASET FROM KAGGLE...")
print("(Model has never seen these images before)")
print("=" * 60)

path = kagglehub.dataset_download("asefjamilajwad/car-crash-dataset-ccd")
print(f"\nDataset downloaded to: {path}")

# ── explore structure ───────────────────────────────────────────────────────
print("\n[DATASET STRUCTURE]")
dataset_path = Path(path)
all_items = []
for root, dirs, files in os.walk(dataset_path):
    level = root.replace(str(dataset_path), '').count(os.sep)
    indent = '  ' * level
    folder_name = os.path.basename(root)
    img_count = sum(1 for f in files if f.lower().endswith(('.jpg','.jpeg','.png','.bmp')))
    if img_count > 0 or level <= 2:
        print(f"{indent}{folder_name}/ ({img_count} images)")
    for f in files[:3]:
        if f.lower().endswith(('.jpg','.jpeg','.png','.bmp')):
            all_items.append(Path(root) / f)

# ── collect all images with labels ─────────────────────────────────────────
print("\n[COLLECTING IMAGES]")
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

accident_images = []
normal_images   = []

for root, dirs, files in os.walk(dataset_path):
    folder = os.path.basename(root).lower()
    for f in files:
        if Path(f).suffix.lower() not in IMAGE_EXTS:
            continue
        fpath = Path(root) / f
        # classify folder by name
        if any(k in folder for k in ['accident', 'crash', 'collision', 'incident']):
            accident_images.append(fpath)
        elif any(k in folder for k in ['normal', 'non', 'noaccident', 'no_accident',
                                        'nonaccident', 'safe', 'regular']):
            normal_images.append(fpath)

print(f"  Accident images found:     {len(accident_images)}")
print(f"  Non-accident images found: {len(normal_images)}")

# If folder structure unclear, list top-level folders
if len(accident_images) == 0 and len(normal_images) == 0:
    print("\n  Could not auto-detect. Listing all folders:")
    for root, dirs, files in os.walk(dataset_path):
        level = root.replace(str(dataset_path), '').count(os.sep)
        if level <= 3:
            imgs = sum(1 for f in files if Path(f).suffix.lower() in IMAGE_EXTS)
            print(f"    {root}  [{imgs} images]")

# ── load model ──────────────────────────────────────────────────────────────
class AccidentDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights=None)
        nf = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(nf, 512),  nn.BatchNorm1d(512),  nn.ReLU(),
            nn.Dropout(0.4), nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.classifier(self.backbone(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = AccidentDetector().to(device)
ckpt   = torch.load("models/accident_detector_best.pth", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(img_path):
    try:
        img    = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out          = model(tensor)
            prob_acc     = 1 - torch.sigmoid(out).item()
        return prob_acc
    except:
        return None

# ── run inference ───────────────────────────────────────────────────────────
if len(accident_images) > 0 or len(normal_images) > 0:
    print("\n[RUNNING INFERENCE ON NEW DATASET]")

    # cap at 200 each to keep it fast
    acc_sample  = accident_images[:200]
    norm_sample = normal_images[:200]

    acc_correct,  acc_total  = 0, 0
    norm_correct, norm_total = 0, 0

    print(f"  Testing {len(acc_sample)} accident images...")
    for p in acc_sample:
        prob = predict_image(p)
        if prob is None: continue
        acc_total += 1
        if prob >= 0.5:
            acc_correct += 1

    print(f"  Testing {len(norm_sample)} non-accident images...")
    for p in norm_sample:
        prob = predict_image(p)
        if prob is None: continue
        norm_total += 1
        if prob < 0.5:
            norm_correct += 1

    total   = acc_total + norm_total
    correct = acc_correct + norm_correct
    accuracy = correct / total * 100 if total > 0 else 0

    tp = acc_correct
    fn = acc_total - acc_correct
    tn = norm_correct
    fp = norm_total - norm_correct
    precision   = tp/(tp+fp)*100 if (tp+fp) > 0 else 0
    recall      = tp/(tp+fn)*100 if (tp+fn) > 0 else 0
    f1          = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0

    print("\n" + "=" * 60)
    print("GENERALIZATION TEST RESULTS")
    print("(Completely new dataset — never used in training)")
    print("=" * 60)
    print(f"  Accident images:     {acc_correct}/{acc_total} correct  ({acc_correct/acc_total*100:.1f}% recall)")
    print(f"  Non-accident images: {norm_correct}/{norm_total} correct  ({norm_correct/norm_total*100:.1f}% specificity)")
    print(f"")
    print(f"  Overall Accuracy:  {accuracy:.2f}%")
    print(f"  Precision:         {precision:.2f}%")
    print(f"  Recall:            {recall:.2f}%")
    print(f"  F1-Score:          {f1:.2f}%")
    print(f"  False Positives:   {fp}")
    print(f"  False Negatives:   {fn}")
    print("=" * 60)
    if accuracy >= 90:
        print("  VERDICT: Model generalizes WELL to unseen data.")
    elif accuracy >= 75:
        print("  VERDICT: Model generalizes MODERATELY to unseen data.")
    else:
        print("  VERDICT: Model struggles on this new dataset.")
    print("=" * 60)
