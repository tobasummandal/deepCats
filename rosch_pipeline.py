# -*- coding: utf-8 -*-
"""
Rosch dataset pipeline (full-scale): PyTorch torchvision AlexNet (ImageNet
pretrained) with forward hook on penultimate layer (post-ReLU fc7, 4096-dim).

Scans /data/Rosch directly (source of truth), joins labels from
rosch_dataset_info.csv (best-effort -- CSV may not be exhaustive).

Parallelized via torch.utils.data.DataLoader (num_workers, pin_memory) and
GPU batching to handle ~100K images in roughly a minute. RDM / silhouette /
MDS are NOT computed here -- run run_mds.py on the saved vectors instead.

Outputs:  <out_dir>/rosch_vectors.npy   (N, 4096) float32
          <out_dir>/rosch_labels.csv    columns: path, name, category, super, basic, sub
"""

import os
import argparse
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


VALID_EXT = (".jpg", ".jpeg", ".png")


# --- A. Model setup with hook ---

def setup_model(device):
    """Load pretrained AlexNet and register hook on penultimate layer (post-ReLU fc7)."""
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    model.train(False)  # inference mode
    model.to(device)

    activations = {}

    def hook_fn(module, inp, out):
        activations["penultimate"] = out.detach()

    # classifier[5] is the ReLU after fc7 (Linear 4096->4096)
    model.classifier[5].register_forward_hook(hook_fn)
    return model, activations


# --- B. Directory scan + label join ---

def scan_directory(base_dir):
    """One level deep: base_dir/class_folder/image. Returns DataFrame."""
    rows = []
    for cls in sorted(os.listdir(base_dir)):
        cls_path = os.path.join(base_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        for fname in os.listdir(cls_path):
            if fname.startswith(".") or not fname.lower().endswith(VALID_EXT):
                continue
            rows.append({
                "path": os.path.join(cls_path, fname),
                "name": fname,
                "category": cls,
            })
    return pd.DataFrame(rows)


def parse_class_folder(cls):
    """Fallback when CSV has no entry: 'vehicle_car_sedan' -> (vehicle, car, sedan).
    Two-token folders ('bird_cardinal') -> (bird, cardinal, cardinal)."""
    parts = cls.split("_", 2)
    if len(parts) == 1:
        return parts[0], parts[0], parts[0]
    if len(parts) == 2:
        return parts[0], parts[1], parts[1]
    return parts[0], parts[1], parts[2]


def attach_labels(img_info, csv_path):
    """Left-join super/basic/sub from CSV by class_folder. Fill missing by parsing folder name."""
    if csv_path and os.path.isfile(csv_path):
        info = pd.read_csv(
            csv_path,
            usecols=["class_folder", "superordinate", "basic", "subordinate"],
        ).drop_duplicates(subset=["class_folder"])
        info = info.rename(columns={"superordinate": "super", "subordinate": "sub"})
        img_info = img_info.merge(info, left_on="category", right_on="class_folder", how="left")
        img_info = img_info.drop(columns=["class_folder"])

    need_fill = img_info["super"].isna() if "super" in img_info.columns else pd.Series([True] * len(img_info))
    if need_fill.any():
        parsed = img_info.loc[need_fill, "category"].map(parse_class_folder)
        img_info.loc[need_fill, "super"] = parsed.map(lambda t: t[0])
        img_info.loc[need_fill, "basic"] = parsed.map(lambda t: t[1])
        img_info.loc[need_fill, "sub"] = parsed.map(lambda t: t[2])
    return img_info


# --- C. Parallel Dataset / DataLoader ---

PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class ImagePathDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            return PREPROCESS(img), idx, True
        except Exception:
            return torch.zeros(3, 224, 224), idx, False


# --- D. Feature extraction (parallel) ---

def extract_features(model, activations, paths, device, batch_size=256, num_workers=8):
    """Parallel feature extraction. Returns (features, valid_mask)."""
    ds = ImagePathDataset(paths)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    n = len(paths)
    feats = np.empty((n, 4096), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)

    t0 = time.time()
    done = 0
    with torch.no_grad():
        for batch, idxs, ok in loader:
            batch = batch.to(device, non_blocking=True)
            model(batch)
            f = activations["penultimate"]
            if f.dim() > 2:
                f = f.reshape(f.shape[0], -1)
            f = f.cpu().numpy().astype(np.float32, copy=False)
            idxs_np = idxs.numpy()
            ok_np = ok.numpy().astype(bool)
            feats[idxs_np] = f
            valid[idxs_np] = ok_np
            done += batch.shape[0]
            if done % (batch_size * 20) < batch_size:
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-6)
                eta = (n - done) / max(rate, 1e-6)
                print("  %d/%d  (%.0f img/s, ETA %.1fs)" % (done, n, rate, eta))
    print("  feature extraction: %.1fs total" % (time.time() - t0))
    return feats, valid


# --- E. Export ---

def export_for_ebrw(vectors, df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "rosch_vectors.npy"), vectors)
    cols = [c for c in ["path", "name", "category", "super", "basic", "sub"] if c in df.columns]
    df[cols].to_csv(os.path.join(out_dir, "rosch_labels.csv"), index=False)
    print("Saved vectors %s and labels to %s" % (str(vectors.shape), out_dir))


# --- Main pipeline ---

def run_pipeline(image_base_dir, csv_path="rosch_dataset_info.csv",
                 batch_size=256, num_workers=8, out_dir="rosch_pipeline_out"):
    t_start = time.time()
    os.makedirs(out_dir, exist_ok=True)

    print("Scanning %s ..." % image_base_dir)
    df = scan_directory(image_base_dir)
    if df.empty:
        raise FileNotFoundError("No images found under " + image_base_dir)
    print("  found %d images across %d class folders" % (len(df), df["category"].nunique()))

    df = attach_labels(df, csv_path)
    print("  labels filled: super=%d basic=%d sub=%d unique" % (
        df["super"].nunique(), df["basic"].nunique(), df["sub"].nunique()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)
    model, activations = setup_model(device)

    print("Extracting fc7 features (batch=%d, workers=%d)..." % (batch_size, num_workers))
    vectors, valid = extract_features(
        model, activations, df["path"].tolist(),
        device=device, batch_size=batch_size, num_workers=num_workers,
    )
    n_bad = (~valid).sum()
    if n_bad:
        print("  dropping %d unreadable images" % n_bad)
        vectors = vectors[valid]
        df = df.iloc[np.where(valid)[0]].reset_index(drop=True)

    export_for_ebrw(vectors, df, out_dir=out_dir)
    print("TOTAL: %.1fs" % (time.time() - t_start))
    return {"vectors": vectors, "df": df}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rosch pipeline: scan /data/Rosch -> AlexNet fc7 features -> save for run_mds.py")
    parser.add_argument("image_base_dir", nargs="?", default="/data/Rosch",
                        help="Base directory (structure: base_dir/class_folder/image)")
    parser.add_argument("--csv", default="rosch_dataset_info.csv",
                        help="CSV with class_folder,superordinate,basic,subordinate (best-effort labels)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--out-dir", default="rosch_pipeline_out")
    args = parser.parse_args()

    run_pipeline(
        image_base_dir=args.image_base_dir,
        csv_path=args.csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        out_dir=args.out_dir,
    )
