import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import config
from src.dataset import LandmarkCSVDataset
from src.model import build_model

def collate_landmarks(batch):
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "target_vectors": torch.stack([item["target_vectors"] for item in batch]),
        "image_path": [item["image_path"] for item in batch],
        "width": torch.tensor([item["width"] for item in batch], dtype=torch.float32),
        "height": torch.tensor([item["height"] for item in batch], dtype=torch.float32),
    }

def split_dataframe(df, val_ratio, seed):
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_size = int(len(df_shuffled) * val_ratio)
    val_df = df_shuffled.iloc[:val_size]
    train_df = df_shuffled.iloc[val_size:]
    return train_df, val_df

def run_epoch(model, loader, criterion, optimizer, device, train=True, scaler=None):
    model.train() if train else model.eval()
    total = 0.0
    vector_criterion = nn.MSELoss()
    
    desc = "Train" if train else "Valid"
    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    
    with torch.set_grad_enabled(train):
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            target_vectors = batch["target_vectors"].to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                preds, pred_vectors = model(images)
                loss_hm = criterion(preds, targets)
                loss_vec = vector_criterion(pred_vectors, target_vectors)
                
                # Multi-task Loss Scaling (Heatmap MSE is tiny because most pixels are 0)
                is_hrnet = isinstance(criterion, nn.MSELoss)
                scaled_loss_hm = loss_hm * 1000.0 if is_hrnet else loss_hm
                scaled_loss_vec = loss_vec * 0.1
                loss = scaled_loss_hm + scaled_loss_vec
                
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                    
            total += loss.item() * images.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
    return total / max(1, len(loader.dataset))

def main():
    parser = argparse.ArgumentParser(description="Train landmark regression model (CLEAN V3).")
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata.csv"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--backbone", choices=["resnet18", "simple_cnn", "hrnet"], default="hrnet")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints/perfect_model.pt"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    if not args.metadata.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {args.metadata}")

    # Load and filter data
    df = pd.read_csv(args.metadata)
    initial_len = len(df)
    
    # 1. 제거: 오류 데이터
    df = df[~df["keypoints"].str.contains(r"\[0\.0,\s*0\.0\]", regex=True)]
    
    # 2. 제거: 해부학적 오류 및 정면 필터 적용
    valid_indices = []
    for idx, row in df.iterrows():
        kps = torch.tensor(json.loads(row["keypoints"])).numpy()
        # Frontal face filter (left_eye x > right_eye x)
        if not (kps[0, 0] - kps[1, 0]) > 0.005:
            continue
        # Shoulders below nose
        nose_y = kps[2, 1]
        ls_y = kps[4, 1]
        rs_y = kps[5, 1]
        if min(ls_y, rs_y) < nose_y:
            continue
        # Extreme y-difference
        if abs(ls_y - rs_y) > 0.3:
            continue
        valid_indices.append(idx)
    
    df = df.loc[valid_indices]
    print(f"[INFO] Dataset filtered from {initial_len} to {len(df)} clean/anatomical images.")

    if len(df) < 2:
        raise ValueError("Not enough data to train.")
        
    train_df, val_df = split_dataframe(df, args.val_ratio, args.seed)
    names = json.loads(df.iloc[0]["keypoint_names"])

    # Output type
    output_type = "heatmap" if args.backbone == "hrnet" else "coords"

    # Datasets
    train_ds = LandmarkCSVDataset(args.metadata, image_size=args.image_size, train=True, dataframe=train_df, output_type=output_type)
    val_ds = LandmarkCSVDataset(args.metadata, image_size=args.image_size, train=False, dataframe=val_df, output_type=output_type)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_landmarks, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_landmarks, pin_memory=True)

    # Model, Loss, Optimizer
    model = build_model(num_points=len(names), backbone=args.backbone, pretrained=args.pretrained).to(device)
    criterion = nn.MSELoss() if args.backbone == "hrnet" else nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    best_val = float("inf")
    start_epoch = 1

    # Resume from checkpoint
    latest_ckpt_path = args.checkpoint.parent / f"{args.checkpoint.stem}_latest{args.checkpoint.suffix}"
    resume_path = latest_ckpt_path if latest_ckpt_path.exists() else args.checkpoint

    if resume_path.exists():
        print(f"\n[복구] 기존 가중치 파일({resume_path.name})을 불러옵니다.")
        checkpoint = torch.load(resume_path, map_location=device)
        
        # [HOTFIX] 모양(Shape)이 바뀐 가중치는 자동으로 무시하고 로드하기
        state_dict = checkpoint["model_state"]
        model_state_dict = model.state_dict()
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if k in model_state_dict and v.shape != model_state_dict[k].shape:
                print(f"[WARNING] {k} 가중치는 크기가 달라 초기화됩니다. (과거: {v.shape} -> 현재: {model_state_dict[k].shape})")
                continue
            filtered_state_dict[k] = v
            
        model.load_state_dict(filtered_state_dict, strict=False)
        # if "optimizer_state" in checkpoint: optimizer.load_state_dict(checkpoint["optimizer_state"])
        # if "scheduler_state" in checkpoint: scheduler.load_state_dict(checkpoint["scheduler_state"])
        if "epoch" in checkpoint: start_epoch = checkpoint["epoch"] + 1
        if "best_val_loss" in checkpoint: best_val = checkpoint["best_val_loss"]

    print("\n[INFO] Starting training...")
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True, scaler=scaler)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        
        print(f"Epoch {epoch:03d}: train_loss={train_loss:.6f} val_loss={val_loss:.6f} lr={scheduler.get_last_lr()[0]:.6f}")
        scheduler.step()
        
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epoch": epoch,
                "backbone": args.backbone,
                "best_val_loss": best_val,
                "image_size": args.image_size,
                "keypoint_names": names,
            }, args.checkpoint)
            print(f"[*] New best model saved to {args.checkpoint.name}")

        # 항상 현재 에포크 상태를 latest로 저장하여 완벽한 이어하기 지원
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "backbone": args.backbone,
            "best_val_loss": best_val,
            "image_size": args.image_size,
            "keypoint_names": names,
        }, latest_ckpt_path)

        # 10 에포크마다 휴식/계속 여부 묻기
        if epoch % 10 == 0 and epoch < args.epochs:
            ans = input(f"\n[INFO] {epoch} 에포크가 완료되었습니다. 계속 진행하시겠습니까? (Y/N - N 입력 시 학습 일시정지): ").strip().upper()
            if ans == 'N':
                print("[INFO] 학습을 안전하게 일시 중단합니다. 나중에 다시 실행하시면 현재 위치부터 정확히 이어서 학습됩니다!")
                break

if __name__ == "__main__":
    main()
