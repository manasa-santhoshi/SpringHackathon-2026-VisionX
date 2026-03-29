"""
Evaluation script for MPED-RNN anomaly detection on CHAD.

Computes per-sequence anomaly scores on the test set and evaluates using:
- AUC-ROC (Receiver Operating Characteristic)
- AUC-PR (Precision-Recall)
- EER (Equal Error Rate)

Saves per-video and per-camera results to JSON for dashboard visualization.

Usage:
    python -m src.anomaly.evaluate --model-dir models/anomaly/cam_1_2_3_4
    python -m src.anomaly.evaluate --model-dir models/anomaly/cam_1_2_3_4 --cameras 1 3
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from torch.utils.data import DataLoader

from src.anomaly.data import CHADSkeletonDataset
from src.anomaly.model import MPEDRNN

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute Equal Error Rate."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    try:
        eer = brentq(lambda x: interp1d(fpr, fnr)(x) - x, 0.0, 1.0)
    except ValueError:
        eer = 0.5
    return float(eer)


def evaluate(
    model_dir: str,
    data_root: str | None = None,
    cameras: list[int] | None = None,
    batch_size: int = 512,
    device: str | None = None,
) -> dict:
    """Evaluate trained MPED-RNN on CHAD test set."""
    model_dir = Path(model_dir)

    # Load config
    with open(model_dir / "config.json") as f:
        config = json.load(f)

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if data_root is None:
        data_root = PROJECT_ROOT / "data" / "raw" / "CHAD" / "CHAD_Meta"

    eval_cameras = cameras or config["cameras"]

    # Load model
    model = MPEDRNN(
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        pred_len=config["pred_len"],
    ).to(device)
    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))
    model.eval()

    # Load test data
    print("Loading CHAD test data...")
    test_dataset = CHADSkeletonDataset(
        data_root=data_root,
        split="test",
        split_num=1,
        cameras=eval_cameras,
        seq_len=config["seq_len"],
        pred_len=config["pred_len"],
        stride=1,
    )
    print(f"Test sequences: {len(test_dataset)}")

    if len(test_dataset) == 0:
        raise RuntimeError("No test data found.")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Compute anomaly scores
    all_scores = []
    all_labels = []
    all_cameras = []
    all_videos = []

    print("Computing anomaly scores...")
    with torch.no_grad():
        for batch in test_loader:
            x = batch["input"].to(device)
            target = batch["target"].to(device)

            scores = model.anomaly_score(x, target)

            all_scores.extend(scores.cpu().numpy().tolist())
            all_labels.extend(batch["label"])
            all_cameras.extend(batch["camera_id"])
            all_videos.extend(batch["video_id"])

    scores_arr = np.array(all_scores)
    labels_arr = np.array(all_labels)
    cameras_arr = np.array(all_cameras)

    # Overall metrics
    fpr, tpr, _ = roc_curve(labels_arr, scores_arr)
    auc_roc = float(auc(fpr, tpr))

    precision, recall, _ = precision_recall_curve(labels_arr, scores_arr)
    auc_pr = float(auc(recall, precision))

    eer = compute_eer(labels_arr, scores_arr)

    results = {
        "overall": {
            "auc_roc": round(auc_roc, 4),
            "auc_pr": round(auc_pr, 4),
            "eer": round(eer, 4),
            "num_sequences": len(scores_arr),
            "num_anomalous": int(labels_arr.sum()),
            "num_normal": int((1 - labels_arr).sum()),
        },
        "per_camera": {},
    }

    print(f"\nOverall: AUC-ROC={auc_roc:.4f}, AUC-PR={auc_pr:.4f}, EER={eer:.4f}")

    # Per-camera metrics
    for cam_id in sorted(set(cameras_arr)):
        mask = cameras_arr == cam_id
        cam_scores = scores_arr[mask]
        cam_labels = labels_arr[mask]

        if cam_labels.sum() == 0 or (1 - cam_labels).sum() == 0:
            continue

        cam_fpr, cam_tpr, _ = roc_curve(cam_labels, cam_scores)
        cam_auc_roc = float(auc(cam_fpr, cam_tpr))
        cam_precision, cam_recall, _ = precision_recall_curve(cam_labels, cam_scores)
        cam_auc_pr = float(auc(cam_recall, cam_precision))
        cam_eer = compute_eer(cam_labels, cam_scores)

        results["per_camera"][str(cam_id)] = {
            "auc_roc": round(cam_auc_roc, 4),
            "auc_pr": round(cam_auc_pr, 4),
            "eer": round(cam_eer, 4),
            "num_sequences": int(mask.sum()),
        }
        print(f"Camera {cam_id}: AUC-ROC={cam_auc_roc:.4f}, AUC-PR={cam_auc_pr:.4f}, EER={cam_eer:.4f}")

    # Per-video anomaly scores (for dashboard timeline)
    video_scores = defaultdict(lambda: {"scores": [], "labels": [], "camera_id": None})
    for score, label, vid, cam in zip(all_scores, all_labels, all_videos, all_cameras):
        video_scores[vid]["scores"].append(score)
        video_scores[vid]["labels"].append(int(label))
        video_scores[vid]["camera_id"] = int(cam)

    video_results = {}
    for vid, data in video_scores.items():
        scores_list = data["scores"]
        video_results[vid] = {
            "camera_id": data["camera_id"],
            "mean_score": round(float(np.mean(scores_list)), 6),
            "max_score": round(float(np.max(scores_list)), 6),
            "has_anomaly": any(l == 1 for l in data["labels"]),
            "num_sequences": len(scores_list),
            "score_timeline": [round(s, 6) for s in scores_list[::max(1, len(scores_list) // 200)]],
        }

    results["per_video"] = video_results

    # Compute threshold at EER for dashboard alerts
    fpr_arr, tpr_arr, thresholds = roc_curve(labels_arr, scores_arr)
    fnr_arr = 1 - tpr_arr
    eer_idx = np.argmin(np.abs(fpr_arr - fnr_arr))
    results["threshold"] = round(float(thresholds[eer_idx]), 6) if eer_idx < len(thresholds) else round(float(np.median(scores_arr)), 6)

    # Save results
    output_path = model_dir / "evaluation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate MPED-RNN on CHAD")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to trained model directory",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Path to CHAD_Meta directory",
    )
    parser.add_argument(
        "--cameras",
        type=int,
        nargs="+",
        default=None,
        help="Camera IDs to evaluate (default: same as training)",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()
    evaluate(
        model_dir=args.model_dir,
        data_root=args.data_root,
        cameras=args.cameras,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
