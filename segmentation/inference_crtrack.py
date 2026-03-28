"""
Inference script for SAMWISE on CRTrack.

This script runs per-view segmentation inference for each CRTrack sample and saves:
1) binary masks per frame
2) bbox json derived from predicted masks
3) quantitative metrics (CVRIDF1 / CVRMA style aggregation)
4) visualization outputs (bbox-overlaid images + videos)
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

import opts
import util.misc as utils
from datasets.crtrack import build as build_crtrack_dataset
from models.samwise import build_samwise
from util.misc import on_load_checkpoint


VIEW_NAMES = ("view1", "view2", "view3")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def mask_to_bbox(mask: np.ndarray):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [int(x1), int(y1), int(x2), int(y2)]


def save_binary_mask(mask: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    img.save(path)


def run_single_view(model, view_samples, view_targets, threshold, device):
    captions = [t["caption"] for t in view_targets]
    with torch.no_grad():
        outputs = model(view_samples.to(device), captions, view_targets)

    pred_masks = torch.cat(outputs["masks"], dim=0)  # [B*T, 1, h, w]
    bsz = view_samples.tensors.shape[0]
    num_frames = view_samples.tensors.shape[1]
    h, w = view_samples.tensors.shape[-2:]
    pred_masks = pred_masks.view(bsz, num_frames, 1, h, w)
    pred_masks = (pred_masks.sigmoid() > threshold).cpu().numpy().astype(np.uint8)
    return pred_masks


def bbox_iou_xyxy(a, b):
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1 + 1)
    ih = max(0.0, inter_y2 - inter_y1 + 1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1 + 1) * max(0.0, ay2 - ay1 + 1)
    area_b = max(0.0, bx2 - bx1 + 1) * max(0.0, by2 - by1 + 1)
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return float(inter / denom)


def update_counts(stats, pred_bbox, gt_bbox, iou_thresh):
    has_pred = pred_bbox is not None
    has_gt = gt_bbox is not None
    if has_gt:
        stats["gt"] += 1
    if has_pred:
        stats["pred"] += 1

    if has_gt and has_pred:
        iou = bbox_iou_xyxy(pred_bbox, gt_bbox)
        if iou >= iou_thresh:
            stats["tp"] += 1
        else:
            stats["fp"] += 1
            stats["fn"] += 1
    elif has_pred and (not has_gt):
        stats["fp"] += 1
    elif has_gt and (not has_pred):
        stats["fn"] += 1


def counts_to_metrics(stats):
    """CRMOT_evaluation/MOT/MOT_metrics.py aligned formulas.

    - IDF1 = 2 * IDTP / (n_gt + n_tr) * 100
    - MOTA = (1 - (fn + fp + id_switches) / n_gt) * 100
    In CRMOT eval scripts, CVRIDF1 corresponds to IDF1 and CVRMA corresponds to MOTA.
    """
    tp = stats["tp"]
    fp = stats["fp"]
    fn = stats["fn"]
    n_gt = stats["gt"]
    n_tr = stats["pred"]
    id_switches = stats.get("id_switches", 0)

    idf1 = (2.0 * tp / (n_gt + n_tr) * 100.0) if (n_gt + n_tr) > 0 else 0.0
    mota = (1.0 - (fn + fp + id_switches) / n_gt) * 100.0 if n_gt > 0 else float("-inf")

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt": n_gt,
        "pred": n_tr,
        "id_switches": id_switches,
        "IDF1": float(idf1),
        "MOTA": float(mota),
        "CVRIDF1": float(idf1),
        "CVRMA": float(mota),
    }


def find_rgb_frame(root: Path, scene: str, clip: str, view_name: str, frame_id: int):
    frame_stem = f"{int(frame_id):06d}"
    search_root = root / "images" / "train" / scene / clip / view_name
    for ext in IMAGE_EXTS:
        p = search_root / f"{frame_stem}{ext}"
        if p.exists():
            return p
    # fallback: any suffix with same stem
    matches = sorted(search_root.glob(f"{frame_stem}.*")) if search_root.exists() else []
    return matches[0] if matches else None


def draw_bbox(image_bgr, bbox, color=(0, 255, 0), label="pred"):
    if bbox is None:
        return image_bgr
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image_bgr, label, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return image_bgr


def save_video_from_frames(frame_paths, out_file: Path, fps=8):
    if len(frame_paths) == 0:
        return
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        return
    h, w = first.shape[:2]
    out_file.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_file), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for p in frame_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        writer.write(img)
    writer.release()


def build_eval_root(args):
    crtrack_path = Path(args.crtrack_path)
    return crtrack_path / "CRTrack_In-domain" if (crtrack_path / "CRTrack_In-domain").exists() else crtrack_path


def infer_crtrack(args, model):
    dataset = build_crtrack_dataset("train", args)
    sampler = torch.utils.data.SequentialSampler(dataset)
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=utils.collate_fn_crtrack_three_view,
    )

    output_dir = Path(args.output_dir)
    masks_root = output_dir / "crtrack_masks"
    bbox_root = output_dir / "crtrack_bboxes"
    viz_img_root = output_dir / "crtrack_visualization" / "images"
    viz_video_root = output_dir / "crtrack_visualization" / "videos"
    masks_root.mkdir(parents=True, exist_ok=True)
    bbox_root.mkdir(parents=True, exist_ok=True)
    viz_img_root.mkdir(parents=True, exist_ok=True)
    viz_video_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "num_samples": len(dataset),
        "threshold": args.threshold,
        "iou_threshold": args.eval_iou_thresh,
        "samples": [],
    }

    per_sequence_stats = {}
    scene_metric_bucket = {}

    eval_root = build_eval_root(args)

    max_to_process = min(len(dataset), args.max_samples) if args.max_samples > 0 else len(dataset)
    print(f"Start processing {max_to_process} samples for CRTrack inference...")

    model.eval()
    for sample_idx, (samples, targets) in enumerate(loader):
        if args.max_samples > 0 and sample_idx >= args.max_samples:
            break

        processed = sample_idx + 1
        print(f"[Sample {processed}/{max_to_process}] running 3-view inference...")
        sample_record = {"sample_idx": sample_idx, "views": {}}

        for view_idx, view_name in enumerate(VIEW_NAMES):
            view_samples = samples[view_idx]
            view_targets = targets[view_idx]

            pred_masks = run_single_view(
                model=model,
                view_samples=view_samples,
                view_targets=view_targets,
                threshold=args.threshold,
                device=args.device,
            )

            target = view_targets[0]
            scene = str(target.get("scene", "unknown_scene"))
            clip = str(target.get("clip", "unknown_clip"))
            obj_id = int(target.get("object_id", torch.tensor(-1)).item()) if hasattr(target.get("object_id", -1), "item") else int(target.get("object_id", -1))
            frame_ids = target["frames_idx"].tolist()
            gt_boxes = target["boxes"].cpu().numpy().tolist()
            gt_valid = target["valid"].cpu().numpy().tolist()

            seq_key = f"{scene}_{clip}_{view_name}_obj{obj_id}"
            if seq_key not in per_sequence_stats:
                per_sequence_stats[seq_key] = {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0, "id_switches": 0, "scene": scene}

            view_record = {
                "scene": scene,
                "clip": clip,
                "object_id": obj_id,
                "frame_ids": frame_ids,
                "frames": {},
            }

            per_frame_masks = pred_masks[0, :, 0]  # batch size fixed to 1
            rendered_frame_paths = []
            for i, frame_id in enumerate(frame_ids):
                frame_mask = per_frame_masks[i]
                pred_bbox = mask_to_bbox(frame_mask)

                gt_bbox = None
                if int(gt_valid[i]) == 1:
                    gx1, gy1, gx2, gy2 = gt_boxes[i]
                    gt_bbox = [int(gx1), int(gy1), int(gx2), int(gy2)]

                update_counts(per_sequence_stats[seq_key], pred_bbox, gt_bbox, args.eval_iou_thresh)

                mask_path = masks_root / scene / clip / view_name / f"{int(frame_id):06d}.png"
                save_binary_mask(frame_mask, mask_path)

                rgb_path = find_rgb_frame(eval_root, scene, clip, view_name, frame_id)
                vis_rel_path = None
                if rgb_path is not None:
                    vis_path = viz_img_root / scene / clip / view_name / f"{int(frame_id):06d}.jpg"
                    vis_path.parent.mkdir(parents=True, exist_ok=True)
                    img = cv2.imread(str(rgb_path))
                    if img is not None:
                        img = draw_bbox(img, pred_bbox, color=(0, 255, 0), label="pred")
                        img = draw_bbox(img, gt_bbox, color=(255, 0, 0), label="gt")
                        cv2.imwrite(str(vis_path), img)
                        rendered_frame_paths.append(vis_path)
                        vis_rel_path = str(vis_path.relative_to(output_dir))

                view_record["frames"][str(frame_id)] = {
                    "mask_path": str(mask_path.relative_to(output_dir)),
                    "bbox_xyxy": pred_bbox,
                    "gt_bbox_xyxy": gt_bbox,
                    "has_fg": bool(frame_mask.any()),
                    "vis_image_path": vis_rel_path,
                }

            bbox_file = bbox_root / scene / clip / f"{view_name}_sample{sample_idx:06d}.json"
            bbox_file.parent.mkdir(parents=True, exist_ok=True)
            with open(bbox_file, "w", encoding="utf-8") as fp:
                json.dump(view_record, fp, ensure_ascii=False, indent=2)

            video_file = viz_video_root / scene / clip / f"{view_name}_sample{sample_idx:06d}.mp4"
            save_video_from_frames(rendered_frame_paths, video_file, fps=args.viz_fps)

            view_record["bbox_json"] = str(bbox_file.relative_to(output_dir))
            view_record["vis_video_path"] = str(video_file.relative_to(output_dir)) if video_file.exists() else None
            sample_record["views"][view_name] = view_record

        summary["samples"].append(sample_record)

        print(f"[Sample {processed}/{max_to_process}] finished. Saved masks/bboxes/visualizations.")

    # sequence-level metrics
    seq_metrics = {}
    for seq_name, st in per_sequence_stats.items():
        metrics = counts_to_metrics(st)
        seq_metrics[seq_name] = metrics

        scene = st["scene"]
        if scene not in scene_metric_bucket:
            scene_metric_bucket[scene] = {"CVRIDF1": [], "CVRMA": []}
        scene_metric_bucket[scene]["CVRIDF1"].append(metrics["CVRIDF1"])
        scene_metric_bucket[scene]["CVRMA"].append(metrics["CVRMA"])

    # scene-level + all-scenes averages (CRMOT-style: average over sequences)
    scene_metrics = {}
    all_idf1 = []
    all_cvrma = []
    for scene, vals in scene_metric_bucket.items():
        scene_idf1 = float(np.mean(vals["CVRIDF1"])) if len(vals["CVRIDF1"]) else 0.0
        scene_cvrma = float(np.mean(vals["CVRMA"])) if len(vals["CVRMA"]) else 0.0
        scene_metrics[scene] = {
            "num_sequences": len(vals["CVRIDF1"]),
            "CVRIDF1": scene_idf1,
            "CVRMA": scene_cvrma,
        }
        all_idf1.extend(vals["CVRIDF1"])
        all_cvrma.extend(vals["CVRMA"])

    overall_metrics = {
        "num_sequences": len(seq_metrics),
        "CVRIDF1": float(np.mean(all_idf1)) if len(all_idf1) else 0.0,
        "CVRMA": float(np.mean(all_cvrma)) if len(all_cvrma) else 0.0,
    }

    metric_report = {
        "metric_notes": {
            "CVRIDF1": "Computed from per-sequence IDF1 and averaged over sequences.",
            "CVRMA": "Mapped from per-sequence MOTA in CRMOT_evaluation (MOTA in *_CVRMA.xlsx corresponds to CVRMA).",
            "iou_threshold": args.eval_iou_thresh,
        },
        "per_sequence": seq_metrics,
        "per_scene": scene_metrics,
        "overall": overall_metrics,
    }

    summary_file = output_dir / "crtrack_inference_summary.json"
    with open(summary_file, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    metrics_file = output_dir / "crtrack_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as fp:
        json.dump(metric_report, fp, ensure_ascii=False, indent=2)

    print(f"Saved summary to: {summary_file}")
    print(f"Saved metrics to: {metrics_file}")
    print(f"Overall CVRIDF1={overall_metrics['CVRIDF1']:.2f}, CVRMA={overall_metrics['CVRMA']:.2f}")
    print("CRTrack inference + evaluation finished.")


def main(args):
    args.batch_size = 1
    print(args)

    seed = args.seed + utils.get_rank()
    utils.init_distributed_mode(args)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.log_file = output_dir / "log.txt"
    with open(args.log_file, "w") as fp:
        fp.writelines(" ".join(sys.argv) + "\n")
        fp.writelines(str(args.__dict__) + "\n\n")

    start_time = time.time()

    model = build_samwise(args)
    device = torch.device(args.device)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if list(checkpoint["model"].keys())[0].startswith("module"):
            checkpoint["model"] = {k.replace("module.", ""): v for k, v in checkpoint["model"].items()}
        checkpoint = on_load_checkpoint(model_without_ddp, checkpoint)
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint["model"], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith("total_params") or k.endswith("total_ops"))]
        if len(missing_keys) > 0:
            print("Missing Keys: {}".format(missing_keys))
        if len(unexpected_keys) > 0:
            print("Unexpected Keys: {}".format(unexpected_keys))
    else:
        raise ValueError("Please specify --resume checkpoint for CRTrack inference.")

    print("Start CRTrack inference")
    infer_crtrack(args, model)

    total_time = time.time() - start_time
    print("Total inference time: %.4f s" % total_time)


if __name__ == "__main__":
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    parser = argparse.ArgumentParser("SAMWISE CRTrack inference script", parents=[opts.get_args_parser()])
    parser.add_argument("--max_samples", default=-1, type=int, help="Optional cap on number of CRTrack samples to process; -1 means all")
    parser.add_argument("--eval_iou_thresh", default=0.5, type=float, help="IoU threshold for bbox matching in CVRIDF1/CVRMA evaluation")
    parser.add_argument("--viz_fps", default=8, type=int, help="FPS for saved visualization videos")
    args = parser.parse_args()

    if args.max_samples == 0:
        raise ValueError("--max_samples cannot be 0")

    main(args)
