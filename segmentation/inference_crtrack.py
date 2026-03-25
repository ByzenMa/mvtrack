"""
Inference script for SAMWISE on CRTrack.

This script runs per-view segmentation inference for each CRTrack sample and saves:
1) binary masks per frame
2) bbox json derived from predicted masks
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

import opts
import util.misc as utils
from util.misc import on_load_checkpoint
from models.samwise import build_samwise
from datasets.crtrack import build as build_crtrack_dataset


VIEW_NAMES = ("view1", "view2", "view3")


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
    masks_root.mkdir(parents=True, exist_ok=True)
    bbox_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "num_samples": len(dataset),
        "threshold": args.threshold,
        "samples": [],
    }

    model.eval()
    for sample_idx, (samples, targets) in enumerate(loader):
        if args.max_samples > 0 and sample_idx >= args.max_samples:
            break
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

            view_record = {
                "scene": scene,
                "clip": clip,
                "object_id": obj_id,
                "frame_ids": frame_ids,
                "frames": {},
            }

            # batch size fixed to 1
            per_frame_masks = pred_masks[0, :, 0]
            for i, frame_id in enumerate(frame_ids):
                frame_mask = per_frame_masks[i]
                bbox = mask_to_bbox(frame_mask)

                mask_path = masks_root / scene / clip / view_name / f"{int(frame_id):06d}.png"
                save_binary_mask(frame_mask, mask_path)

                view_record["frames"][str(frame_id)] = {
                    "mask_path": str(mask_path.relative_to(output_dir)),
                    "bbox_xyxy": bbox,
                    "has_fg": bool(frame_mask.any()),
                }

            bbox_file = bbox_root / scene / clip / f"{view_name}_sample{sample_idx:06d}.json"
            bbox_file.parent.mkdir(parents=True, exist_ok=True)
            with open(bbox_file, "w", encoding="utf-8") as fp:
                json.dump(view_record, fp, ensure_ascii=False, indent=2)

            view_record["bbox_json"] = str(bbox_file.relative_to(output_dir))
            sample_record["views"][view_name] = view_record

        summary["samples"].append(sample_record)

        if (sample_idx + 1) % 20 == 0:
            print(f"Processed {sample_idx + 1}/{len(dataset)} samples")

    summary_file = output_dir / "crtrack_inference_summary.json"
    with open(summary_file, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    print(f"Saved summary to: {summary_file}")


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
    args = parser.parse_args()

    if args.max_samples == 0:
        raise ValueError("--max_samples cannot be 0")

    main(args)
