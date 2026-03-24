#!/usr/bin/env python3
"""Convert CRTrack pseudo labels into the standalone ReID project dataset layout."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple



VIEW_COLUMNS = (("view1", 1), ("view2", 2), ("view3", 3))


@dataclass(frozen=True)
class CropRecord:
    src_path: Path
    scene: str
    clip: str
    frame_id: int
    camid: int
    local_id: int
    global_pid: int
    source_global_id: int
    text: str
    bbox: Tuple[int, int, int, int]


@dataclass(frozen=True)
class SavedImage:
    relative_path: str
    scene: str
    clip: str
    frame_id: int
    camid: int
    local_id: int
    global_pid: int
    source_global_id: int
    text: str
    split: str
    bbox: Tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the standalone ReID project dataset from CRTrack_In-domain_gt. "
            "Identity labels are read strictly from ids_with_text_cross_view_selected."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("../data/CRTrack_my1/CRTrack_In-domain"),
        help="CRTrack_In-domain_gt root directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("../data/CRTrack-ReID"),
        help="Output directory in the ReID project layout.",
    )
    parser.add_argument(
        "--selected-root",
        type=Path,
        default=None,
        help="Directory containing ids_with_text_cross_view_selected CSV files. Defaults to <source-root>/ids_with_text_cross_view_selected.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Discard bbox entries with score below this threshold.",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=4,
        help="Discard crops whose clipped width or height is smaller than this value.",
    )
    parser.add_argument(
        "--split-mode",
        choices=("train_only", "train_query_gallery"),
        default="train_query_gallery",
        help=(
            "How to populate the ReID dataset directory. train_query_gallery keeps one cross-view sample for query and one for gallery per pid when possible; "
            "train_only puts every crop into train and leaves empty query/test folders."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Remove the existing output directory before writing the converted dataset.",
    )
    return parser.parse_args()




def open_image(image_path: Path):
    from PIL import Image

    return Image.open(image_path)


def read_selected_rows(csv_path: Path) -> List[dict]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def collect_identity_maps(selected_root: Path) -> Tuple[Dict[Tuple[str, str, int], int], Dict[int, dict]]:
    pid_lookup: Dict[Tuple[str, str, int], int] = {}
    pid_metadata: Dict[int, dict] = {}
    next_pid = 0

    csv_paths = sorted(selected_root.glob("*/*/*_id_match_texts.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No selected CSV files found under {selected_root}")

    for csv_path in csv_paths:
        scene = csv_path.parent.parent.name
        clip = csv_path.parent.name
        for row in read_selected_rows(csv_path):
            source_global_id = int(row["id"])
            key = (scene, clip, source_global_id)
            if key in pid_lookup:
                raise ValueError(f"Duplicate identity key detected: {key}")
            pid_lookup[key] = next_pid
            pid_metadata[next_pid] = {
                "scene": scene,
                "clip": clip,
                "source_global_id": source_global_id,
                "text": row.get("text", "").strip(),
                "views": {column: int(row[column]) for column, _ in VIEW_COLUMNS},
            }
            next_pid += 1

    return pid_lookup, pid_metadata


def parse_bbox_file(bbox_path: Path, score_threshold: float) -> Dict[Tuple[int, int], Tuple[int, int, int, int]]:
    records: Dict[Tuple[int, int], Tuple[int, int, int, int]] = {}
    with bbox_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 7:
                raise ValueError(f"Unexpected bbox format in {bbox_path}:{line_number}: {line}")
            frame_id = int(float(parts[0]))
            local_id = int(float(parts[1]))
            x = int(math.floor(float(parts[2])))
            y = int(math.floor(float(parts[3])))
            w = int(math.ceil(float(parts[4])))
            h = int(math.ceil(float(parts[5])))
            score = float(parts[6])
            if score < score_threshold:
                continue
            records[(frame_id, local_id)] = (x, y, w, h)
    return records


def clip_bbox(box: Tuple[int, int, int, int], width: int, height: int, min_size: int) -> Tuple[int, int, int, int] | None:
    x, y, w, h = box
    left = max(0, x)
    top = max(0, y)
    right = min(width, x + max(w, 0))
    bottom = min(height, y + max(h, 0))
    if right - left < min_size or bottom - top < min_size:
        return None
    return left, top, right, bottom


def iter_clip_records(
    source_root: Path,
    selected_root: Path,
    pid_lookup: Dict[Tuple[str, str, int], int],
    pid_metadata: Dict[int, dict],
    score_threshold: float,
    min_size: int,
) -> List[CropRecord]:
    image_root = source_root / "images" / "train"
    clip_records: List[CropRecord] = []

    csv_paths = sorted(selected_root.glob("*/*/*_id_match_texts.csv"))
    for csv_path in csv_paths:
        scene = csv_path.parent.parent.name
        clip = csv_path.parent.name
        clip_dir = image_root / scene / clip
        bbox_dir = clip_dir / "bbox"
        if not clip_dir.exists() or not bbox_dir.exists():
            continue

        rows = read_selected_rows(csv_path)
        local_to_pid_by_cam: Dict[int, Dict[int, Tuple[int, int, str]]] = defaultdict(dict)
        for row in rows:
            source_global_id = int(row["id"])
            global_pid = pid_lookup[(scene, clip, source_global_id)]
            text = pid_metadata[global_pid]["text"]
            for column, camid in VIEW_COLUMNS:
                local_id = int(row[column])
                local_to_pid_by_cam[camid][local_id] = (global_pid, source_global_id, text)

        for _, camid in VIEW_COLUMNS:
            view_name = f"{scene}_View{camid}"
            image_dir = clip_dir / view_name
            bbox_path = bbox_dir / f"{view_name}_gt.txt"
            if not image_dir.exists() or not bbox_path.exists():
                continue

            bbox_records = parse_bbox_file(bbox_path, score_threshold=score_threshold)
            local_to_pid = local_to_pid_by_cam.get(camid, {})
            if not local_to_pid:
                continue

            for image_path in sorted(image_dir.glob("*.jpg")):
                frame_id = int(image_path.stem.rsplit("_", 1)[-1])
                with open_image(image_path) as image:
                    for local_id, (global_pid, source_global_id, text) in local_to_pid.items():
                        bbox = bbox_records.get((frame_id, local_id))
                        if bbox is None:
                            continue
                        clipped = clip_bbox(bbox, image.width, image.height, min_size=min_size)
                        if clipped is None:
                            continue
                        clip_records.append(
                            CropRecord(
                                src_path=image_path,
                                scene=scene,
                                clip=clip,
                                frame_id=frame_id,
                                camid=camid,
                                local_id=local_id,
                                global_pid=global_pid,
                                source_global_id=source_global_id,
                                text=text,
                                bbox=clipped,
                            )
                        )
    return clip_records


def choose_split(records: Sequence[CropRecord], split_mode: str) -> Dict[int, List[str]]:
    if split_mode == "train_only":
        return {index: ["train"] for index in range(len(records))}

    per_pid: Dict[int, List[int]] = defaultdict(list)
    for index, record in enumerate(records):
        per_pid[record.global_pid].append(index)

    assignments: Dict[int, List[str]] = {index: ["train"] for index in range(len(records))}
    for indices in per_pid.values():
        grouped_by_cam: Dict[int, List[int]] = defaultdict(list)
        for index in indices:
            grouped_by_cam[records[index].camid].append(index)
        ordered_cams = sorted(grouped_by_cam)
        if len(ordered_cams) < 2:
            continue

        query_index = sorted(
            grouped_by_cam[ordered_cams[0]],
            key=lambda idx: (records[idx].frame_id, records[idx].clip, records[idx].scene),
        )[0]
        gallery_index = sorted(
            grouped_by_cam[ordered_cams[1]],
            key=lambda idx: (records[idx].frame_id, records[idx].clip, records[idx].scene),
        )[0]

        assignments[query_index] = ["query"]
        assignments[gallery_index] = ["test"]

    return assignments


def write_dataset(records: Sequence[CropRecord], assignments: Dict[int, List[str]], output_root: Path) -> List[SavedImage]:
    for split_name in ("train", "query", "test"):
        (output_root / split_name).mkdir(parents=True, exist_ok=True)

    saved: List[SavedImage] = []
    cnt = 0
    for index, record in enumerate(records):
        filename = (
            f"{record.global_pid:06d}_c{record.camid}_"
            f"{record.scene}_{record.clip}_f{record.frame_id:06d}_lid{record.local_id:04d}.jpg"
        )
        for split_name in assignments[index]:
            destination = output_root / split_name / filename
            with open_image(record.src_path) as image:
                crop = image.crop(record.bbox)
                crop.save(destination, quality=95)
            saved.append(
                SavedImage(
                    relative_path=str(Path(split_name) / filename),
                    scene=record.scene,
                    clip=record.clip,
                    frame_id=record.frame_id,
                    camid=record.camid,
                    local_id=record.local_id,
                    global_pid=record.global_pid,
                    source_global_id=record.source_global_id,
                    text=record.text,
                    split=split_name,
                    bbox=record.bbox,
                )
            )

        cnt += 1
        if cnt % 100 == 0:
            print(f"{cnt}/{len(records)}")

    return saved


def write_metadata(output_root: Path, pid_metadata: Dict[int, dict], saved: Sequence[SavedImage]) -> None:
    metadata_path = output_root / "metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "relative_path",
                "split",
                "global_pid",
                "source_global_id",
                "camid",
                "scene",
                "clip",
                "frame_id",
                "local_id",
                "bbox_left",
                "bbox_top",
                "bbox_right",
                "bbox_bottom",
                "text",
            ]
        )
        for item in saved:
            left, top, right, bottom = item.bbox
            writer.writerow(
                [
                    item.relative_path,
                    item.split,
                    item.global_pid,
                    item.source_global_id,
                    item.camid,
                    item.scene,
                    item.clip,
                    item.frame_id,
                    item.local_id,
                    left,
                    top,
                    right,
                    bottom,
                    item.text,
                ]
            )

    summary = {
        "num_pids": len(pid_metadata),
        "num_images": len(saved),
        "split_counts": dict(Counter(item.split for item in saved)),
        "camera_counts": dict(Counter(item.camid for item in saved)),
        "scene_counts": dict(Counter(item.scene for item in saved)),
        "identities": pid_metadata,
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    selected_root = (args.selected_root or (source_root / "ids_with_text_cross_view_selected")).resolve()
    output_root = args.output_root.resolve()

    if not source_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")
    if not selected_root.exists():
        raise FileNotFoundError(f"Selected label directory does not exist: {selected_root}")

    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    pid_lookup, pid_metadata = collect_identity_maps(selected_root)
    records = iter_clip_records(
        source_root=source_root,
        selected_root=selected_root,
        pid_lookup=pid_lookup,
        pid_metadata=pid_metadata,
        score_threshold=args.score_threshold,
        min_size=args.min_size,
    )
    assignments = choose_split(records, split_mode=args.split_mode)
    saved = write_dataset(records, assignments, output_root)
    write_metadata(output_root, pid_metadata, saved)

    split_counts = Counter(item.split for item in saved)
    print(f"source_root: {source_root}")
    print(f"selected_root: {selected_root}")
    print(f"output_root: {output_root}")
    print(f"num_identities: {len(pid_metadata)}")
    print(f"num_records: {len(records)}")
    print(f"saved_images: {len(saved)}")
    print(f"split_counts: {dict(split_counts)}")


if __name__ == "__main__":
    main()