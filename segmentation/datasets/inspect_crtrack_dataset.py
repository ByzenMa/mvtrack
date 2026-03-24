import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from datasets.crtrack import CRTrackDataset
from datasets.transform_utils import make_coco_transforms


def _build_real_view_frame_and_ann(dataset, meta, view_name, frame_id):
    view_data = dataset._get_view_data(meta["view_pkls"][view_name])
    h, w = dataset._infer_hw(view_data)
    obj_id = meta["view_obj_ids"][view_name]

    decoded = dataset._decode_obj_mask(view_data, frame_id, obj_id)
    if decoded is None:
        mask = np.zeros((h, w), dtype=np.uint8)
    else:
        mask = (decoded > 0).astype(np.uint8)

    original = dataset._load_real_rgb_frame(meta, view_name, frame_id, h, w)
    if original is None:
        raise RuntimeError(
            f"Missing RGB frame: {meta['scene']}/{meta['clip']}/{view_name}, frame_id={frame_id}. "
            "This sample should have been filtered by dataset strict RGB check."
        )

    box = None
    if mask.any():
        y1, y2, x1, x2 = dataset._mask_to_box(mask)
        box = (int(x1), int(y1), int(x2), int(y2))

    return original, mask, box


def _overlay_mask(img, mask, color=(255, 0, 0), alpha=0.45):
    arr = np.asarray(img).astype(np.float32)
    m = mask.astype(bool)
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    arr[m] = arr[m] * (1 - alpha) + color_arr * alpha
    return Image.fromarray(arr.astype(np.uint8))


def _draw_bbox(img, box, color=(0, 255, 0), width=4):
    out = img.copy()
    if box is None:
        return out
    draw = ImageDraw.Draw(out)
    draw.rectangle(box, outline=color, width=width)
    return out


def _fit_text(text, max_chars=75):
    text = text.strip() if text else ""
    if not text:
        return "(empty view text)"
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _load_font(font_size=30):
    """Load a visibly large font with graceful fallback."""
    candidate_fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for font_path in candidate_fonts:
        try:
            return ImageFont.truetype(font_path, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def _annotate_text_on_image_top(img, text, font):
    """Overlay view text on top of the original image for easy visual check."""
    out = img.copy()
    draw = ImageDraw.Draw(out)

    margin = 10
    text_box = draw.textbbox((0, 0), text, font=font)
    txt_w = text_box[2] - text_box[0]
    txt_h = text_box[3] - text_box[1]

    x = margin
    y = margin
    rect_w = min(out.width - margin * 2, txt_w + 20)
    rect_h = txt_h + 16
    draw.rectangle([x - 6, y - 4, x - 6 + rect_w, y - 4 + rect_h], fill=(0, 0, 0, 170))
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return out


def _build_output_file_path(output_path, meta, frame_id):
    out = Path(output_path)
    out_dir = out.parent if out.suffix.lower() == ".png" else out

    scene = str(meta["scene"])
    clip = str(meta["clip"])
    v1 = int(meta["view_obj_ids"]["view1"])
    v2 = int(meta["view_obj_ids"]["view2"])
    v3 = int(meta["view_obj_ids"]["view3"])
    filename = f"{scene}_{clip}_{v1}_{v2}_{v3}_{int(frame_id)}.png"
    return out_dir / filename

def generate_crtrack_preview(
    crtrack_path="data/CRTrack_my1",
    num_frames=8,
    output_path="output/crtrack_preview.png",
    seed=None,
):
    if seed is not None:
        random.seed(seed)

    root = Path(crtrack_path)
    dataset_root = root / "CRTrack_In-domain" if (root / "CRTrack_In-domain").exists() else root
    dataset = CRTrackDataset(
        root=dataset_root,
        transforms=make_coco_transforms("train", max_size=1024, resize=False),
        num_frames=num_frames,
        strict_rgb_check=True,
    )

    if len(dataset) == 0:
        raise RuntimeError("CRTrack dataset is empty after strict RGB filtering.")

    idx = random.randrange(len(dataset))
    meta = dataset.metas[idx]

    col_titles = ["Original (Real RGB)", "+Mask", "+BBox"]
    view_names = dataset.VIEW_NAMES

    shared_frame_ids = meta.get("shared_frame_ids", None)
    if shared_frame_ids is None or len(shared_frame_ids) == 0:
        raise RuntimeError("Invalid sample: no shared frame ids across 3 views.")
    frame_id = random.choice(shared_frame_ids)

    rows = []
    for view_name in view_names:
        original, mask, box = _build_real_view_frame_and_ann(dataset, meta, view_name, frame_id)
        with_mask = _overlay_mask(original, mask)
        with_box = _draw_bbox(original, box)
        rows.append((view_name, frame_id, original, with_mask, with_box))

    w, h = rows[0][2].size
    gap = 20
    left_pad = 18
    top_pad = 18
    row_title_h = 28
    canvas_w = left_pad * 2 + (w * 3) + gap * 2
    canvas_h = top_pad * 2 + row_title_h + len(rows) * (h + gap)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(font_size=30)
    text_font = _load_font(font_size=34)

    y = top_pad
    for i, t in enumerate(col_titles):
        x = left_pad + i * (w + gap)
        draw.text((x + 8, y + 6), t, fill=(10, 10, 10), font=title_font)

    y += row_title_h
    for view_name, frame_id, im0, im1, im2 in rows:
        view_text = meta.get("view_texts", {}).get(view_name, "")
        txt = f"{view_name} | frame {frame_id}: {_fit_text(view_text)}"
        im0 = _annotate_text_on_image_top(im0, txt, text_font)

        for i, im in enumerate([im0, im1, im2]):
            x = left_pad + i * (w + gap)
            canvas.paste(im, (x, y))
        y += h + gap

    frame_id_for_name = rows[0][1]
    out = _build_output_file_path(output_path, meta, frame_id_for_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, format="PNG")
    return out


def main():
    parser = argparse.ArgumentParser("Inspect CRTrack dataset sample")
    parser.add_argument("--crtrack_path", default="../data/CRTrack_my1", type=str)
    parser.add_argument("--num_frames", default=8, type=int)
    parser.add_argument("--output_path", default="output/crtrack_preview.png", type=str)
    parser.add_argument("--seed", default=None, type=int)
    args = parser.parse_args()

    out = generate_crtrack_preview(
        crtrack_path=args.crtrack_path,
        num_frames=args.num_frames,
        output_path=args.output_path,
        seed=args.seed,
    )
    print(f"saved preview to: {out}")


if __name__ == "__main__":
    main()