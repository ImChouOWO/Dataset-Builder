
import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import sys
import textwrap
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Callable
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_IMAGE_DIR_NAMES = "images,img,imgs,image"
DEFAULT_LABEL_DIR_NAMES = "labels,label,txt,annotations"


@dataclass(frozen=True)
class FolderPair:
    root: Path
    images_dir: Path
    labels_dir: Path

    def display(self) -> str:
        return f"{self.images_dir}  <->  {self.labels_dir}"


@dataclass(frozen=True)
class ImageLabelItem:
    image_path: Path
    label_path: Optional[Path]
    pair_root: Path


@dataclass
class BuildConfig:
    parent_dir: Path
    output_dir: Path
    image_dir_names: List[str]
    label_dir_names: List[str]
    train_ratio: float
    val_ratio: float
    test_ratio: float
    id_to_name: Dict[int, str]
    merge_rules: Dict[int, int]
    selected_pair_indices: Optional[List[int]] = None
    operation: str = "copy"
    overwrite: bool = False
    include_unlabeled: bool = False
    seed: int = 42

    resize_enabled: bool = False
    resize_width: int = 640
    resize_height: int = 640
    resize_pad_black: bool = True

def resize_image_yolo(
    src_img: Path,
    dst_img: Path,
    target_w: int,
    target_h: int,
    pad_black: bool = True,
) -> Tuple[int, int, float, float, float, float]:
    """
    回傳:
    src_w, src_h, scale_x, scale_y, pad_x, pad_y

    pad_black=True:
        letterbox，保持比例並補黑邊
    pad_black=False:
        直接拉伸到指定大小，不補黑邊
    """
    with Image.open(src_img) as im:
        im = im.convert("RGB")
        src_w, src_h = im.size

        if pad_black:
            scale = min(target_w / src_w, target_h / src_h)
            new_w = int(round(src_w * scale))
            new_h = int(round(src_h * scale))

            resized = im.resize((new_w, new_h), Image.Resampling.LANCZOS)

            canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
            pad_x = (target_w - new_w) / 2.0
            pad_y = (target_h - new_h) / 2.0

            canvas.paste(resized, (int(round(pad_x)), int(round(pad_y))))
            canvas.save(dst_img)

            return src_w, src_h, scale, scale, pad_x, pad_y

        else:
            resized = im.resize((target_w, target_h), Image.Resampling.LANCZOS)
            resized.save(dst_img)

            scale_x = target_w / src_w
            scale_y = target_h / src_h

            return src_w, src_h, scale_x, scale_y, 0.0, 0.0


def rewrite_label_text_with_resize(
    label_text: str,
    merge_rules: Dict[int, int],
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
    scale_x: float,
    scale_y: float,
    pad_x: float,
    pad_y: float,
) -> Tuple[str, Counter]:
    output_lines: List[str] = []
    counter: Counter = Counter()

    for line_no, raw_line in enumerate(label_text.splitlines(), start=1):
        line = raw_line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 5:
            raise ValueError(f"Invalid YOLO label line {line_no}: '{raw_line}'")

        try:
            src_id = int(float(parts[0]))
            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError as exc:
            raise ValueError(f"Invalid YOLO label line {line_no}: '{raw_line}'") from exc

        dst_id = merge_rules.get(src_id, src_id)

        x_abs = x * src_w
        y_abs = y * src_h
        w_abs = w * src_w
        h_abs = h * src_h

        x_new = x_abs * scale_x + pad_x
        y_new = y_abs * scale_y + pad_y
        w_new = w_abs * scale_x
        h_new = h_abs * scale_y

        x_norm = x_new / dst_w
        y_norm = y_new / dst_h
        w_norm = w_new / dst_w
        h_norm = h_new / dst_h

        x_norm = min(max(x_norm, 0.0), 1.0)
        y_norm = min(max(y_norm, 0.0), 1.0)
        w_norm = min(max(w_norm, 0.0), 1.0)
        h_norm = min(max(h_norm, 0.0), 1.0)

        parts[0] = str(dst_id)
        parts[1] = f"{x_norm:.8f}"
        parts[2] = f"{y_norm:.8f}"
        parts[3] = f"{w_norm:.8f}"
        parts[4] = f"{h_norm:.8f}"

        counter[dst_id] += 1
        output_lines.append(" ".join(parts))

    return "\n".join(output_lines) + ("\n" if output_lines else ""), counter


def normalize_names(raw: str | Sequence[str]) -> List[str]:
    if isinstance(raw, str):
        parts = re.split(r"[,;\n\t]+", raw)
    else:
        parts = list(raw)

    names = []
    for p in parts:
        p = str(p).strip().strip("/\\")
        if p:
            names.append(p.lower())

    return sorted(set(names))


def parse_class_mapping(text: str) -> Dict[int, str]:
    text = text.strip()
    if not text:
        return {}

    try:
        obj = json.loads(text)

        if isinstance(obj, list):
            return {
                i: str(name).strip()
                for i, name in enumerate(obj)
                if str(name).strip()
            }

        if isinstance(obj, dict):
            return {
                int(k): str(v).strip()
                for k, v in obj.items()
                if str(v).strip()
            }

    except Exception:
        pass

    result: Dict[int, str] = {}
    implicit_index = 0

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        if ":" in line:
            left, right = line.split(":", 1)
            if left.strip().isdigit():
                result[int(left.strip())] = right.strip()
                continue

        parts = line.split(maxsplit=1)

        if len(parts) == 2 and parts[0].isdigit():
            result[int(parts[0])] = parts[1].strip()
        else:
            result[implicit_index] = line
            implicit_index += 1

    return {k: v for k, v in sorted(result.items()) if v}


def parse_merge_rules(text: str, id_to_name: Dict[int, str]) -> Dict[int, int]:
    name_to_id = {name: idx for idx, name in id_to_name.items()}
    rules: Dict[int, int] = {}

    if not text.strip():
        return rules

    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        if "->" not in line:
            raise ValueError(f"Merge rule line {line_no} must contain '->': {line}")

        left, right = [x.strip() for x in line.split("->", 1)]

        def token_to_id(token: str) -> int:
            token = token.strip()

            if token.isdigit():
                return int(token)

            if token in name_to_id:
                return name_to_id[token]

            raise ValueError(f"Unknown class token '{token}' in merge rule line {line_no}")

        target_id = token_to_id(right)

        for src_token in left.split(","):
            src_id = token_to_id(src_token)
            rules[src_id] = target_id

    return rules


def os_walk_safe(parent_dir: Path):
    import os

    for root, dirs, files in os.walk(parent_dir):
        dirs[:] = [
            d for d in dirs
            if d not in {".git", "__pycache__", ".venv", "venv", "dist", "build"}
        ]
        yield root, dirs, files


def scan_folder_pairs(
    parent_dir: Path,
    image_dir_names: Sequence[str],
    label_dir_names: Sequence[str],
) -> List[FolderPair]:
    parent_dir = parent_dir.expanduser().resolve()
    image_names = set(normalize_names(image_dir_names))
    label_names = set(normalize_names(label_dir_names))

    pairs: List[FolderPair] = []
    seen: set[Tuple[Path, Path]] = set()

    if not parent_dir.exists() or not parent_dir.is_dir():
        raise FileNotFoundError(
            f"Parent folder does not exist or is not a directory: {parent_dir}"
        )

    for root, dirs, _files in os_walk_safe(parent_dir):
        root_path = Path(root)
        children = {Path(d).name.lower(): Path(root) / d for d in dirs}

        image_dirs = [p for name, p in children.items() if name in image_names]
        label_dirs = [p for name, p in children.items() if name in label_names]

        if not image_dirs or not label_dirs:
            continue

        for img_dir in image_dirs:
            for lbl_dir in label_dirs:
                key = (img_dir.resolve(), lbl_dir.resolve())

                if key not in seen:
                    seen.add(key)
                    pairs.append(
                        FolderPair(
                            root=root_path.resolve(),
                            images_dir=key[0],
                            labels_dir=key[1],
                        )
                    )

    return pairs


def collect_items(
    pairs: Sequence[FolderPair],
    include_unlabeled: bool = False,
) -> List[ImageLabelItem]:
    items: List[ImageLabelItem] = []

    for pair in pairs:
        for image_path in sorted(pair.images_dir.rglob("*")):
            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue

            rel = image_path.relative_to(pair.images_dir)
            label_path = pair.labels_dir / rel.with_suffix(".txt")

            if label_path.exists():
                items.append(
                    ImageLabelItem(
                        image_path=image_path,
                        label_path=label_path,
                        pair_root=pair.root,
                    )
                )
            elif include_unlabeled:
                items.append(
                    ImageLabelItem(
                        image_path=image_path,
                        label_path=None,
                        pair_root=pair.root,
                    )
                )

    return items


def split_items(
    items: Sequence[ImageLabelItem],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> Dict[str, List[ImageLabelItem]]:
    ratios = [float(train_ratio), float(val_ratio), float(test_ratio)]

    if any(r < 0 for r in ratios):
        raise ValueError("Split ratios must be non-negative")

    total = sum(ratios)

    if total <= 0:
        raise ValueError("At least one split ratio must be greater than zero")

    train_r, val_r, _test_r = [r / total for r in ratios]

    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_r)
    n_val = int(n * val_r)

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def unique_output_stem(image_path: Path, parent_hint: Path) -> str:
    try:
        rel = image_path.relative_to(parent_hint)
    except Exception:
        rel = image_path

    digest = hashlib.sha1(str(rel).encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"{image_path.stem}_{digest}"


def rewrite_label_text(label_text: str, merge_rules: Dict[int, int]) -> Tuple[str, Counter]:
    output_lines: List[str] = []
    counter: Counter = Counter()

    for line_no, raw_line in enumerate(label_text.splitlines(), start=1):
        line = raw_line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 5:
            raise ValueError(f"Invalid YOLO label line {line_no}: '{raw_line}'")

        try:
            src_id = int(float(parts[0]))
        except ValueError as exc:
            raise ValueError(
                f"Invalid class id at label line {line_no}: '{parts[0]}'"
            ) from exc

        dst_id = merge_rules.get(src_id, src_id)
        parts[0] = str(dst_id)

        counter[dst_id] += 1
        output_lines.append(" ".join(parts))

    return "\n".join(output_lines) + ("\n" if output_lines else ""), counter


def counter_to_named_rows(
    counter: Counter,
    id_to_name: Dict[int, str],
) -> List[Dict[str, object]]:
    rows = []

    for cls_id, count in sorted(counter.items()):
        rows.append(
            {
                "class_id": cls_id,
                "class_name": id_to_name.get(cls_id, f"class_{cls_id}"),
                "count": int(count),
            }
        )

    return rows


def analyze_items(
    items: Sequence[ImageLabelItem],
    merge_rules: Dict[int, int],
    id_to_name: Dict[int, str],
) -> Dict[str, object]:
    original = Counter()
    mapped = Counter()
    invalid_labels: List[str] = []
    missing_labels = 0

    for item in items:
        if item.label_path is None:
            missing_labels += 1
            continue

        try:
            text = item.label_path.read_text(encoding="utf-8")

            for raw_line in text.splitlines():
                line = raw_line.strip()

                if not line:
                    continue

                parts = line.split()

                if len(parts) < 5:
                    raise ValueError(f"invalid line: {raw_line}")

                cls_id = int(float(parts[0]))
                original[cls_id] += 1
                mapped[merge_rules.get(cls_id, cls_id)] += 1

        except Exception as exc:
            invalid_labels.append(f"{item.label_path}: {exc}")

    return {
        "num_images": len(items),
        "num_labeled_images": len(items) - missing_labels,
        "num_unlabeled_images": missing_labels,
        "original_distribution": counter_to_named_rows(original, id_to_name),
        "mapped_distribution": counter_to_named_rows(mapped, id_to_name),
        "invalid_labels": invalid_labels,
    }


def ensure_output_dirs(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"Output path exists but is not a folder: {output_dir}")

        if any(output_dir.iterdir()) and not overwrite:
            raise FileExistsError(
                "Output folder is not empty. Enable overwrite or choose another folder: "
                f"{output_dir}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    (output_dir / "reports").mkdir(parents=True, exist_ok=True)


def write_data_yaml(output_dir: Path, id_to_name: Dict[int, str]) -> None:
    if not id_to_name:
        names = []
    else:
        max_id = max(id_to_name)
        names = [id_to_name.get(i, f"class_{i}") for i in range(max_id + 1)]

    def yaml_quote(s: str) -> str:
        return json.dumps(s, ensure_ascii=False)

    yaml_lines = [
        f"path: {yaml_quote(str(output_dir.resolve()))}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(names)}",
        "names:",
    ]

    for i, name in enumerate(names):
        yaml_lines.append(f"  {i}: {yaml_quote(name)}")

    (output_dir / "data.yaml").write_text(
        "\n".join(yaml_lines) + "\n",
        encoding="utf-8",
    )

    (output_dir / "classes.txt").write_text(
        "\n".join(names) + ("\n" if names else ""),
        encoding="utf-8",
    )

def verify_resized_outputs(
    manifest_rows: List[Dict[str, str]],
    target_w: int,
    target_h: int,
) -> Dict[str, object]:
    errors = []
    checked_images = 0
    checked_labels = 0

    for row in manifest_rows:
        img_path = Path(row["output_image"])
        lbl_path = Path(row["output_label"])

        checked_images += 1

        try:
            with Image.open(img_path) as im:
                w, h = im.size

            if w != target_w or h != target_h:
                errors.append(f"Image size mismatch: {img_path} got {w}x{h}, expected {target_w}x{target_h}")
        except Exception as exc:
            errors.append(f"Cannot read image: {img_path}: {exc}")

        if lbl_path.exists():
            checked_labels += 1
            try:
                for line_no, raw in enumerate(lbl_path.read_text(encoding="utf-8").splitlines(), start=1):
                    line = raw.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) < 5:
                        errors.append(f"Invalid label line: {lbl_path}:{line_no}: {raw}")
                        continue

                    x, y, bw, bh = map(float, parts[1:5])

                    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                        errors.append(f"YOLO bbox out of range: {lbl_path}:{line_no}: {raw}")

            except Exception as exc:
                errors.append(f"Cannot read label: {lbl_path}: {exc}")

    return {
        "valid": len(errors) == 0,
        "checked_images": checked_images,
        "checked_labels": checked_labels,
        "errors": errors,
    }

def write_reports(
    output_dir: Path,
    stats: Dict[str, object],
    manifest_rows: List[Dict[str, str]],
) -> None:
    reports_dir = output_dir / "reports"

    (reports_dir / "class_distribution.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for key, filename in (
        ("original_distribution", "class_distribution_original.csv"),
        ("mapped_distribution", "class_distribution_mapped.csv"),
    ):
        with (reports_dir / filename).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["class_id", "class_name", "count"])
            writer.writeheader()
            writer.writerows(stats.get(key, []))

    with (reports_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "source_image",
                "source_label",
                "output_image",
                "output_label",
                "operation",
                "resize_enabled",
                "resize_width",
                "resize_height",
                "resize_pad_black",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)


def build_dataset(
    config: BuildConfig,
    pairs: Optional[List[FolderPair]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, object]:
    if pairs is None:
        pairs = scan_folder_pairs(
            config.parent_dir,
            config.image_dir_names,
            config.label_dir_names,
        )

    if config.selected_pair_indices is not None:
        pairs = [pairs[i] for i in config.selected_pair_indices]

    if not pairs:
        raise RuntimeError("No image/label folder pairs selected or detected")

    items = collect_items(pairs, include_unlabeled=config.include_unlabeled)

    if not items:
        raise RuntimeError("No images found in selected folder pairs")

    pre_stats = analyze_items(items, config.merge_rules, config.id_to_name)
    invalid = pre_stats.get("invalid_labels", [])

    if invalid:
        preview = "\n".join(str(x) for x in invalid[:20])
        raise RuntimeError(
            f"Invalid label files found. Fix them before building.\n{preview}"
        )

    ensure_output_dirs(config.output_dir, overwrite=config.overwrite)

    splits = split_items(
        items,
        config.train_ratio,
        config.val_ratio,
        config.test_ratio,
        seed=config.seed,
    )

    manifest_rows: List[Dict[str, str]] = []

    op = shutil.copy2 if config.operation == "copy" else shutil.move

    total_items = sum(len(v) for v in splits.values())
    processed_items = 0

    if progress_callback:
        progress_callback(0, total_items)

    for split, split_items_list in splits.items():
        for item in split_items_list:
            stem = unique_output_stem(item.image_path, config.parent_dir)

            out_img = (
                config.output_dir
                / "images"
                / split
                / f"{stem}{item.image_path.suffix.lower()}"
            )
            out_lbl = config.output_dir / "labels" / split / f"{stem}.txt"

            if out_img.exists() and not config.overwrite:
                raise FileExistsError(f"Output image already exists: {out_img}")

            if out_lbl.exists() and not config.overwrite:
                raise FileExistsError(f"Output label already exists: {out_lbl}")

            if config.resize_enabled:
                src_w, src_h, scale_x, scale_y, pad_x, pad_y = resize_image_yolo(
                    src_img=item.image_path,
                    dst_img=out_img,
                    target_w=config.resize_width,
                    target_h=config.resize_height,
                    pad_black=config.resize_pad_black,
                )

                if item.label_path is not None:
                    label_text = item.label_path.read_text(encoding="utf-8")
                    rewritten, _label_counts = rewrite_label_text_with_resize(
                        label_text=label_text,
                        merge_rules=config.merge_rules,
                        src_w=src_w,
                        src_h=src_h,
                        dst_w=config.resize_width,
                        dst_h=config.resize_height,
                        scale_x=scale_x,
                        scale_y=scale_y,
                        pad_x=pad_x,
                        pad_y=pad_y,
                    )
                    out_lbl.write_text(rewritten, encoding="utf-8")
                else:
                    out_lbl.write_text("", encoding="utf-8")

                if config.operation == "move":
                    try:
                        item.image_path.unlink()
                    except FileNotFoundError:
                        pass

                    if item.label_path is not None:
                        try:
                            item.label_path.unlink()
                        except FileNotFoundError:
                            pass

            else:
                op(str(item.image_path), str(out_img))

                if item.label_path is not None:
                    label_text = item.label_path.read_text(encoding="utf-8")
                    rewritten, _label_counts = rewrite_label_text(
                        label_text,
                        config.merge_rules,
                    )
                    out_lbl.write_text(rewritten, encoding="utf-8")

                    if config.operation == "move":
                        try:
                            item.label_path.unlink()
                        except FileNotFoundError:
                            pass
                else:
                    out_lbl.write_text("", encoding="utf-8")

            manifest_rows.append(
                {
                    "split": split,
                    "source_image": str(item.image_path),
                    "source_label": str(item.label_path) if item.label_path else "",
                    "output_image": str(out_img),
                    "output_label": str(out_lbl),
                    "operation": config.operation,
                    "resize_enabled": str(config.resize_enabled),
                    "resize_width": str(config.resize_width if config.resize_enabled else ""),
                    "resize_height": str(config.resize_height if config.resize_enabled else ""),
                    "resize_pad_black": str(config.resize_pad_black if config.resize_enabled else ""),
                }
            )

            processed_items += 1

            if progress_callback:
                progress_callback(processed_items, total_items)

    final_stats = analyze_items(
        [
            ImageLabelItem(
                Path(row["output_image"]),
                Path(row["output_label"]),
                config.output_dir,
            )
            for row in manifest_rows
        ],
        {},
        config.id_to_name,
    )

    final_stats["split_counts"] = {k: len(v) for k, v in splits.items()}
    final_stats["selected_pairs"] = [p.display() for p in pairs]

    if config.resize_enabled:
        resize_check = verify_resized_outputs(
            manifest_rows=manifest_rows,
            target_w=config.resize_width,
            target_h=config.resize_height,
        )

        final_stats["resize"] = {
            "enabled": True,
            "width": config.resize_width,
            "height": config.resize_height,
            "pad_black": config.resize_pad_black,
            "verified": resize_check["valid"],
            "checked_images": resize_check["checked_images"],
            "checked_labels": resize_check["checked_labels"],
            "errors": resize_check["errors"][:50],
        }

        if not resize_check["valid"]:
            preview = "\n".join(resize_check["errors"][:10])
            raise RuntimeError(f"Resize verification failed:\n{preview}")
    else:
        final_stats["resize"] = {
            "enabled": False,
        }

    write_data_yaml(config.output_dir, config.id_to_name)
    write_reports(config.output_dir, final_stats, manifest_rows)

    return final_stats


def format_stats(stats: Dict[str, object]) -> str:
    lines = []
    lines.append(f"Images: {stats.get('num_images', 0)}")
    lines.append(f"Labeled images: {stats.get('num_labeled_images', 0)}")
    lines.append(f"Unlabeled images: {stats.get('num_unlabeled_images', 0)}")

    if "split_counts" in stats:
        lines.append(f"Splits: {stats['split_counts']}")

    lines.append("\nClass distribution:")

    rows = stats.get("mapped_distribution", []) or []

    if not rows:
        lines.append("  No objects found")

    for row in rows:
        lines.append(f"  {row['class_id']:>3}  {row['class_name']:<24} {row['count']}")

    invalid = stats.get("invalid_labels", []) or []

    if invalid:
        lines.append("\nInvalid labels:")
        lines.extend(f"  {x}" for x in invalid[:30])

    return "\n".join(lines)


def format_stats_localized(stats: Dict[str, object], lang: str = "en") -> str:
    zh = lang.startswith("zh")

    label = {
        "images": "圖像數量" if zh else "Images",
        "labeled_images": "有標記圖像" if zh else "Labeled images",
        "unlabeled_images": "無標記圖像" if zh else "Unlabeled images",
        "splits": "資料切分" if zh else "Splits",
        "class_distribution": "類別分布" if zh else "Class distribution",
        "no_objects": "未找到任何物件標記" if zh else "No objects found",
        "invalid_labels": "無效標記檔" if zh else "Invalid labels",
    }

    lines = []
    lines.append(f"{label['images']}: {stats.get('num_images', 0)}")
    lines.append(f"{label['labeled_images']}: {stats.get('num_labeled_images', 0)}")
    lines.append(f"{label['unlabeled_images']}: {stats.get('num_unlabeled_images', 0)}")

    if "split_counts" in stats:
        lines.append(f"{label['splits']}: {stats['split_counts']}")

    lines.append(f"\n{label['class_distribution']}:")

    rows = stats.get("mapped_distribution", []) or []

    if not rows:
        lines.append(f"  {label['no_objects']}")

    for row in rows:
        lines.append(f"  {row['class_id']:>3}  {row['class_name']:<24} {row['count']}")

    invalid = stats.get("invalid_labels", []) or []

    if invalid:
        lines.append(f"\n{label['invalid_labels']}:")
        lines.extend(f"  {x}" for x in invalid[:30])

    return "\n".join(lines)


def launch_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk
    import threading
    import queue
    I18N = {
        "zh": {
            "app_title": "YOLO 資料集產生工具",
            "language": "語言",
            "paths": "路徑設定",
            "parent_folder": "父級資料夾",
            "output_folder": "輸出資料夾",
            "browse": "瀏覽",
            "folder_discovery_and_split": "資料夾偵測與資料切分",
            "image_folder_names": "圖像資料夾名稱",
            "label_folder_names": "標記檔資料夾名稱",
            "train_val_test": "Train / Val / Test",
            "seed": "隨機種子",
            "copy_source_files": "複製來源檔案",
            "move_source_files": "移動來源檔案",
            "overwrite_output": "允許覆寫非空輸出資料夾",
            "include_unlabeled": "包含無標記圖像",
            "detected_folder_pairs": "偵測到的圖像 / 標記資料夾配對",
            "scan_pairs": "掃描配對",
            "select_all": "全選",
            "clear_selection": "清除選取",
            "class_mapping_and_merge_rules": "類別對應與合併規則",
            "class_mapping_hint": "類別對應格式：'0 person'，或每行一個 class name。",
            "import_class_mapping": "匯入類別對應",
            "status_distribution": "狀態 / 類別分布",
            "analyze_distribution": "分析類別分布",
            "build_dataset": "建立資料集",
            "choose_parent_title": "選擇父級資料夾",
            "choose_output_title": "選擇輸出資料夾",
            "import_class_mapping_title": "匯入類別對應檔",
            "import_failed": "匯入失敗",
            "error": "錯誤",
            "warning": "警告",
            "move_source_files_title": "移動來源檔案",
            "move_source_files_message": "Move 模式會將圖像從來源資料夾移動到輸出資料夾，來源圖像將不再保留。是否繼續？",
            "completed": "完成",
            "dataset_generated": "資料集已產生",
            "detected_n_pairs": "偵測到 {n} 組資料夾配對。",
            "build_completed": "建立完成。",
            "output": "輸出位置",
            "source_classes": "來源類別（Shift/Ctrl + Click 可多選）",
            "target_class": "合併到目標類別",
            "refresh_class_list": "更新類別清單",
            "add_merge_rule": "新增合併規則",
            "current_merge_rules": "目前合併規則",
            "remove_selected_rule": "移除選取規則",
            "clear_merge_rules": "清除全部規則",
            "no_target_class": "請先選擇目標類別",
            "no_source_class": "請先選擇至少一個來源類別",
            "class_list_updated": "類別清單已更新，共 {n} 個類別。",
            "same_class_ignored": "來源類別與目標類別相同者已略過。",
            "merge_rule_added": "已新增 / 更新 {n} 條合併規則。",
            "page": "頁面",
            "main_tool_page": "主工具",
            "info_page": "使用說明",
            "custom_target_class_name": "自定義合併後類別名",
            "progress": "建立進度",
            "progress": "建立進度",
            "resize_output_image": "縮放輸出圖像",
            "resize_width": "寬度",
            "resize_height": "高度",
            "keep_ratio_pad_black": "保持比例並填充黑邊",
        },
        "en": {
            "app_title": "YOLO Dataset Builder",
            "language": "Language",
            "paths": "Paths",
            "parent_folder": "Parent folder",
            "output_folder": "Output folder",
            "browse": "Browse",
            "folder_discovery_and_split": "Folder discovery and split",
            "image_folder_names": "Image folder names",
            "label_folder_names": "Label folder names",
            "train_val_test": "Train / Val / Test",
            "seed": "Seed",
            "copy_source_files": "Copy source files",
            "move_source_files": "Move source files",
            "overwrite_output": "Overwrite non-empty output folder",
            "include_unlabeled": "Include unlabeled images",
            "detected_folder_pairs": "Detected folder pairs",
            "scan_pairs": "Scan pairs",
            "select_all": "Select all",
            "clear_selection": "Clear selection",
            "class_mapping_and_merge_rules": "Class mapping and merge rules",
            "class_mapping_hint": "Class mapping. Format: '0 person' or one class name per line.",
            "import_class_mapping": "Import class mapping",
            "status_distribution": "Status / distribution",
            "analyze_distribution": "Analyze distribution",
            "build_dataset": "Build dataset",
            "choose_parent_title": "Choose parent folder",
            "choose_output_title": "Choose output folder",
            "import_class_mapping_title": "Import class mapping",
            "import_failed": "Import failed",
            "error": "Error",
            "warning": "Warning",
            "move_source_files_title": "Move source files",
            "move_source_files_message": "Move mode will remove images from the source folders after moving them to output. Continue?",
            "completed": "Completed",
            "dataset_generated": "Dataset generated",
            "detected_n_pairs": "Detected {n} folder pair(s).",
            "build_completed": "Build completed.",
            "output": "Output",
            "source_classes": "Source classes(Shift/Ctrl + Click to multi-select)",
            "target_class": "Merge into target class",
            "refresh_class_list": "Refresh class list",
            "add_merge_rule": "Add merge rule",
            "current_merge_rules": "Current merge rules",
            "remove_selected_rule": "Remove selected rule",
            "clear_merge_rules": "Clear all rules",
            "no_target_class": "Select a target class first",
            "no_source_class": "Select at least one source class first",
            "class_list_updated": "Class list updated. {n} class(es) found.",
            "same_class_ignored": "Rules with the same source and target class were ignored.",
            "merge_rule_added": "{n} merge rule(s) added / updated.",
            "page": "Page",
            "main_tool_page": "Main Tool",
            "info_page": "Instructions",
            "custom_target_class_name": "Custom merged class name",
            "progress": "Build progress",
            "progress": "Build progress",
            "resize_output_image": "Resize output image",
            "resize_width": "Width",
            "resize_height": "Height",
            "keep_ratio_pad_black": "Keep ratio and pad black",
        },
    }

    class App(tk.Tk):
        def __init__(self):
            super().__init__()

            self.lang = "zh"
            self.lang_display_var = tk.StringVar(value="繁體中文")
            self.i18n_widgets: List[Tuple[object, str, str]] = []

            self.pairs: List[FolderPair] = []
            self.gui_merge_rules: Dict[int, int] = {}
            self.custom_merged_class_names: Dict[int, str] = {}
            self.class_display_to_id: Dict[str, int] = {}
            self.class_option_rows: List[Tuple[int, str, int]] = []

            self.title(self.t("app_title"))
            self.geometry("1920x1080")
            self.minsize(1080, 760)

            self._build_ui()
            self.refresh_language()
            self.bulid_thread = None
            self.build_queue = queue.Queue()
            self.is_building = False
            
        
        def enqueue_build_progress(self, current: int, total: int):
            self.build_queue.put(("progress", current, total))


        def poll_build_queue(self):
            try:
                while True:
                    msg = self.build_queue.get_nowait()
                    msg_type = msg[0]

                    if msg_type == "progress":
                        _msg_type, current, total = msg
                        self.update_build_progress(current, total)

                    elif msg_type == "done":
                        _msg_type, stats, output_dir = msg

                        self.is_building = False
                        self.update_build_progress(0, 0)

                        self.write_status(
                            self.t("build_completed")
                            + "\n\n"
                            + format_stats_localized(stats, self.lang)
                            + f"\n\n{self.t('output')}: {Path(output_dir).resolve()}"
                        )

                        messagebox.showinfo(
                            self.t("completed"),
                            f"{self.t('dataset_generated')}:\n{Path(output_dir).resolve()}",
                        )

                    elif msg_type == "error":
                        _msg_type, error_text, traceback_text = msg

                        self.is_building = False
                        self.write_status(traceback_text)
                        messagebox.showerror(self.t("error"), error_text)

            except queue.Empty:
                pass

            if self.is_building:
                self.after(100, self.poll_build_queue)


        
        def t(self, key: str) -> str:
            return I18N.get(self.lang, I18N["en"]).get(key, key)

        def bind_i18n(self, widget, key: str, option: str = "text"):
            self.i18n_widgets.append((widget, key, option))
            widget.configure(**{option: self.t(key)})
            return widget

        def refresh_language(self, _event=None):
            self.lang = "en" if self.lang_display_var.get() == "English" else "zh"
            self.title(self.t("app_title"))

            for widget, key, option in self.i18n_widgets:
                try:
                    widget.configure(**{option: self.t(key)})
                except Exception:
                    pass

            if hasattr(self, "merge_rule_list"):
                self.refresh_merge_rule_list()

            if hasattr(self, "page_combo"):
                self.page_options = {
                    "main": self.t("main_tool_page"),
                    "info": self.t("info_page"),
                }
                self.page_combo["values"] = list(self.page_options.values())
                current_page = getattr(self, "current_page_key", "main")
                self.page_var.set(self.page_options[current_page])

        def get_effective_id_to_name(self) -> Dict[int, str]:
            id_to_name = self.get_current_id_to_name()
            id_to_name.update(self.custom_merged_class_names)
            return id_to_name

        def make_final_class_plan(self) -> Tuple[Dict[int, str], Dict[int, int]]:
            original_id_to_name = self.get_current_id_to_name()
            effective_id_to_name = self.get_effective_id_to_name()

            source_ids = set(original_id_to_name.keys())

            for class_id, _class_name, _count in self.class_option_rows:
                if class_id in original_id_to_name:
                    source_ids.add(class_id)

            for src_id in self.gui_merge_rules.keys():
                source_ids.add(src_id)

            merged_source_ids = sorted(self.gui_merge_rules.keys())
            unmerged_source_ids = sorted(source_ids - set(merged_source_ids))
            ordered_source_ids = merged_source_ids + unmerged_source_ids

            name_to_final_id: Dict[str, int] = {}
            final_id_to_name: Dict[int, str] = {}
            final_merge_rules: Dict[int, int] = {}

            for src_id in ordered_source_ids:
                target_id = self.gui_merge_rules.get(src_id, src_id)
                final_name = effective_id_to_name.get(target_id, f"class_{target_id}").strip()

                if not final_name:
                    final_name = f"class_{target_id}"

                dedupe_key = final_name.casefold()

                if dedupe_key not in name_to_final_id:
                    new_id = len(name_to_final_id)
                    name_to_final_id[dedupe_key] = new_id
                    final_id_to_name[new_id] = final_name

                final_merge_rules[src_id] = name_to_final_id[dedupe_key]

            return final_id_to_name, final_merge_rules

        def on_page_selected(self, _event=None):
            selected_text = self.page_var.get()
            if selected_text == self.t("info_page"):
                self.show_page("info")
            else:
                self.show_page("main")

        def show_page(self, page_key: str):
            self.current_page_key = page_key

            if page_key == "info":
                self.info_page.tkraise()
            else:
                self.main_page.tkraise()

        def _build_ui(self):
            pad = {"padx": 8, "pady": 5}

            top_bar = ttk.Frame(self)
            top_bar.pack(fill="x", padx=8, pady=(6, 0))
            top_bar.columnconfigure(0, weight=1)

            self.bind_i18n(ttk.Label(top_bar), "language").grid(
                row=0,
                column=1,
                sticky="e",
                padx=(0, 6),
            )

            lang_combo = ttk.Combobox(
                top_bar,
                textvariable=self.lang_display_var,
                values=["繁體中文", "English"],
                state="readonly",
                width=12,
            )
            lang_combo.grid(row=0, column=2, sticky="e")
            lang_combo.bind("<<ComboboxSelected>>", self.refresh_language)

            self.bind_i18n(ttk.Label(top_bar), "page").grid(
                row=0,
                column=3,
                sticky="e",
                padx=(12, 6),
            )

            self.current_page_key = "main"
            self.page_options = {
                "main": self.t("main_tool_page"),
                "info": self.t("info_page"),
            }

            self.page_var = tk.StringVar(value=self.page_options["main"])

            self.page_combo = ttk.Combobox(
                top_bar,
                textvariable=self.page_var,
                values=list(self.page_options.values()),
                state="readonly",
                width=12,
            )
            self.page_combo.grid(row=0, column=4, sticky="e")
            self.page_combo.bind("<<ComboboxSelected>>", self.on_page_selected)

            self.page_container = ttk.Frame(self)
            self.page_container.pack(fill="both", expand=True)

            self.main_page = ttk.Frame(self.page_container)
            self.info_page = ttk.Frame(self.page_container)

            self.page_container.rowconfigure(0, weight=1)
            self.page_container.columnconfigure(0, weight=1)

            self.main_page.grid(row=0, column=0, sticky="nsew")
            self.info_page.grid(row=0, column=0, sticky="nsew")

            ttk.Label(
                self.info_page,
                text=(
                    "操作流程說明：\n\n"
                    "1. 選擇父級資料夾與輸出資料夾。\n"
                    "2. 設定圖像資料夾名稱與標記檔資料夾名稱。\n"
                    "3. 點擊「掃描配對」，確認 images 與 labels 配對結果。\n"
                    "4. 設定 Train / Val / Test 切分比例。\n"
                    "5. 輸入或匯入 class id 與 class name 對應。\n"
                    "6. 點擊「更新類別清單」，查看各類別數量。\n"
                    "7. 如需合併類別，選擇來源類別與目標類別後，點擊「新增合併規則」。\n"
                    "8. 點擊「分析類別分布」確認資料狀態。\n"
                    "9. 點擊「建立資料集」輸出 YOLO 格式資料集。\n\n"
                    "注意：預設使用 Copy，不會修改來源資料；Move 模式會移動原始檔案。"
                    f"\n\n{'=' * 40}\n\n"
                    "Workflow Instructions:\n\n"
                    "1. Select the parent folder and output folder.\n"
                    "2. Specify the image folder names and label folder names.\n"
                    "3. Click \"Scan pairs\" to verify the detected images and labels folder pairs.\n"
                    "4. Set the Train / Val / Test split ratios.\n"
                    "5. Input or import the class id to class name mapping.\n"
                    "6. Click \"Refresh class list\" to view the distribution of each class.\n"
                    "7. To merge classes, select source classes and a target class, then click \"Add merge rule\".\n"
                    "8. Click \"Analyze distribution\" to check dataset statistics.\n"
                    "9. Click \"Build dataset\" to generate the YOLO-format dataset.\n\n"
                    "Note: Copy mode is used by default and will not modify source data; Move mode will relocate original files."
                ),
                justify="left",
                anchor="w",
            ).pack(anchor="nw", padx=16, pady=16)

            frm = ttk.Frame(self.main_page)
            frm.pack(fill="both", expand=True)

            paths = self.bind_i18n(ttk.LabelFrame(frm), "paths")
            paths.pack(fill="x", **pad)

            self.parent_var = tk.StringVar()
            self.output_var = tk.StringVar()

            self._row_path(paths, 0, "parent_folder", self.parent_var, self.choose_parent)
            self._row_path(paths, 1, "output_folder", self.output_var, self.choose_output)

            opts = self.bind_i18n(ttk.LabelFrame(frm), "folder_discovery_and_split")
            opts.pack(fill="x", **pad)

            self.image_names_var = tk.StringVar(value=DEFAULT_IMAGE_DIR_NAMES)
            self.label_names_var = tk.StringVar(value=DEFAULT_LABEL_DIR_NAMES)
            self.train_var = tk.StringVar(value="0.8")
            self.val_var = tk.StringVar(value="0.1")
            self.test_var = tk.StringVar(value="0.1")
            self.include_unlabeled_var = tk.BooleanVar(value=False)
            self.overwrite_var = tk.BooleanVar(value=False)
            self.operation_var = tk.StringVar(value="copy")
            self.seed_var = tk.StringVar(value="42")
            self.resize_enabled_var = tk.BooleanVar(value=False)
            self.resize_width_var = tk.StringVar(value="640")
            self.resize_height_var = tk.StringVar(value="640")
            self.resize_pad_black_var = tk.BooleanVar(value=True)

            self.bind_i18n(ttk.Label(opts), "image_folder_names").grid(row=0, column=0, sticky="w", **pad)
            ttk.Entry(opts, textvariable=self.image_names_var, width=40).grid(row=0, column=1, sticky="we", **pad)

            self.bind_i18n(ttk.Label(opts), "label_folder_names").grid(row=0, column=2, sticky="w", **pad)
            ttk.Entry(opts, textvariable=self.label_names_var, width=40).grid(row=0, column=3, sticky="we", **pad)

            self.bind_i18n(ttk.Label(opts), "train_val_test").grid(row=1, column=0, sticky="w", **pad)

            ratio_frame = ttk.Frame(opts)
            ratio_frame.grid(row=1, column=1, sticky="w", **pad)

            ttk.Entry(ratio_frame, textvariable=self.train_var, width=8).pack(side="left")
            ttk.Entry(ratio_frame, textvariable=self.val_var, width=8).pack(side="left", padx=4)
            ttk.Entry(ratio_frame, textvariable=self.test_var, width=8).pack(side="left")

            self.bind_i18n(ttk.Label(opts), "seed").grid(row=1, column=2, sticky="w", **pad)
            ttk.Entry(opts, textvariable=self.seed_var, width=10).grid(row=1, column=3, sticky="w", **pad)

            self.bind_i18n(
                ttk.Radiobutton(opts, variable=self.operation_var, value="copy"),
                "copy_source_files",
            ).grid(row=2, column=0, sticky="w", **pad)

            self.bind_i18n(
                ttk.Radiobutton(opts, variable=self.operation_var, value="move"),
                "move_source_files",
            ).grid(row=2, column=1, sticky="w", **pad)

            self.bind_i18n(
                ttk.Checkbutton(opts, variable=self.overwrite_var),
                "overwrite_output",
            ).grid(row=2, column=2, sticky="w", **pad)

            self.bind_i18n(
                ttk.Checkbutton(opts, variable=self.include_unlabeled_var),
                "include_unlabeled",
            ).grid(row=2, column=3, sticky="w", **pad)
            resize_frame = ttk.Frame(opts)
            resize_frame.grid(row=3, column=0, columnspan=4, sticky="w", padx=8, pady=5)

            self.bind_i18n(
                ttk.Checkbutton(
                    resize_frame,
                    variable=self.resize_enabled_var,
                ),
                "resize_output_image",
            ).pack(side="left", padx=(0, 14))

            self.bind_i18n(
                ttk.Label(resize_frame),
                "resize_width",
            ).pack(side="left", padx=(0, 4))

            ttk.Entry(
                resize_frame,
                textvariable=self.resize_width_var,
                width=8,
            ).pack(side="left", padx=(0, 12))

            self.bind_i18n(
                ttk.Label(resize_frame),
                "resize_height",
            ).pack(side="left", padx=(0, 4))

            ttk.Entry(
                resize_frame,
                textvariable=self.resize_height_var,
                width=8,
            ).pack(side="left", padx=(0, 12))

            self.bind_i18n(
                ttk.Checkbutton(
                    resize_frame,
                    variable=self.resize_pad_black_var,
                ),
                "keep_ratio_pad_black",
            ).pack(side="left", padx=(0, 0))


            opts.columnconfigure(1, weight=1)
            opts.columnconfigure(3, weight=1)

            pair_frame = self.bind_i18n(ttk.LabelFrame(frm), "detected_folder_pairs")
            pair_frame.pack(fill="both", expand=False, **pad)

            btns = ttk.Frame(pair_frame)
            btns.pack(fill="x")

            self.bind_i18n(ttk.Button(btns, command=self.scan_pairs), "scan_pairs").pack(side="left", padx=4, pady=4)
            self.bind_i18n(ttk.Button(btns, command=lambda: self.pair_list.selection_set(0, tk.END)), "select_all").pack(side="left", padx=4, pady=4)
            self.bind_i18n(ttk.Button(btns, command=lambda: self.pair_list.selection_clear(0, tk.END)), "clear_selection").pack(side="left", padx=4, pady=4)

            self.pair_list = tk.Listbox(pair_frame, selectmode="extended", height=6)
            self.pair_list.pack(fill="x", padx=6, pady=6)

            main_pane = tk.PanedWindow(frm, orient=tk.VERTICAL, sashwidth=6, bd=0, relief="flat")
            main_pane.pack(fill="both", expand=True, padx=8, pady=5)

            class_frame = self.bind_i18n(ttk.LabelFrame(main_pane), "class_mapping_and_merge_rules")
            class_frame.columnconfigure(0, weight=1)
            class_frame.columnconfigure(1, weight=1)
            class_frame.rowconfigure(0, weight=1)
            main_pane.add(class_frame, minsize=240)

            left = ttk.Frame(class_frame)
            right = ttk.Frame(class_frame)

            left.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
            right.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)

            left.rowconfigure(1, weight=1)
            left.columnconfigure(0, weight=1)

            self.bind_i18n(ttk.Label(left), "class_mapping_hint").grid(row=0, column=0, sticky="w")

            self.class_text = scrolledtext.ScrolledText(left, height=8, wrap="none")
            self.class_text.grid(row=1, column=0, sticky="nsew", pady=(2, 4))

            self.bind_i18n(ttk.Button(left, command=self.import_classes), "import_class_mapping").grid(row=2, column=0, sticky="w", pady=4)

            right.columnconfigure(0, weight=1)
            right.columnconfigure(1, weight=1)
            right.rowconfigure(1, weight=1)
            right.rowconfigure(5, weight=1)

            self.bind_i18n(ttk.Label(right), "source_classes").grid(row=0, column=0, sticky="w")
            self.bind_i18n(ttk.Label(right), "target_class").grid(row=0, column=1, sticky="w")

            self.source_class_list = tk.Listbox(right, selectmode="extended", exportselection=False, height=8)
            self.source_class_list.grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=(2, 4))

            target_area = ttk.Frame(right)
            target_area.grid(row=1, column=1, sticky="nsew", pady=(2, 4))
            target_area.columnconfigure(0, weight=1)

            self.target_class_var = tk.StringVar()
            self.target_class_combo = ttk.Combobox(target_area, textvariable=self.target_class_var, state="readonly")
            self.target_class_combo.grid(row=0, column=0, sticky="we")

            self.bind_i18n(ttk.Label(target_area), "custom_target_class_name").grid(row=1, column=0, sticky="w", pady=(6, 0))

            self.custom_target_class_var = tk.StringVar()
            ttk.Entry(target_area, textvariable=self.custom_target_class_var).grid(row=2, column=0, sticky="we", pady=(2, 0))

            self.bind_i18n(ttk.Button(target_area, command=self.update_class_options), "refresh_class_list").grid(row=3, column=0, sticky="we", pady=(6, 0))
            self.bind_i18n(ttk.Button(target_area, command=self.add_merge_rule_from_selection), "add_merge_rule").grid(row=4, column=0, sticky="we", pady=(6, 0))

            self.bind_i18n(ttk.Label(right), "current_merge_rules").grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))

            self.merge_rule_list = tk.Listbox(right, selectmode="extended", exportselection=False, height=5)
            self.merge_rule_list.grid(row=5, column=0, columnspan=2, sticky="nsew", pady=(2, 4))

            rule_btns = ttk.Frame(right)
            rule_btns.grid(row=6, column=0, columnspan=2, sticky="we")
            rule_btns.columnconfigure(0, weight=1)
            rule_btns.columnconfigure(1, weight=1)

            self.bind_i18n(ttk.Button(rule_btns, command=self.remove_selected_merge_rules), "remove_selected_rule").grid(row=0, column=0, sticky="we", padx=(0, 4))
            self.bind_i18n(ttk.Button(rule_btns, command=self.clear_merge_rules), "clear_merge_rules").grid(row=0, column=1, sticky="we", padx=(4, 0))

            output_frame = self.bind_i18n(ttk.LabelFrame(main_pane), "status_distribution")
            output_frame.rowconfigure(1, weight=1)
            output_frame.columnconfigure(0, weight=1)
            main_pane.add(output_frame, minsize=180)

            action_bar = ttk.Frame(output_frame)
            action_bar.grid(row=0, column=0, sticky="we", padx=6, pady=4)

            self.bind_i18n(ttk.Button(action_bar, command=self.analyze), "analyze_distribution").pack(side="left", padx=4)
            self.bind_i18n(ttk.Button(action_bar, command=self.build), "build_dataset").pack(side="left", padx=4)

            self.status_text = scrolledtext.ScrolledText(output_frame, height=10, wrap="none")
            self.status_text.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)

            progress_frame = ttk.Frame(output_frame)
            progress_frame.grid(row=2, column=0, sticky="we", padx=6, pady=(0, 6))
            progress_frame.columnconfigure(1, weight=1)

            self.bind_i18n(ttk.Label(progress_frame), "progress").grid(row=0, column=0, sticky="w", padx=(0, 8))

            self.progress_var = tk.DoubleVar(value=0)
            self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
            self.progress_bar.grid(row=0, column=1, sticky="we", padx=(0, 8))

            self.progress_label_var = tk.StringVar(value="0 / 0")
            ttk.Label(progress_frame, textvariable=self.progress_label_var, width=12).grid(row=0, column=2, sticky="e")

            self.show_page("main")

        def _row_path(self, parent, row, label_key, var, cmd):
            self.bind_i18n(ttk.Label(parent), label_key).grid(row=row, column=0, sticky="w", padx=8, pady=5)
            ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="we", padx=8, pady=5)
            self.bind_i18n(ttk.Button(parent, command=cmd), "browse").grid(row=row, column=2, sticky="e", padx=8, pady=5)
            parent.columnconfigure(1, weight=1)

        def choose_parent(self):
            path = filedialog.askdirectory(title=self.t("choose_parent_title"))
            if path:
                self.parent_var.set(path)

        def choose_output(self):
            path = filedialog.askdirectory(title=self.t("choose_output_title"))
            if path:
                self.output_var.set(path)

        def import_classes(self):
            path = filedialog.askopenfilename(
                title=self.t("import_class_mapping_title"),
                filetypes=[
                    ("Text/JSON", "*.txt *.json *.names *.yaml *.yml"),
                    ("All files", "*.*"),
                ],
            )

            if not path:
                return

            try:
                text = Path(path).read_text(encoding="utf-8")
                self.class_text.delete("1.0", tk.END)
                self.class_text.insert(tk.END, text)

                if self.pairs:
                    self.update_class_options(write_message=False)
                else:
                    self.refresh_merge_rule_list()

            except Exception as exc:
                messagebox.showerror(self.t("import_failed"), str(exc))

        def get_selected_pairs(self) -> List[FolderPair]:
            if not self.pairs:
                return []

            sel = list(self.pair_list.curselection())
            if not sel:
                return self.pairs

            return [self.pairs[i] for i in sel]

        def scan_pairs(self):
            try:
                parent = Path(self.parent_var.get()).expanduser()

                self.pairs = scan_folder_pairs(
                    parent,
                    normalize_names(self.image_names_var.get()),
                    normalize_names(self.label_names_var.get()),
                )

                self.pair_list.delete(0, tk.END)

                for p in self.pairs:
                    self.pair_list.insert(tk.END, p.display())

                self.pair_list.selection_set(0, tk.END)
                self.write_status(self.t("detected_n_pairs").format(n=len(self.pairs)) + "\n")

            except Exception as exc:
                self.show_error(exc)

        def get_class_name(self, class_id: int, id_to_name: Dict[int, str]) -> str:
            return id_to_name.get(class_id, f"class_{class_id}")

        def format_class_display(
            self,
            class_id: int,
            id_to_name: Dict[int, str],
            count: Optional[int] = None,
        ) -> str:
            name = self.get_class_name(class_id, id_to_name)

            if count is None:
                return f"{class_id:>3}  {name}"

            return f"{class_id:>3}  {name:<24} {count}"

        def get_current_id_to_name(self) -> Dict[int, str]:
            return parse_class_mapping(self.class_text.get("1.0", "end"))

        def get_current_items_for_class_scan(self) -> List[ImageLabelItem]:
            if not self.pairs:
                self.scan_pairs()

            selected_pairs = self.get_selected_pairs()
            return collect_items(selected_pairs, include_unlabeled=self.include_unlabeled_var.get())

        def update_class_options(self, write_message: bool = True):
            try:
                id_to_name = self.get_current_id_to_name()
                effective_id_to_name = self.get_effective_id_to_name()
                items = self.get_current_items_for_class_scan()

                counter = Counter()
                invalid_count = 0

                for item in items:
                    if item.label_path is None:
                        continue

                    try:
                        text = item.label_path.read_text(encoding="utf-8")

                        for raw_line in text.splitlines():
                            line = raw_line.strip()

                            if not line:
                                continue

                            parts = line.split()

                            if len(parts) < 5:
                                invalid_count += 1
                                continue

                            class_id = int(float(parts[0]))
                            counter[class_id] += 1

                    except Exception:
                        invalid_count += 1
                        continue

                all_class_ids = set(counter.keys()) | set(id_to_name.keys()) | set(self.custom_merged_class_names.keys())
                class_ids = sorted(all_class_ids)

                self.class_option_rows = [
                    (
                        class_id,
                        self.get_class_name(class_id, effective_id_to_name),
                        int(counter.get(class_id, 0)),
                    )
                    for class_id in class_ids
                ]

                self.class_display_to_id.clear()
                self.source_class_list.delete(0, tk.END)

                combo_values = []

                for class_id, class_name, count in self.class_option_rows:
                    display = f"{class_id:>3}  {class_name:<24} {count}"
                    self.class_display_to_id[display] = class_id
                    self.source_class_list.insert(tk.END, display)
                    combo_values.append(display)

                self.target_class_combo["values"] = combo_values

                current_target = self.target_class_var.get()
                if current_target not in combo_values:
                    self.target_class_var.set(combo_values[0] if combo_values else "")

                valid_ids = set(class_ids)

                self.gui_merge_rules = {
                    src_id: target_id
                    for src_id, target_id in self.gui_merge_rules.items()
                    if src_id in valid_ids and target_id in valid_ids
                }

                used_target_ids = set(self.gui_merge_rules.values())
                self.custom_merged_class_names = {
                    class_id: name
                    for class_id, name in self.custom_merged_class_names.items()
                    if class_id in used_target_ids
                }

                self.refresh_merge_rule_list()

                if write_message:
                    msg = self.t("class_list_updated").format(n=len(class_ids))
                    if invalid_count:
                        msg += (
                            "\nInvalid / unreadable label lines skipped during "
                            f"class-list refresh: {invalid_count}"
                        )
                    self.write_status(msg)

            except Exception as exc:
                self.show_error(exc)

        def add_merge_rule_from_selection(self):
            try:
                custom_name = self.custom_target_class_var.get().strip()

                selected_indices = list(self.source_class_list.curselection())

                if not selected_indices:
                    messagebox.showwarning(self.t("warning"), self.t("no_source_class"))
                    return

                if custom_name:
                    existing_name_to_id = {
                        name.casefold(): class_id
                        for class_id, name in self.custom_merged_class_names.items()
                    }

                    if custom_name.casefold() in existing_name_to_id:
                        target_id = existing_name_to_id[custom_name.casefold()]
                    else:
                        existing_ids = set(self.get_current_id_to_name().keys()) | set(self.custom_merged_class_names.keys())
                        target_id = 0
                        while target_id in existing_ids:
                            target_id += 1

                        self.custom_merged_class_names[target_id] = custom_name
                else:
                    target_display = self.target_class_var.get()

                    if not target_display:
                        messagebox.showwarning(self.t("warning"), self.t("no_target_class"))
                        return

                    target_id = self.class_display_to_id.get(target_display)

                    if target_id is None:
                        messagebox.showwarning(self.t("warning"), self.t("no_target_class"))
                        return

                added = 0
                same_class_ignored = False

                for index in selected_indices:
                    src_display = self.source_class_list.get(index)
                    src_id = self.class_display_to_id.get(src_display)

                    if src_id is None:
                        continue

                    if src_id == target_id:
                        same_class_ignored = True
                        continue

                    self.gui_merge_rules[src_id] = target_id
                    added += 1

                self.refresh_merge_rule_list()
                self.update_class_options(write_message=False)

                lines = []

                if added:
                    lines.append(self.t("merge_rule_added").format(n=added))

                if same_class_ignored:
                    lines.append(self.t("same_class_ignored"))

                if lines:
                    self.write_status("\n".join(lines))

            except Exception as exc:
                self.show_error(exc)

        def refresh_merge_rule_list(self):
            try:
                if not hasattr(self, "merge_rule_list"):
                    return

                effective_id_to_name = self.get_effective_id_to_name()

                try:
                    final_id_to_name, final_merge_rules = self.make_final_class_plan()
                except Exception:
                    final_id_to_name = {}
                    final_merge_rules = {}

                self.merge_rule_list.delete(0, tk.END)

                for src_id, target_id in sorted(self.gui_merge_rules.items()):
                    src_display = self.format_class_display(src_id, effective_id_to_name)

                    final_target_id = final_merge_rules.get(src_id)
                    if final_target_id is None:
                        target_display = self.format_class_display(target_id, effective_id_to_name)
                    else:
                        target_name = final_id_to_name.get(final_target_id, f"class_{final_target_id}")
                        target_display = f"{final_target_id:>3}  {target_name}"

                    self.merge_rule_list.insert(
                        tk.END,
                        f"{src_display}  ->  {target_display}",
                    )

            except Exception as exc:
                self.show_error(exc)

        def remove_selected_merge_rules(self):
            try:
                selected_indices = list(self.merge_rule_list.curselection())

                if not selected_indices:
                    return

                sorted_rules = sorted(self.gui_merge_rules.items())

                for index in reversed(selected_indices):
                    if index < len(sorted_rules):
                        src_id, _target_id = sorted_rules[index]
                        self.gui_merge_rules.pop(src_id, None)

                used_target_ids = set(self.gui_merge_rules.values())
                self.custom_merged_class_names = {
                    class_id: name
                    for class_id, name in self.custom_merged_class_names.items()
                    if class_id in used_target_ids
                }

                self.refresh_merge_rule_list()
                self.update_class_options(write_message=False)

            except Exception as exc:
                self.show_error(exc)

        def clear_merge_rules(self):
            self.gui_merge_rules.clear()
            self.custom_merged_class_names.clear()

            if hasattr(self, "custom_target_class_var"):
                self.custom_target_class_var.set("")

            self.refresh_merge_rule_list()
            self.update_class_options(write_message=False)

        def make_config(self) -> BuildConfig:
            id_to_name, merge_rules = self.make_final_class_plan()

            selected = list(self.pair_list.curselection()) if self.pairs else None

            if selected == []:
                selected = None

            return BuildConfig(
                parent_dir=Path(self.parent_var.get()).expanduser(),
                output_dir=Path(self.output_var.get()).expanduser(),
                image_dir_names=normalize_names(self.image_names_var.get()),
                label_dir_names=normalize_names(self.label_names_var.get()),
                train_ratio=float(self.train_var.get()),
                val_ratio=float(self.val_var.get()),
                test_ratio=float(self.test_var.get()),
                id_to_name=id_to_name,
                merge_rules=merge_rules,
                selected_pair_indices=selected,
                operation=self.operation_var.get(),
                overwrite=self.overwrite_var.get(),
                include_unlabeled=self.include_unlabeled_var.get(),
                seed=int(self.seed_var.get()),

                resize_enabled=self.resize_enabled_var.get(),
                resize_width=int(self.resize_width_var.get()),
                resize_height=int(self.resize_height_var.get()),
                resize_pad_black=self.resize_pad_black_var.get(),
            )

        def analyze(self):
            try:
                if not self.pairs:
                    self.scan_pairs()

                self.update_class_options(write_message=False)

                config = self.make_config()
                selected_pairs = self.get_selected_pairs()

                items = collect_items(selected_pairs, include_unlabeled=config.include_unlabeled)
                stats = analyze_items(items, config.merge_rules, config.id_to_name)
                self.write_status(format_stats_localized(stats, self.lang))

            except Exception as exc:
                self.show_error(exc)

        def update_build_progress(self, current: int, total: int):
            if total <= 0:
                percent = 0
            else:
                percent = current / total * 100

            self.progress_var.set(percent)
            self.progress_label_var.set(f"{current} / {total}")
            self.update_idletasks()

        def build(self):
            try:
                if self.is_building:
                    messagebox.showwarning(
                        self.t("warning"),
                        "Dataset build is already running.",
                    )
                    return

                if self.operation_var.get() == "move":
                    ok = messagebox.askyesno(
                        self.t("move_source_files_title"),
                        self.t("move_source_files_message"),
                    )

                    if not ok:
                        return

                if not self.pairs:
                    self.scan_pairs()

                self.update_class_options(write_message=False)

                config = self.make_config()
                pairs = list(self.pairs)

                self.is_building = True
                self.update_build_progress(0, 0)

                def worker():
                    try:
                        stats = build_dataset(
                            config,
                            pairs=pairs,
                            progress_callback=self.enqueue_build_progress,
                        )

                        self.build_queue.put(
                            (
                                "done",
                                stats,
                                str(config.output_dir),
                            )
                        )

                    except Exception as exc:
                        self.build_queue.put(
                            (
                                "error",
                                str(exc),
                                traceback.format_exc(),
                            )
                        )

                self.build_thread = threading.Thread(
                    target=worker,
                    daemon=True,
                )
                self.build_thread.start()

                self.after(100, self.poll_build_queue)

            except Exception as exc:
                self.is_building = False
                self.show_error(exc)

        def write_status(self, text):
            self.status_text.delete("1.0", "end")
            self.status_text.insert("end", text)

        def show_error(self, exc):
            tb = traceback.format_exc()
            self.write_status(tb)
            messagebox.showerror(self.t("error"), str(exc))

    app = App()
    app.mainloop()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a YOLO dataset by scanning image/label folder pairs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Class mapping examples:
              --classes classes.txt

            Merge rules examples:
              --merge-rules "2,3 -> 1"
              --merge-rules-file merge.txt
            """
        ),
    )

    parser.add_argument("--gui", action="store_true", help="Launch GUI")
    parser.add_argument("--parent", type=Path, help="Parent folder to scan")
    parser.add_argument("--output", type=Path, help="Output dataset folder")
    parser.add_argument("--image-dir-names", default=DEFAULT_IMAGE_DIR_NAMES)
    parser.add_argument("--label-dir-names", default=DEFAULT_LABEL_DIR_NAMES)
    parser.add_argument("--split", nargs=3, type=float, default=[0.8, 0.1, 0.1])
    parser.add_argument("--classes", type=Path)
    parser.add_argument("--classes-text", default="")
    parser.add_argument("--merge-rules", default="")
    parser.add_argument("--merge-rules-file", type=Path)
    parser.add_argument("--operation", choices=["copy", "move"], default="copy")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-unlabeled", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--resize", action="store_true", help="Resize output images")
    parser.add_argument("--resize-width", type=int, default=640)
    parser.add_argument("--resize-height", type=int, default=640)
    parser.add_argument(
        "--no-pad-black",
        action="store_true",
        help="Resize by stretching instead of keeping aspect ratio with black padding",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.gui or (not args.parent and not args.output):
        launch_gui()
        return 0

    if not args.parent or not args.output:
        print("--parent and --output are required in CLI mode. Use --gui for GUI mode.", file=sys.stderr)
        return 2

    class_text = ""

    if args.classes:
        class_text = args.classes.read_text(encoding="utf-8")

    if args.classes_text:
        class_text += "\n" + args.classes_text

    id_to_name = parse_class_mapping(class_text)

    merge_text = args.merge_rules or ""

    if args.merge_rules_file:
        merge_text += "\n" + args.merge_rules_file.read_text(encoding="utf-8")

    merge_rules = parse_merge_rules(merge_text, id_to_name)

    config = BuildConfig(
        parent_dir=args.parent,
        output_dir=args.output,
        image_dir_names=normalize_names(args.image_dir_names),
        label_dir_names=normalize_names(args.label_dir_names),
        train_ratio=args.split[0],
        val_ratio=args.split[1],
        test_ratio=args.split[2],
        id_to_name=id_to_name,
        merge_rules=merge_rules,
        operation=args.operation,
        overwrite=args.overwrite,
        include_unlabeled=args.include_unlabeled,
        seed=args.seed,
        resize_enabled=args.resize,
        resize_width=args.resize_width,
        resize_height=args.resize_height,
        resize_pad_black=not args.no_pad_black,
    )

    pairs = scan_folder_pairs(config.parent_dir, config.image_dir_names, config.label_dir_names)

    print(f"Detected {len(pairs)} folder pair(s)")
    for p in pairs:
        print(f"  {p.display()}")

    items = collect_items(pairs, include_unlabeled=config.include_unlabeled)
    stats = analyze_items(items, config.merge_rules, config.id_to_name)

    if args.analyze_only:
        print(format_stats(stats))
        return 0

    final_stats = build_dataset(config, pairs=pairs)

    print("\nBuild completed.")
    print(format_stats(final_stats))
    print(f"\nOutput: {config.output_dir.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())