import os
import pandas as pd
from tqdm import tqdm
import cv2

# CSV路径
csv_path = "yolo_src/foid_labels_v100.csv"
df = pd.read_csv(csv_path)

# 图像根目录（与测试脚本一致）
IMAGE_FOLDER = "/home/douglass/yolo_fish/yolo_fish_img"
LABEL_OUT_DIR = "/home/douglass/yolo_fish/yolo_fish_label"
os.makedirs(LABEL_OUT_DIR, exist_ok=True)

# data.yaml 中的 names 顺序（index 即为 YOLO class_id）
NAMES = [
    "Albacore", "Bigeye tuna", "Black marlin", "Blue marlin", "Brama", "Escolar",
    "Great barracuda", "Human", "Indo Pacific sailfish", "Lancetfish",
    "Long snouted lancetfish", "Mahi mahi", "Marlin", "Mola mola", "No fish",
    "Oilfish", "Opah", "Pelagic stingray", "Pomfret", "Rainbow runner",
    "Roudie scolar", "Shark", "Shortbill spearfish", "Sickle pomfret", "Skipjack tuna",
    "Snake mackerel", "Striped marlin", "Swordfish", "Thresher shark", "Tuna",
    "Unknown", "Wahoo", "Water", "Yellowfin tuna"
]

# 构建映射: label_l1(去除多余空格) -> class_id
name_to_id = {name: idx for idx, name in enumerate(NAMES)}

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def clamp(val, low, high):
    return max(low, min(high, val))

def index_images(root):
    mapping = {}
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in SUPPORTED_EXTS:
                stem = os.path.splitext(fn)[0]
                path = os.path.join(dirpath, fn)
                # 如存在同名不同路径，仅保留首次发现并提示
                if stem in mapping and mapping[stem] != path:
                    tqdm.write(f"警告: 发现重复文件名(不同路径) '{stem}', 使用 {mapping[stem]}")
                else:
                    mapping[stem] = path
    return mapping

img_index = index_images(IMAGE_FOLDER)

def process_group(item):
    img_id, group = item
    key = str(img_id)
    img_path = img_index.get(key)
    if not img_path:
        tqdm.write(f"跳过: 找不到图像 {key} 于 {IMAGE_FOLDER}")
        return

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        tqdm.write(f"跳过: 无法读取图像 {img_path}")
        return
    img_h, img_w = img.shape[:2]

    lines = []
    for _, row in group.iterrows():
        label = str(row.get("label_l1", "")).strip()
        cls = name_to_id.get(label, name_to_id.get("Unknown"))

        x_min = clamp(row["x_min"], 0, img_w)
        x_max = clamp(row["x_max"], 0, img_w)
        y_min = clamp(row["y_min"], 0, img_h)
        y_max = clamp(row["y_max"], 0, img_h)

        if x_max <= x_min or y_max <= y_min:
            continue

        x_center = (x_min + x_max) / 2 / img_w
        y_center = (y_min + y_max) / 2 / img_h
        width = (x_max - x_min) / img_w
        height = (y_max - y_min) / img_h

        lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    if lines:
        label_path = os.path.join(LABEL_OUT_DIR, f"{key}.txt")
        with open(label_path, "w") as f:
            f.write("\n".join(lines))

# 使用 map + tqdm 处理所有分组
total_groups = df["img_id"].nunique()
list(tqdm(map(process_group, df.groupby("img_id")), total=total_groups))