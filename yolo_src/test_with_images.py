import os
import cv2
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
import torch
import math
import random

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor, as_completed


# ======================
# 路径配置
# ======================
IMAGE_FOLDER = "test"
LABEL_FOLDER = "/home/douglass/yolo_fish/yolo_fish_label"
OUTPUT_FOLDER = "./dataset/output"

COUNT_FOLDER = os.path.join(OUTPUT_FOLDER, "count_csv")
DETECT_FOLDER = os.path.join(OUTPUT_FOLDER, "detection_csv")
PRED_IMG_DIR = os.path.join(OUTPUT_FOLDER, "pred_images")
METRIC_FOLDER = os.path.join(OUTPUT_FOLDER, "metrics_csv")
MODEL_PATH = "yolo_src/best_final.pt"

CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
IMG_SIZE = 640
"""
并行/向量化说明：
- 使用 PyTorch DataLoader 自定义数据集进行批量加载，充分利用多线程/多进程 I/O 与 GPU/CPU 并行计算能力。
- 将批次中的图像列表传递给 Ultralytics YOLO，一次性完成推理，保持原有评估与日志逻辑不变。
"""

# 按设备选择合适的 batch 大小（避免显存/内存溢出）
BATCH_SIZE = 32
NUM_WORKERS = 8
PIN_MEMORY = torch.cuda.is_available()
# I/O 优化参数（可按机器调整）
PREFETCH_FACTOR = 4  # 每个 worker 预取的批次数
PERSISTENT_WORKERS = True  # 复用 DataLoader worker，减少重复启动开销
CACHE_IMAGES = False  # 如内存充足可设为 True，将所有图像预加载到内存
CACHE_THREADS = min(8, max(1, NUM_WORKERS))
CV2_NUM_THREADS = 0  # 避免与 DataLoader workers 线程过度竞争

MATCH_IOU_THR = 0.5

FILTER_OUT_CLASSES = ["Human", "No fish", "Water", "Unknown"]
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 使用设备: {device}")
try:
    cv2.setNumThreads(CV2_NUM_THREADS)
except Exception:
    pass

# 只验证数据集的 1/3（可调整为 0~1 之间的小数），设置随机种子保证可复现
SAMPLE_FRACTION = 1/3  # 即验证 $1/3$ 数据
RANDOM_SEED = 42

# ======================
# 模型加载
# ======================
model = YOLO(MODEL_PATH)
model.to(device)
class_names = model.names

valid_class_ids = [cid for cid, name in class_names.items() if name not in FILTER_OUT_CLASSES]
print("✅ 保留鱼类类别：", [class_names[i] for i in valid_class_ids])

os.makedirs(COUNT_FOLDER, exist_ok=True)
os.makedirs(DETECT_FOLDER, exist_ok=True)
os.makedirs(PRED_IMG_DIR, exist_ok=True)
os.makedirs(METRIC_FOLDER, exist_ok=True)

image_paths = [p for p in Path(IMAGE_FOLDER).glob("*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"]]
if not image_paths:
    raise FileNotFoundError("❌ 未找到测试图片")
# 随机抽样 1/3 的图片进行验证
if SAMPLE_FRACTION and 0 < SAMPLE_FRACTION < 1:
    total_images = len(image_paths)
    random.seed(RANDOM_SEED)
    sample_n = max(1, int(total_images * SAMPLE_FRACTION))
    image_paths = random.sample(image_paths, sample_n)
    print(f"🔍 采样用于验证: {sample_n}/{total_images} ({SAMPLE_FRACTION:.2f})")

# ======================
# 工具函数
# ======================
def yolo_txt_to_xyxy(line, img_w, img_h):
    parts = line.strip().split()
    cid = int(parts[0])
    cx, cy, w, h = map(float, parts[1:5])
    cx *= img_w; cy *= img_h; w *= img_w; h *= img_h
    x1 = max(0, cx - w / 2); y1 = max(0, cy - h / 2)
    x2 = min(img_w - 1, cx + w / 2); y2 = min(img_h - 1, cy + h / 2)
    return cid, int(x1), int(y1), int(x2), int(y2)


def load_gt_boxes(label_file, img_w, img_h):
    gts = []
    if not os.path.exists(label_file):
        return gts
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            cid, x1, y1, x2, y2 = yolo_txt_to_xyxy(line, img_w, img_h)
            if cid in valid_class_ids:
                gts.append({"cls": cid, "xyxy": (x1, y1, x2, y2), "matched": False})
    return gts


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1 + 1); ih = max(0, inter_y2 - inter_y1 + 1)
    inter = iw * ih
    area_a = (ax2-ax1+1)*(ay2-ay1+1); area_b = (bx2-bx1+1)*(by2-by1+1)
    return inter / (area_a + area_b - inter) if (area_a + area_b - inter) > 0 else 0.0


# ======================
# 容器
# ======================
detection_log = []
count_log = []
per_image_log = []


# ======================
# 自定义 Dataset 与 DataLoader
class ImageDataset(Dataset):
    def __init__(self, paths, cache=False, cache_threads=2):
        self.paths = list(paths)
        self.cache_enabled = bool(cache)
        self.cache = None
        if self.cache_enabled:
            self.cache = {}

            def _load_one(p):
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img is None:
                    return None
                h, w = img.shape[:2]
                return (
                    str(p),
                    {
                        "img": img,
                        "name": Path(p).name,
                        "path": p,
                        "w": w,
                        "h": h,
                        "label_file": os.path.join(LABEL_FOLDER, Path(p).stem + ".txt"),
                        "orig_img": img,
                    },
                )

            with ThreadPoolExecutor(max_workers=max(1, cache_threads)) as ex:
                futures = [ex.submit(_load_one, p) for p in self.paths]
                for fut in as_completed(futures):
                    res = fut.result()
                    if res is None:
                        continue
                    k, v = res
                    self.cache[k] = v

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        key = str(p)
        if self.cache_enabled and self.cache is not None and key in self.cache:
            return self.cache[key]

        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            # 返回 None，让 collate 过滤
            return None
        h, w = img.shape[:2]
        label_file = os.path.join(LABEL_FOLDER, Path(p).stem + ".txt")
        return {
            "img": img,
            "name": Path(p).name,
            "path": p,
            "w": w,
            "h": h,
            "label_file": label_file,
            "orig_img": img,
        }


def collate_fn(batch):
    # 过滤空样本，保持列表形式，避免张量堆叠（不同尺寸）
    return [b for b in batch if b is not None]


# ======================
# 使用 DataLoader 批量并行推理（向量化）
# ======================
dataset = ImageDataset(image_paths, cache=CACHE_IMAGES, cache_threads=CACHE_THREADS)
# 仅在有 worker 时设置 prefetch_factor/persistent_workers
loader_kwargs = dict(
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn,
    pin_memory=PIN_MEMORY,
)
if NUM_WORKERS > 0:
    loader_kwargs.update({
        "prefetch_factor": PREFETCH_FACTOR,
        "persistent_workers": PERSISTENT_WORKERS,
    })
loader = DataLoader(dataset, **loader_kwargs)

for batch in tqdm(loader):
    if not batch:
        continue

    imgs = [item["img"] for item in batch]

    # ===== 批量推理 =====
    results = model.predict(
        imgs,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        classes=valid_class_ids,
        verbose=False,
    )

    # 逐结果处理（保持与原逻辑一致）
    for item, res in zip(batch, results):
        img_name = item["name"]
        orig_img = item["orig_img"]
        h, w = item["h"], item["w"]

        # 读取 GT，记录到 CSV
        gt_boxes = load_gt_boxes(item["label_file"], w, h)
        for g in gt_boxes:
            x1, y1, x2, y2 = g["xyxy"]
            detection_log.append({
                "image": img_name,
                "mode": "gt",
                "species": class_names[g["cls"]],
                "confidence": "-",
                "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2
            })

        # 预测框，记录到 CSV + 画框
        pred_items = []
        draw = orig_img.copy()
        boxes = res.boxes
        if boxes is not None:
            for box in boxes:
                cid = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                pred_items.append({"cls": cid, "xyxy": (x1, y1, x2, y2), "conf": conf, "matched": False})

                detection_log.append({
                    "image": img_name,
                    "mode": "pred",
                    "species": class_names[cid],
                    "confidence": round(conf, 4),
                    "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2
                })

                cv2.rectangle(draw, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(draw, f"{class_names[cid]} {conf:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.imwrite(os.path.join(PRED_IMG_DIR, img_name), draw)

        pred_classes = [p["cls"] for p in pred_items]
        gt_classes = [g["cls"] for g in gt_boxes]

        # 数量统计
        for cid in valid_class_ids:
            count_log.append({
                "image": img_name,
                "species": class_names[cid],
                "gt_count": gt_classes.count(cid),
                "pred_count": pred_classes.count(cid)
            })

        # ✅ TP/FP/FN 计算
        img_TP = img_FP = img_FN = 0
        for gi, g in enumerate(gt_boxes):
            best_pi = -1; best_iou = 0
            for pi, p in enumerate(pred_items):
                if p["matched"] or p["cls"] != g["cls"]:
                    continue
                iou = iou_xyxy(g["xyxy"], p["xyxy"])
                if iou >= MATCH_IOU_THR and iou > best_iou:
                    best_iou = iou; best_pi = pi
            if best_pi >= 0:
                img_TP += 1; pred_items[best_pi]["matched"] = True
            else:
                img_FN += 1
        for p in pred_items:
            if not p["matched"]:
                img_FP += 1

        precision = img_TP / (img_TP + img_FP) if (img_TP + img_FP) else 0
        recall = img_TP / (img_TP + img_FN) if (img_TP + img_FN) else 0
        accuracy = img_TP / (img_TP + img_FP + img_FN) if (img_TP + img_FP + img_FN) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

        gt_species = set([class_names[c] for c in gt_classes])
        pred_species = set([class_names[c] for c in pred_classes])
        has_fish = 1 if gt_species else 0
        class_correct = 1 if gt_species == pred_species else 0
        count_correct = 1 if len(gt_classes) == len(pred_classes) else 0

        per_image_log.append({
            "image": img_name, "has_fish": has_fish,
            "GT_species": ",".join(sorted(gt_species)) if has_fish else "None",
            "Pred_species": ",".join(sorted(pred_species)) if pred_species else "None",
            "GT_count": len(gt_classes), "Pred_count": len(pred_classes),
            "Class_correct": class_correct, "Count_correct": count_correct,
            "TP": img_TP, "FP": img_FP, "FN": img_FN,
            "precision": round(precision, 4), "recall": round(recall, 4),
            "accuracy": round(accuracy, 4), "f1": round(f1, 4)
        })


# ======================
# 汇总 & 保存
# ======================
df = pd.DataFrame(per_image_log)
acc_cls_fish = df[df["has_fish"] == 1]["Class_correct"].mean()
acc_cls_all = df["Class_correct"].mean()
acc_cnt_fish = df[df["has_fish"] == 1]["Count_correct"].mean()
acc_cnt_all = df["Count_correct"].mean()

df["Accuracy_Cls_FishImgs"] = acc_cls_fish
df["Accuracy_Cls_AllImgs"] = acc_cls_all
df["Accuracy_Count_FishImgs"] = acc_cnt_fish
df["Accuracy_Count_AllImgs"] = acc_cnt_all

pd.DataFrame(detection_log).to_csv(f"{DETECT_FOLDER}/detections.csv", index=False, encoding='utf-8-sig')
pd.DataFrame(count_log).to_csv(f"{COUNT_FOLDER}/counts.csv", index=False, encoding='utf-8-sig')
df.to_csv(f"{METRIC_FOLDER}/per_image_eval.csv", index=False, encoding='utf-8-sig')

print("\n✅ 结果已生成！")
print(f"📄 每图指标: {METRIC_FOLDER}/per_image_eval.csv")
print(f"📄 检测框(GT+Pred): {DETECT_FOLDER}/detections.csv")
print(f"📄 计数统计: {COUNT_FOLDER}/counts.csv")
print(f"🖼️ 可视化图: {PRED_IMG_DIR}")
print(f"有鱼-种类准确率: {acc_cls_fish:.4f}")
print(f"全图-种类准确率: {acc_cls_all:.4f}")
print(f"有鱼-数量准确率: {acc_cnt_fish:.4f}")
print(f"全图-数量准确率: {acc_cnt_all:.4f}")
