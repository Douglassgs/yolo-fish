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
IMAGE_FOLDER = "/home/douglass/yolo_fish/images/test"
LABEL_FOLDER = "/home/douglass/yolo_fish/labels/test"
OUTPUT_FOLDER = "./dataset/output"

COUNT_FOLDER = os.path.join(OUTPUT_FOLDER, "count_csv")
DETECT_FOLDER = os.path.join(OUTPUT_FOLDER, "detection_csv")
PRED_IMG_DIR = os.path.join(OUTPUT_FOLDER, "pred_images")
METRIC_FOLDER = os.path.join(OUTPUT_FOLDER, "metrics_csv")
MODEL_PATH = "yolo_src/wycBest.pt"

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
SAMPLE_FRACTION = 1  # 即验证 $1/3$ 数据
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

# 用于计算 AP50 的全局预测与 GT 记录
ap_gt_records = []   # 每条: {image, cls, xyxy}
ap_pred_records = [] # 每条: {image, cls, xyxy, conf}


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

        # 读取 GT，记录到 CSV，并为 AP50 保存 GT 记录
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

            ap_gt_records.append({
                "image": img_name,
                "cls": g["cls"],
                "xyxy": (x1, y1, x2, y2),
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

                ap_pred_records.append({
                    "image": img_name,
                    "cls": cid,
                    "xyxy": (x1, y1, x2, y2),
                    "conf": conf,
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


def compute_ap50_per_class(gt_records, pred_records, match_iou_thr=0.5):
    """按类别计算 AP50，返回: {cls_id: ap50} 和 mAP50。"""
    ap_per_class = {}
    eps = 1e-16

    # 根据类别分组 GT 和预测
    from collections import defaultdict

    gt_by_cls = defaultdict(list)
    for g in gt_records:
        gt_by_cls[g["cls"]].append(g)

    pred_by_cls = defaultdict(list)
    for p in pred_records:
        pred_by_cls[p["cls"]].append(p)

    for cls_id in sorted(gt_by_cls.keys()):
        gts = gt_by_cls[cls_id]
        preds = pred_by_cls.get(cls_id, [])

        # 按图像组织 GT，便于标记是否匹配
        gt_per_img = {}
        for g in gts:
            key = g["image"]
            if key not in gt_per_img:
                gt_per_img[key] = []
            gt_per_img[key].append({"xyxy": g["xyxy"], "matched": False})

        # 按置信度降序排序预测
        preds_sorted = sorted(preds, key=lambda x: x["conf"], reverse=True)
        tp = []
        fp = []
        for p in preds_sorted:
            img_gts = gt_per_img.get(p["image"], [])
            best_iou = 0.0
            best_gt = None
            for gt in img_gts:
                if gt["matched"]:
                    continue
                iou = iou_xyxy(gt["xyxy"], p["xyxy"])
                if iou >= match_iou_thr and iou > best_iou:
                    best_iou = iou
                    best_gt = gt
            if best_gt is not None:
                best_gt["matched"] = True
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)

        if not gts:
            ap_per_class[cls_id] = 0.0
            continue

        tp_cum = torch.tensor(tp).cumsum(0).float()
        fp_cum = torch.tensor(fp).cumsum(0).float()
        recalls = tp_cum / (len(gts) + eps)
        precisions = tp_cum / (tp_cum + fp_cum + eps)

        # 传统 11-point 或更细采样，这里用与 COCO 类似的数值积分
        # 在 (0,1) 上插值取 101 个点
        mrec = torch.cat((torch.tensor([0.0]), recalls, torch.tensor([1.0])))
        mpre = torch.cat((torch.tensor([0.0]), precisions, torch.tensor([0.0])))

        # 使 precision 单调不增
        for i in range(mpre.size(0) - 1, 0, -1):
            mpre[i-1] = torch.maximum(mpre[i-1], mpre[i])

        # 在所有 recall 变化点处积分
        indices = (mrec[1:] != mrec[:-1]).nonzero(as_tuple=False).squeeze()
        ap = float(((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1]).sum().item())
        ap_per_class[cls_id] = ap

    if ap_per_class:
        map50 = float(sum(ap_per_class.values()) / len(ap_per_class))
    else:
        map50 = 0.0
    return ap_per_class, map50


# ===== 整体验证集级别的指标（类似 Ultralytics 汇总） =====
total_TP = df["TP"].sum()
total_FP = df["FP"].sum()
total_FN = df["FN"].sum()

global_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) else 0
global_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) else 0
global_f1 = 2 * global_precision * global_recall / (global_precision + global_recall) if (global_precision + global_recall) else 0

# AP50 / mAP50 计算
ap50_per_class, map50 = compute_ap50_per_class(ap_gt_records, ap_pred_records, match_iou_thr=MATCH_IOU_THR)

# 全局指标保存
global_metrics = pd.DataFrame([{
    "num_images": len(df),
    "total_TP": int(total_TP),
    "total_FP": int(total_FP),
    "total_FN": int(total_FN),
    "precision": round(global_precision, 4),
    "recall": round(global_recall, 4),
    "f1": round(global_f1, 4),
    "mAP50": round(map50, 4),
    "Accuracy_Cls_FishImgs": round(acc_cls_fish, 4),
    "Accuracy_Cls_AllImgs": round(acc_cls_all, 4),
    "Accuracy_Count_FishImgs": round(acc_cnt_fish, 4),
    "Accuracy_Count_AllImgs": round(acc_cnt_all, 4),
}])
global_metrics.to_csv(
    os.path.join(METRIC_FOLDER, "global_metrics.csv"),
    index=False,
    encoding="utf-8-sig",
)

# 每类 AP50 也单独保存一份
ap_rows = []
for cid, ap in ap50_per_class.items():
    ap_rows.append({
        "cls_id": cid,
        "cls_name": class_names[cid],
        "AP50": round(ap, 4),
    })
pd.DataFrame(ap_rows).to_csv(
    os.path.join(METRIC_FOLDER, "ap50_per_class.csv"),
    index=False,
    encoding="utf-8-sig",
)

pd.DataFrame(detection_log).to_csv(f"{DETECT_FOLDER}/detections.csv", index=False, encoding='utf-8-sig')
pd.DataFrame(count_log).to_csv(f"{COUNT_FOLDER}/counts.csv", index=False, encoding='utf-8-sig')
df.to_csv(f"{METRIC_FOLDER}/per_image_eval.csv", index=False, encoding='utf-8-sig')

print("\n✅ 结果已生成！")
print(f"📄 每图指标: {METRIC_FOLDER}/per_image_eval.csv")
print(f"📄 全局指标: {METRIC_FOLDER}/global_metrics.csv")
print(f"📄 每类 AP50: {METRIC_FOLDER}/ap50_per_class.csv")
print(f"📄 检测框(GT+Pred): {DETECT_FOLDER}/detections.csv")
print(f"📄 计数统计: {COUNT_FOLDER}/counts.csv")
print(f"🖼️ 可视化图: {PRED_IMG_DIR}")
print(f"有鱼-种类准确率: {acc_cls_fish:.4f}")
print(f"全图-种类准确率: {acc_cls_all:.4f}")
print(f"有鱼-数量准确率: {acc_cnt_fish:.4f}")
print(f"全图-数量准确率: {acc_cnt_all:.4f}")
print(f"整体 Precision: {global_precision:.4f}")
print(f"整体 Recall:    {global_recall:.4f}")
print(f"整体 F1:        {global_f1:.4f}")
print(f"mAP50:          {map50:.4f}")
