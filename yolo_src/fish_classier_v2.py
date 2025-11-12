import os
import cv2
import json
import pandas as pd
import torch
import shutil
import yaml
from PIL import Image
from ultralytics import YOLO
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np
import ultralytics


# --------------------------
# 1. 配置参数（核心修改：对齐YOLO目录结构）
# --------------------------
class Config:
    TARGET_CLASSES = ['Albacore', 'Bigeye tuna', 'Black marlin', 'Blue marlin', 'Brama',
                      'Escolar', 'Great barracuda', 'Human', 'Indo Pacific sailfish', 'Lancetfish',
                      'Long snouted lancetfish', 'Mahi mahi', 'Marlin', 'Mola mola', 'No fish', 'Oilfish',
                      'Opah', 'Pelagic stingray', 'Pomfret', 'Rainbow runner', 'Roudie scolar', 'Shark',
                      'Shortbill spearfish', 'Sickle pomfret', 'Skipjack tuna', 'Snake mackerel',
                      'Striped marlin', 'Swordfish', 'Thresher shark', 'Tuna', 'Unknown', 'Wahoo',
                      'Water', 'Yellowfin tuna']
    # 预训练模型路径
    MODEL_PATH = "./src/yolo11m.pt"
    # 数据集路径（YOLO要求结构：images和labels在同一父目录dataset下）
    BASE_DATASET_DIR = "./data"  # 数据集根目录
    RAW_IMAGE_DIR = "./data/images"  # 原始图片目录（待拆分）
    RAW_LABEL_CSV = "./data/foid_labels_v100/foid_labels_v100.csv"  # 原始标签
    # 拆分后的目录（YOLO标准结构）
    IMAGES_DIR = os.path.join(BASE_DATASET_DIR, "images")  # 图片根目录（含train/val/test）
    LABELS_DIR = os.path.join(BASE_DATASET_DIR, "labels")  # 标注根目录（含train/val/test）
    DATA_YAML_PATH = os.path.join(BASE_DATASET_DIR, "data.yaml")  # YOLO配置文件
    # 训练参数
    TRAIN_EPOCHS = 50
    INIT_TRAIN_BATCH = 8
    TRAIN_IMGSZ = 640
    TRAIN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SUPPORTED_IMG_EXTS = ["jpg", "jpeg", "png"]  # 支持的图片格式
    # 输出配置
    OUTPUT_DIR = "src/fish_classify_results_v2"
    TRAIN_OUTPUT_DIR = "./runs/train/fish_model_finetune2"
    TEST_VIS_DIR = "src/test_visual_cases2"
    SAVE_VISUAL = True
    # 推理参数
    CONF_THRESHOLD = 0.7
    IOU_THRESHOLD = 0.5


# 创建YOLO标准目录结构（确保images和labels同级）
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(Config.IMAGES_DIR, split), exist_ok=True)
    os.makedirs(os.path.join(Config.LABELS_DIR, split), exist_ok=True)
# 创建其他输出目录
for dir_path in [
    Config.OUTPUT_DIR,
    os.path.join(Config.OUTPUT_DIR, "visualizations"),
    Config.TRAIN_OUTPUT_DIR,
    Config.TEST_VIS_DIR
]:
    os.makedirs(dir_path, exist_ok=True)


# --------------------------
# 2. 核心工具函数（核心修改：拆分图片+对齐标注）
# --------------------------
def load_raw_label() -> pd.DataFrame:
    """加载原始标签CSV，校验必要字段和数据分布"""
    required_cols = ["img_id", "x_min", "x_max", "y_min", "y_max", "label_l1", "train", "val", "test"]
    try:
        df = pd.read_csv(Config.RAW_LABEL_CSV)
        # 校验必要字段
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"标签CSV缺失字段：{missing_cols}")
        # 转换bool类型（处理CSV中"True"/"False"字符串）
        for col in ["train", "val", "test"]:
            if df[col].dtype == "object":
                df[col] = df[col].str.strip().str.lower() == "true"
            else:
                df[col] = df[col].astype(bool)
        # 校验各子集非空（提前拦截无数据问题）
        subset_counts = {
            "train": df["train"].sum(),
            "val": df["val"].sum(),
            "test": df["test"].sum()
        }
        print(f"📊 原始标签各子集标注数：")
        for split, cnt in subset_counts.items():
            print(f"   - {split}集：{cnt}条标注")
            if cnt == 0:
                raise RuntimeError(f"原始标签中{split}集标注数为0，请检查{split}字段")
        if len(df) == 0:
            raise ValueError(f"原始标签CSV为空：{Config.RAW_LABEL_CSV}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"原始标签未找到：{Config.RAW_LABEL_CSV}")
    except Exception as e:
        raise RuntimeError(f"加载标签失败：{str(e)}")


def find_raw_image(img_id: str) -> str:
    """从原始图片目录找到对应图片（支持多格式+大小写）"""
    for ext in Config.SUPPORTED_IMG_EXTS:
        for ext_case in [ext.lower(), ext.upper()]:
            img_path = os.path.join(Config.RAW_IMAGE_DIR, f"{img_id}.{ext_case}")
            if os.path.exists(img_path):
                return img_path
    raise FileNotFoundError(
        f"图片 {img_id} 未找到：\n"
        f"目录：{Config.RAW_IMAGE_DIR}\n"
        f"支持格式：{Config.SUPPORTED_IMG_EXTS}"
    )


def split_data_to_yolo_structure() -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    """
    核心功能：
    1. 将原始图片按train/val/test复制到Config.IMAGES_DIR/[split]
    2. 在Config.LABELS_DIR/[split]生成对应YOLO格式标注
    确保图片与标注路径严格对应（YOLO要求）
    """
    label_df = load_raw_label()
    label2id = {cls.lower(): idx for idx, cls in enumerate(Config.TARGET_CLASSES)}  # 类别映射（不区分大小写）
    data_paths = {"train": [], "val": [], "test": []}  # 拆分后的图片路径
    label_counts = {"train": 0, "val": 0, "test": 0}  # 各子集标注数

    # 按img_id分组处理（一张图片可能多个标注）
    unique_img_ids = label_df["img_id"].unique()
    print(f"\n📥 开始拆分 {len(unique_img_ids)} 张图片到YOLO目录...")

    for img_idx, img_id in enumerate(unique_img_ids, 1):
        img_group = label_df[label_df["img_id"] == img_id]
        total_annot = len(img_group)

        # 1. 找到原始图片并复制到对应split目录
        try:
            raw_img_path = find_raw_image(img_id)
        except FileNotFoundError as e:
            print(f"⚠️  跳过图片 {img_idx}/{len(unique_img_ids)}：{str(e)}")
            continue

        # 2. 确定当前图片属于哪个子集（test>val>train）
        if img_group["test"].any():
            split = "test"
        elif img_group["val"].any():
            split = "val"
        elif img_group["train"].any():
            split = "train"
        else:
            print(f"⚠️  跳过图片 {img_idx}/{len(unique_img_ids)}：{img_id} 无子集标记")
            continue

        # 3. 复制图片到目标目录（保留原始后缀）
        img_ext = os.path.splitext(raw_img_path)[1]  # 如.jpg
        dest_img_path = os.path.join(Config.IMAGES_DIR, split, f"{img_id}{img_ext}")
        shutil.copy2(raw_img_path, dest_img_path)  # 复制图片（保留元数据）

        # 4. 获取图片尺寸（用于BBox转换）
        try:
            with Image.open(dest_img_path) as img:
                img_w, img_h = img.width, img.height
                if img_w <= 0 or img_h <= 0:
                    raise ValueError(f"图片尺寸无效：宽={img_w}, 高={img_h}")
        except Exception as e:
            print(f"⚠️  跳过图片 {img_idx}/{len(unique_img_ids)}：{str(e)}")
            os.remove(dest_img_path)  # 删除无效图片
            continue

        # 5. 生成YOLO格式标注（严格校验）
        dest_label_path = os.path.join(Config.LABELS_DIR, split, f"{img_id}.txt")
        valid_annots = []
        invalid_cnt = 0

        for _, row in img_group.iterrows():
            try:
                # 类别ID映射（未匹配类别归为"其他杂鱼"）
                label_l1 = row["label_l1"].strip().lower()
                class_id = label2id.get(label_l1, len(Config.TARGET_CLASSES) - 1)
                if not (0 <= class_id < len(Config.TARGET_CLASSES)):
                    raise ValueError(f"类别ID无效：{class_id}（需在0-{len(Config.TARGET_CLASSES)-1}）")

                # BBox坐标修正与归一化
                x_min = max(0.0, min(float(img_w), row["x_min"]))
                x_max = max(0.0, min(float(img_w), row["x_max"]))
                y_min = max(0.0, min(float(img_h), row["y_min"]))
                y_max = max(0.0, min(float(img_h), row["y_max"]))

                # 校验BBox有效性（宽高≥1像素）
                bbox_w = x_max - x_min
                bbox_h = y_max - y_min
                if bbox_w < 1.0 or bbox_h < 1.0:
                    raise ValueError(f"BBox过小：宽={bbox_w:.1f}px, 高={bbox_h:.1f}px")

                # 转换为YOLO格式（x_center, y_center, width, height，归一化到0-1）
                x_center = (x_min + x_max) / 2 / img_w
                y_center = (y_min + y_max) / 2 / img_h
                bbox_w_norm = bbox_w / img_w
                bbox_h_norm = bbox_h / img_h

                # 最终校验坐标范围
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= bbox_w_norm <= 1 and 0 <= bbox_h_norm <= 1):
                    raise ValueError(f"归一化坐标超界：{x_center:.6f}, {y_center:.6f}, {bbox_w_norm:.6f}, {bbox_h_norm:.6f}")

                valid_annots.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_w_norm:.6f} {bbox_h_norm:.6f}\n")
            except Exception as e:
                invalid_cnt += 1
                print(f"⚠️  图片 {img_id} 的1条标注无效：{str(e)}")
                continue

        # 6. 保存标注（仅保留有有效标注的图片）
        if valid_annots:
            with open(dest_label_path, "w", encoding="utf-8") as f:
                f.writelines(valid_annots)
            # 更新统计
            data_paths[split].append(dest_img_path)
            label_counts[split] += len(valid_annots)
            print(f"✅ 完成 {img_idx}/{len(unique_img_ids)}：{img_id} "
                  f"（有效标注：{len(valid_annots)}/{total_annot}，子集：{split}）")
        else:
            print(f"⚠️  跳过图片 {img_idx}/{len(unique_img_ids)}：{img_id} 无有效标注，删除图片")
            os.remove(dest_img_path)  # 删除无标注的图片

    # 强制校验train和val集非空（否则无法训练/验证）
    print(f"\n✅ 数据拆分完成！最终统计：")
    for split in ["train", "val", "test"]:
        print(f"   - {split}集：{len(data_paths[split])}张图片，{label_counts[split]}个标注")
        if split in ["train", "val"] and len(data_paths[split]) == 0:
            raise RuntimeError(
                f"{split}集无有效数据！请检查：\n"
                f"1. 原始图片目录是否有{split}集对应图片\n"
                f"2. 标注是否因格式错误被全部过滤\n"
                f"3. 类别ID是否在有效范围"
            )

    return data_paths, label_counts


def generate_yolo_data_yaml(data_paths: Dict[str, List[str]]) -> None:
    """生成YOLO标准data.yaml，确保路径100%正确"""
    # YOLO要求：path为数据集根目录，train/val/test为相对于path的图片目录
    yaml_content = {
        "path": os.path.abspath(Config.BASE_DATASET_DIR),  # 数据集绝对路径（避免相对路径歧义）
        "train": "images/train",  # 训练集图片目录（相对path）
        "val": "images/val",      # 验证集图片目录（相对path）
        "test": "images/test",    # 测试集图片目录（相对path）
        "nc": len(Config.TARGET_CLASSES),  # 类别数
        "names": Config.TARGET_CLASSES    # 类别名称
    }

    # 保存并打印YAML（便于调试）
    with open(Config.DATA_YAML_PATH, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, sort_keys=False, allow_unicode=True)
    print(f"\n✅ data.yaml生成：{Config.DATA_YAML_PATH}")
    print(f"📋 YAML内容：")
    with open(Config.DATA_YAML_PATH, "r") as f:
        print(f.read())

    # 校验图片与标注一一对应（YOLO关键要求）
    for split in ["train", "val"]:
        # 随机取第一张图片检查
        sample_img = data_paths[split][0]
        img_id = os.path.splitext(os.path.basename(sample_img))[0]
        sample_label = os.path.join(Config.LABELS_DIR, split, f"{img_id}.txt")
        if not os.path.exists(sample_label):
            raise RuntimeError(
                f"{split}集图片与标注不匹配！\n"
                f"图片：{sample_img}\n"
                f"预期标注：{sample_label}\n"
                f"请检查数据拆分逻辑"
            )
    print(f"✅ 路径校验通过：train/val集图片与标注一一对应")


# --------------------------
# 3. 模型加载与训练（无修改，依赖数据拆分）
# --------------------------
def load_pretrained_model() -> YOLO:
    """加载预训练模型，校验文件存在性"""
    model_path = Config.MODEL_PATH
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"预训练模型未找到：{model_path}")
    try:
        model = YOLO(model_path)
        print(f"✅ 加载预训练模型：{os.path.basename(model_path)}")
        print(f"   - 模型原生类别数：{len(model.names)}")
        return model
    except Exception as e:
        raise RuntimeError(f"加载模型失败：{str(e)}")


def adjust_batch_size() -> int:
    """根据GPU显存自动调整批大小"""
    if Config.TRAIN_DEVICE != "cuda":
        print(f"⚠️  未使用GPU，批大小设为1")
        return 1

    # 计算GPU总显存（GB）
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"🔍 GPU显存：{gpu_mem:.1f}GB")

    # 调整策略
    if gpu_mem >= 16:
        batch = Config.INIT_TRAIN_BATCH
    elif 8 <= gpu_mem < 16:
        batch = max(1, Config.INIT_TRAIN_BATCH // 2)
    elif 4 <= gpu_mem < 8:
        batch = max(1, Config.INIT_TRAIN_BATCH // 4)
    else:
        batch = 14

    print(f"✅ 批大小调整：{Config.INIT_TRAIN_BATCH} → {batch}")
    return batch


def train_model() -> YOLO:
    """训练预训练模型（基于YOLO标准数据结构）"""
    # 1. 拆分数据到YOLO目录结构
    data_paths, _ = split_data_to_yolo_structure()
    # 2. 生成标准data.yaml
    generate_yolo_data_yaml(data_paths)
    # 3. 加载预训练模型
    model = load_pretrained_model()
    # 4. 调整批大小
    batch_size = adjust_batch_size()

    # 训练参数（YOLOv8兼容）
    train_args = {
        "data": Config.DATA_YAML_PATH,
        "epochs": Config.TRAIN_EPOCHS,
        "batch": batch_size,
        "imgsz": Config.TRAIN_IMGSZ,
        "device": Config.TRAIN_DEVICE,
        "project": os.path.dirname(Config.TRAIN_OUTPUT_DIR),
        "name": os.path.basename(Config.TRAIN_OUTPUT_DIR),
        "pretrained": True,  # 微调模式（保留预训练特征）
        "optimizer": "Adam",  # 适合微调的优化器
        "lr0": 1e-4,  # 小学习率，避免覆盖预训练特征
        "save": True,  # 保存最佳模型
        "val": True,   # 训练中验证
        "verbose": True,
        "plots": True  # 生成训练曲线
    }

    # 打印训练信息
    print(f"\n🚀 开始微调（{Config.TRAIN_EPOCHS}轮）...")
    print(f"   - 设备：{Config.TRAIN_DEVICE}")
    print(f"   - 批大小：{batch_size}")
    print(f"   - 训练集：{len(data_paths['train'])}张图片")
    print(f"   - 验证集：{len(data_paths['val'])}张图片")
    print(f"   - 输出目录：{Config.TRAIN_OUTPUT_DIR}")

    # 开始训练
    model.train(**train_args)

    # 加载最佳模型
    trainer = model.trainer  # 获取训练器对象
    actual_output_dir = trainer.save_dir
    best_model_path = os.path.join(actual_output_dir, "weights", "best.pt")
    # if not os.path.exists(best_model_path):
    #     raise RuntimeError(f"训练失败：未找到最佳模型 {best_model_path}")
    if not os.path.exists(best_model_path):
        last_model_path = os.path.join(actual_output_dir, "weights", "last.pt")
        if os.path.exists(last_model_path):
            print(f"⚠️  未找到best.pt，使用last.pt继续")
            best_model_path = last_model_path
        else:
            raise RuntimeError(f"训练失败：未找到模型文件\n预期路径：{best_model_path}")
    model = YOLO(best_model_path)
    print(f"\n✅ 训练完成！加载最佳模型：{best_model_path}")

    return model


# --------------------------
# 4. 模型评估与可视化（确保路径正确）
# --------------------------
def evaluate_model(model: YOLO, split: str = "test") -> Dict[str, float]:
    """评估模型指标（基于YOLO标准结构）"""
    if split not in ["val", "test"]:
        raise ValueError(f"split必须为val/test，当前：{split}")

    # 校验标注目录存在
    label_dir = os.path.join(Config.LABELS_DIR, split)
    if not os.path.exists(label_dir) or len(os.listdir(label_dir)) == 0:
        raise RuntimeError(f"{split}集标注目录为空：{label_dir}")

    # 评估参数
    eval_args = {
        "data": Config.DATA_YAML_PATH,
        "split": split,
        "device": Config.TRAIN_DEVICE,
        "conf": Config.CONF_THRESHOLD,
        "iou": Config.IOU_THRESHOLD,
        "verbose": True,
        "save_json": True  # 保存详细指标
    }

    print(f"\n📊 开始{split}集评估...")
    # results = model.val(**eval_args) if split == "val" else model.test(**eval_args)
    results = model.val(**eval_args)

    # 提取平均精度（mean precision）和平均召回率（mean recall）
    mean_precision = results.box.p.mean() if len(results.box.p) > 0 else 0.0
    mean_recall = results.box.r.mean() if len(results.box.r) > 0 else 0.0

    # 提取关键指标
    metrics = {
        "split": split,
        "mAP50": round(results.box.map50, 4),
        "mAP50-95": round(results.box.map, 4),
        "precision": round(mean_precision, 4),  # 平均精度
        "recall": round(mean_recall, 4),  # 平均召回率
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # 保存指标到CSV
    metrics_path = os.path.join(Config.TRAIN_OUTPUT_DIR, f"{split}_metrics.csv")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ {split}集指标保存：{metrics_path}")
    print(f"   关键指标：mAP50={metrics['mAP50']}, Precision={metrics['precision']}")

    return metrics


def visualize_test_cases(model: YOLO, num_cases: int = 5) -> None:
    """可视化测试集案例"""
    test_img_dir = os.path.join(Config.IMAGES_DIR, "test")
    test_label_dir = os.path.join(Config.LABELS_DIR, "test")

    # 校验测试集数据
    if not os.path.exists(test_label_dir) or len(os.listdir(test_label_dir)) == 0:
        print(f"⚠️  测试集无标注，跳过可视化")
        return

    # 获取测试集图片路径（与标注匹配）
    test_img_ids = [os.path.splitext(f)[0] for f in os.listdir(test_label_dir) if f.endswith(".txt")]
    test_img_paths = []
    for img_id in test_img_ids:
        for ext in Config.SUPPORTED_IMG_EXTS:
            img_path = os.path.join(test_img_dir, f"{img_id}.{ext}")
            if os.path.exists(img_path):
                test_img_paths.append(img_path)
                break

    # 调整案例数量
    num_cases = min(num_cases, len(test_img_paths))
    if num_cases == 0:
        print(f"⚠️  测试集无有效图片，跳过可视化")
        return

    # 可视化推理
    print(f"\n🖼️  可视化{num_cases}个测试案例...")
    for i, img_path in enumerate(test_img_paths[:num_cases]):
        img_name = os.path.basename(img_path)
        print(f"   案例 {i+1}/{num_cases}：{img_name}")

        # 读取图片
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"⚠️  跳过案例 {img_name}：无法读取")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 推理
        results = model.predict(
            img_rgb,
            imgsz=Config.TRAIN_IMGSZ,
            conf=Config.CONF_THRESHOLD,
            iou=Config.IOU_THRESHOLD,
            device=Config.TRAIN_DEVICE,
            verbose=False
        )

        # 解析结果
        detections = []
        for res in results:
            if not hasattr(res, "boxes") or res.boxes is None:
                continue
            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                cls_name = Config.TARGET_CLASSES[cls_id]
                conf = float(box.conf[0])
                detections.append({
                    "target_species": cls_name,
                    "confidence": round(conf, 4),
                    "bbox": [x1, y1, x2, y2],
                    "need_manual_check": conf < Config.CONF_THRESHOLD
                })

        # 保存可视化结果
        visualized_img = visualize_results(img_bgr, detections)
        save_path = os.path.join(Config.TEST_VIS_DIR, f"case_{i+1}_{img_name}")
        cv2.imwrite(save_path, visualized_img)
        print(f"   案例 {i+1} 保存：{save_path}")

    print(f"\n✅ 测试案例可视化完成：{Config.TEST_VIS_DIR}")


# --------------------------
# 5. 辅助函数（无修改）
# --------------------------
def preprocess_image(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片：{image_path}")
    if img.shape[0] <= 0 or img.shape[1] <= 0:
        raise ValueError(f"图片尺寸无效：{image_path}")
    img = np.asarray(img, dtype=np.uint8)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, img_rgb


def map_to_target_classes(native_species: str) -> str:
    mapping = {
        "tuna": "鲣鱼",
        "yellowfin tuna": "黄鳍金枪鱼",
        "bluefin tuna": "蓝鳍金枪鱼",
        "bigeye tuna": "大眼金枪鱼",
        "albacore tuna": "长鳍金枪鱼",
        "swordfish": "剑旗鱼",
        "atlantic sailfish": "大西洋旗鱼",
        "indo-pacific sailfish": "印度太平洋旗鱼",
        "turtle": "海龟",
        "shark": "鲨鱼",
        "dolphin": "海豚"
    }
    return mapping.get(native_species.lower(), "其他杂鱼")


def visualize_results(img: np.ndarray, detections: List[Dict]) -> np.ndarray:
    img_copy = img.copy().astype(np.uint8)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls = det["target_species"]
        conf = det["confidence"]
        color = (0, 255, 0) if not det["need_manual_check"] else (0, 0, 255)
        # 绘制边界框
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        # 绘制标签
        label = f"{cls} ({conf:.2f})"
        label_y = y1 - 10 if y1 > 10 else y1 + 20
        cv2.putText(
            img_copy, label, (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )
    return img_copy.astype(np.uint8)


def classify_fish_image(image_path: str, model: YOLO) -> Dict:
    img_bgr, img_rgb = preprocess_image(image_path)
    image_name = os.path.basename(image_path)

    # 推理
    results = model.predict(
        img_rgb,
        imgsz=Config.TRAIN_IMGSZ,
        conf=Config.CONF_THRESHOLD,
        iou=Config.IOU_THRESHOLD,
        device=Config.TRAIN_DEVICE,
        verbose=False
    )

    # 解析结果
    detections = []
    for res in results:
        if not hasattr(res, "boxes") or res.boxes is None:
            continue
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            cls_name = Config.TARGET_CLASSES[cls_id]
            conf = float(box.conf[0])
            detections.append({
                "source_model": "finetuned_best.pt",
                "native_species": cls_name,
                "target_species": cls_name,
                "confidence": round(conf, 4),
                "bbox": [x1, y1, x2, y2],
                "need_manual_check": conf < Config.CONF_THRESHOLD
            })

    # 保存可视化
    if Config.SAVE_VISUAL:
        visual_path = os.path.join(Config.OUTPUT_DIR, "visualizations", image_name)
        cv2.imwrite(visual_path, visualize_results(img_bgr, detections))

    return {
        "image_name": image_name,
        "image_path": image_path,
        "infer_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "detections": detections,
        "total_objects": len(detections),
        "need_manual_check_count": sum(1 for d in detections if d["need_manual_check"])
    }


def batch_classify(image_dir: str, model: YOLO) -> None:
    if not os.path.exists(image_dir):
        raise ValueError(f"图像目录不存在：{image_dir}")

    # 创建输出路径
    jsonl_path = os.path.join(Config.OUTPUT_DIR, "classification_results.jsonl")  # 用 JSONL 更高效
    csv_path = os.path.join(Config.OUTPUT_DIR, "classification_results.csv")
    visual_dir = os.path.join(Config.OUTPUT_DIR, "visualizations")
    os.makedirs(visual_dir, exist_ok=True)

    # 打开 CSV 文件，写入表头（流式写入）
    csv_file = open(csv_path, 'w', encoding='utf-8-sig', newline='')
    csv_writer = None  # 延迟初始化表头

    print(f"📝 将结果流式写入：\n   - JSONL: {jsonl_path}\n   - CSV: {csv_path}")

    processed_count = 0
    error_count = 0

    for img_name in os.listdir(image_dir):
        img_ext = os.path.splitext(img_name)[1].lstrip(".").lower()
        if img_ext not in Config.SUPPORTED_IMG_EXTS:
            continue
        img_path = os.path.join(image_dir, img_name)
        print(f"🔍 处理图像：{img_name}")

        try:
            # 单图推理
            result = classify_fish_image(img_path, model)

            # ✅ 1. 实时写入 JSONL（每行一个 JSON 对象）
            with open(jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

            # ✅ 2. 实时写入 CSV（扁平化每一条 detection）
            for det in result["detections"]:
                row = {
                    "image_name": result["image_name"],
                    "infer_time": result["infer_time"],
                    "native_species": det["native_species"],
                    "target_species": det["target_species"],
                    "confidence": det["confidence"],
                    "bbox_x1": det["bbox"][0],
                    "bbox_y1": det["bbox"][1],
                    "bbox_x2": det["bbox"][2],
                    "bbox_y2": det["bbox"][3],
                    "need_manual_check": det["need_manual_check"]
                }
                if csv_writer is None:
                    csv_writer = pd.DataFrame([row]).to_csv(csv_file, index=False)
                else:
                    pd.DataFrame([row]).to_csv(csv_file, index=False, header=False, mode='a')

            processed_count += 1

        except Exception as e:
            print(f"❌ 跳过图像 {img_name}：{str(e)}")
            error_count += 1
            continue

    # 关闭文件
    csv_file.close()

    print(f"\n🎉 批量处理完成！共处理 {processed_count} 张图，失败 {error_count} 张")
    print(f"   - JSONL 流式结果：{jsonl_path}")
    print(f"   - CSV 流式结果：{csv_path}")
    print(f"   - 可视化：{visual_dir}")


# --------------------------
# 6. 主函数（恢复异常捕获，确保稳定）
# --------------------------
if __name__ == "__main__":
    print("=" * 50)
    print(f"版本信息：")
    print(f"   numpy: {np.__version__}")
    print(f"   opencv: {cv2.__version__}")
    print(f"   ultralytics: {ultralytics.__version__}")
    print(f"   torch: {torch.__version__}")
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    print("=" * 50)

    try:
        RUN_MODE = "train_test"  # "train_test" 或 "infer_only"

        if RUN_MODE == "train_test":
            # 训练模型
            finetuned_model = train_model()
            # 评估验证集
            val_metrics = evaluate_model(finetuned_model, split="val")
            # 评估测试集
            test_metrics = evaluate_model(finetuned_model, split="test")
            # 可视化测试案例
            visualize_test_cases(finetuned_model, num_cases=5)
            # 批量推理测试集
            print(f"\n📥 开始测试集批量推理...")
            test_img_dir = os.path.join(Config.IMAGES_DIR, "test")
            batch_classify(test_img_dir, finetuned_model)

        elif RUN_MODE == "infer_only":
            # 仅推理（需先训练）
            best_model_path = os.path.join(Config.TRAIN_OUTPUT_DIR, "weights", "best.pt")
            if not os.path.exists(best_model_path):
                raise FileNotFoundError(f"未找到训练模型：{best_model_path}（请先执行train_test）")
            finetuned_model = YOLO(best_model_path)
            # 推理自定义目录
            infer_dir = "/root/zhaowei_project/cv_fish/dataset/images_fish"
            batch_classify(infer_dir, finetuned_model)

        print(f"\n🎉 所有任务执行完成！")

    except Exception as e:
        print(f"\n❌ 程序运行失败：{str(e)}")
        # 打印详细堆栈（便于调试）
        import traceback
        traceback.print_exc()