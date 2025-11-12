import os
import yaml
import torch
from ultralytics import YOLO
from datetime import datetime

# --------------------------
# 1. 配置参数
# --------------------------
class Config:
    TARGET_CLASSES = ['Albacore', 'Bigeye tuna', 'Black marlin', 'Blue marlin', 'Brama',
                      'Escolar', 'Great barracuda', 'Human', 'Indo Pacific sailfish', 'Lancetfish',
                      'Long snouted lancetfish', 'Mahi mahi', 'Marlin', 'Mola mola', 'No fish', 'Oilfish',
                      'Opah', 'Pelagic stingray', 'Pomfret', 'Rainbow runner', 'Roudie scolar', 'Shark',
                      'Shortbill spearfish', 'Sickle pomfret', 'Skipjack tuna', 'Snake mackerel',
                      'Striped marlin', 'Swordfish', 'Thresher shark', 'Tuna', 'Unknown', 'Wahoo',
                      'Water', 'Yellowfin tuna']

    MODEL_PATH = "./Model/FishInv.pt"
    BASE_DATASET_DIR = "./data"
    DATA_YAML_PATH = os.path.join(BASE_DATASET_DIR, "data.yaml")

    TRAIN_EPOCHS = 100
    INIT_TRAIN_BATCH = 16
    TRAIN_IMGSZ = 640
    TRAIN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    OUTPUT_DIR = "src/fish_classify_results_v2"
    TRAIN_OUTPUT_DIR = "./runs/train/fishinv_frozen_100"


os.makedirs(Config.TRAIN_OUTPUT_DIR, exist_ok=True)


# --------------------------
# 冻结 Backbone
# --------------------------
def freeze_backbone(model):
    """
    Freeze YOLOv5 backbone layers: model.0 ~ model.9
    """
    freeze_layers = list(range(10))  # freeze model.0 to model.9
    freeze_count = 0

    for name, param in model.model.named_parameters():
        layer_id = name.split('.')[1] if name.startswith("model.") else None
        if layer_id and layer_id.isdigit() and int(layer_id) in freeze_layers:
            param.requires_grad = False
            freeze_count += 1

    print(f"✅ 冻结 YOLOv5 backbone 层数: {freeze_count}")



# --------------------------
# 2. 生成 data.yaml
# --------------------------
def generate_yolo_data_yaml():
    if os.path.exists(Config.DATA_YAML_PATH):
        print(f"✅ 使用已有 data.yaml: {Config.DATA_YAML_PATH}")
        return

    yaml_content = {
        "path": os.path.abspath(Config.BASE_DATASET_DIR),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(Config.TARGET_CLASSES),
        "names": Config.TARGET_CLASSES
    }

    with open(Config.DATA_YAML_PATH, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, sort_keys=False, allow_unicode=True)

    print(f"✅ 自动生成 data.yaml: {Config.DATA_YAML_PATH}")


# --------------------------
# 3. 加载模型
# --------------------------
def load_pretrained_model():
    model_path = Config.MODEL_PATH
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ 预训练模型未找到：{model_path}")

    model = YOLO(model_path)
    print(f"✅ 加载预训练模型：{os.path.basename(model_path)}")

    # 加入冻结 backbone
    freeze_backbone(model)

    return model


def adjust_batch_size():
    if Config.TRAIN_DEVICE != "cuda":
        return 1

    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if gpu_mem >= 16:
        batch = Config.INIT_TRAIN_BATCH
    elif 8 <= gpu_mem < 16:
        batch = max(1, Config.INIT_TRAIN_BATCH // 2)
    elif 4 <= gpu_mem < 8:
        batch = max(1, Config.INIT_TRAIN_BATCH // 4)
    else:
        batch = 1

    print(f"✅ 批大小调整为：{batch}")
    return batch


# --------------------------
# 4. 训练函数
# --------------------------
def train_model():
    generate_yolo_data_yaml()
    model = load_pretrained_model()
    batch_size = adjust_batch_size()

    train_args = {
        "data": Config.DATA_YAML_PATH,
        "epochs": Config.TRAIN_EPOCHS,
        "batch": batch_size,
        "imgsz": Config.TRAIN_IMGSZ,
        "device": Config.TRAIN_DEVICE,
        "project": os.path.dirname(Config.TRAIN_OUTPUT_DIR),
        "name": os.path.basename(Config.TRAIN_OUTPUT_DIR),
        "pretrained": True,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "save": True,
        "val": True,
        "verbose": True,
        "plots": True,
        "patience": 20,

        # ✅ 船上环境增强: 反光/水滴/背光/夜间
        "hsv_h": 0.02,
        "hsv_s": 0.7,
        "hsv_v": 0.7,
        "degrees": 2.0,
        "translate": 0.1,
        "scale": 0.4,
        "perspective": 0.0005,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.2,
        "copy_paste": 0.3,
    }

    print(f"\n🚀 开始训练")
    print(f"设备: {Config.TRAIN_DEVICE} | Batch: {batch_size}")
    print(f"输出路径: {Config.TRAIN_OUTPUT_DIR}")

    model.train(**train_args)

    trainer = model.trainer
    best = os.path.join(trainer.save_dir, "weights", "best.pt")
    last = os.path.join(trainer.save_dir, "weights", "last.pt")

    final = best if os.path.exists(best) else last
    print(f"\n✅ 训练完成！模型保存：{final}")
    return model


# --------------------------
# 主入口
# --------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("鱼类模型训练（带模块微调）")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    print("=" * 50)

    try:
        train_model()
        print("\n🎉 训练成功！")
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()
