import os
import torch
from ultralytics_offline import YOLO

# ========================
# 配置区域
# ========================
WEIGHTS_PATH = "./runs/train/fishinv_frozen_1002/weights/best.pt"  # ← 使用第一阶段 best.pt
DATA_YAML = "./data/data.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50     # 第二阶段建议 30~80
BATCH = 16
IMGSZ = 640

# 解冻 model.6~model.9
UNFREEZE_LAYERS = list(range(6, 10))


def unfreeze_partial_backbone(model):
    unfreeze_count = 0
    freeze_count = 0

    for name, param in model.model.named_parameters():
        layer_id = name.split('.')[1] if name.startswith("model.") else None
        if layer_id and layer_id.isdigit():
            idx = int(layer_id)
            if idx < 6:     # model.0~5: 继续冻结
                param.requires_grad = False
                freeze_count += 1
            elif idx in UNFREEZE_LAYERS:  # model.6~9 解冻
                param.requires_grad = True
                unfreeze_count += 1
            else:           # 10+ 默认训练
                param.requires_grad = True

    print(f"✅ 解冻层: {UNFREEZE_LAYERS}, 解冻参数: {unfreeze_count}, 继续冻结: {freeze_count}")


def main():
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"❌ 找不到权重: {WEIGHTS_PATH}")

    print(f"🚀 二阶段微调开始（部分 backbone 解冻）")
    model = YOLO(WEIGHTS_PATH)

    unfreeze_partial_backbone(model)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        device=DEVICE,
        pretrained=False,         # ⚠️ 二阶段必须 False
        optimizer="AdamW",
        lr0=0.0002,               # ✅ 小学习率
        weight_decay=0.001,
        patience=15,
        mosaic=0.7,               # 略减 Mosaic
        mixup=0.15,
        hsv_h=0.02,
        hsv_s=0.6,
        hsv_v=0.6,
        translate=0.1,
        scale=0.4,
        flipud=0.0,
        fliplr=0.4,
        copy_paste=0.2,
        perspective=0.0004,
        plots=True,
        verbose=True,
    )

    print("🎯 二阶段训练完成！建议继续评估，并准备第三阶段 LR 微调")


if __name__ == "__main__":
    main()
