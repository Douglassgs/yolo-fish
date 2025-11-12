import os
import torch
from ultralytics import YOLO

# =========================
# 配置
# =========================
PREV_BEST = "./runs/detect/train/weights/best.pt"  # ✅ 第二阶段的 best.pt
DATA_YAML = "./data/data.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 40        # 建议 20~40
BATCH = 16
IMGSZ = 640

LR = 1e-5          # ✅ 全解冻极低学习率


def full_unfreeze(model):
    total = 0
    for name, param in model.model.named_parameters():
        param.requires_grad = True
        total += 1
    print(f"✅ 全部解冻完成，训练参数数：{total}")


def main():
    if not os.path.exists(PREV_BEST):
        raise FileNotFoundError(f"❌ 找不到权重: {PREV_BEST}")

    print(f"🚀 第三阶段：全解冻精调（Full Fine-tune）")
    model = YOLO(PREV_BEST)

    full_unfreeze(model)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        device=DEVICE,

        pretrained=False,            # ⚠️ 必须 False
        optimizer="AdamW",
        lr0=LR,                      # ✅ 极低学习率
        lrf=0.01,                    # ✅ Cosine final LR
        cos_lr=True,                 # ✅ 启用余弦退火
        momentum=0.937,
        weight_decay=0.0005,

        patience=10,
        warmup_epochs=2,
        warmup_bias_lr=0.0005,

        # ✅ 船上真实增强
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.6,
        translate=0.10,
        scale=0.35,
        fliplr=0.4,
        mosaic=0.3,                 # ✅ 再减一点增强，稳定训练
        mixup=0.05,
        copy_paste=0.1,
        perspective=0.0004,

        augment=True,
        plots=True,
        verbose=True
    )

    print("🎉 第三阶段完成！建议导出模型做实船测试")


if __name__ == "__main__":
    main()
