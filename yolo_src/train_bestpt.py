from ultralytics import YOLO
import torch
import numpy as np


def format_metric(value):
    if value is None:
        return 0.0
    if isinstance(value, (np.ndarray, list)):
        return float(value[0]) if len(value) > 0 else 0.0
    return float(value)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 从当前最优权重继续训练（根据你的实际路径调整）
    model = YOLO("yolo_src/wycBest.pt")

    total_epochs = 15  # 控制总轮数，避免训练过久

    # 可选：轻微冻结 backbone 前几层，稳定特征（不想冻结可设为 0）
    freeze_layers = 5
    if freeze_layers > 0:
        print(f"Freezing first {freeze_layers} layers.")
        for i, (_, param) in enumerate(model.model.named_parameters()):
            if i < freeze_layers:
                param.requires_grad = False

    print("Starting occlusion-aware fine-tuning...")

    results = model.train(
        data="yolo_src/data.yaml",
        epochs=total_epochs,
        imgsz=512,           # 减小输入尺寸，加快训练
        batch=16,             # 适配 8G 显存
        device=device,
        amp=True,

        # 优化器与学习率（稍高一点，加快收敛）
        optimizer="AdamW",
        lr0=7e-4,
        lrf=0.01,
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=2.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        cos_lr=True,

        # 数据增强：针对遮挡 / 模糊、尺度变化做强化
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.15,      # 稍大平移，模拟遮挡和局部可见
        scale=0.5,           # 更大缩放范围
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.8,          # 前期增强复杂场景
        mixup=0.0,
        copy_paste=0.0,

        # 训练策略
        pretrained=True,
        freeze=None,         # 手动 freeze 已处理
        close_mosaic=5,      # 5 个 epoch 后关闭 mosaic，利于收敛
        patience=5,          # 5 个 epoch 指标无提升则早停

        # 日志与保存
        project="runs/train",
        name="fishing_detector_occ_ft",
        exist_ok=False,
        val=True,
        save=True,
        save_period=-1,
        verbose=True,

        seed=0,
        deterministic=True,
        single_cls=False,
    )

    print("Fine-tuning completed!")
    print("Results saved to: runs/train/fishing_detector_occ_ft")

    # 使用新的 best 权重验证
    print("Validating best fine-tuned model...")
    best_model = YOLO("yolo_src/wycBest.pt")
    metrics = best_model.val(
        data="yolo_src/data.yaml",
        imgsz=512,
        batch=8,
        device=device,
        verbose=True,
    )

    print("Validation Results:")
    if metrics and hasattr(metrics, "box"):
        box_metrics = metrics.box
        map50 = format_metric(box_metrics.map50)
        map_val = format_metric(box_metrics.map)
        precision = format_metric(box_metrics.p)
        recall = format_metric(box_metrics.r)

        print(f"mAP50: {map50:.4f}")
        print(f"mAP50-95: {map_val:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
    else:
        print("Validation metrics not available")


if __name__ == "__main__":
    main()