# move_dataset_layout.py
import os
import shutil
from pathlib import Path

# 旧结构的根目录：包含 train/ val/ test/ 这三个文件夹
OLD_ROOT = Path("/home/douglass/yolo_fish")

# 新结构：在同一个根目录下创建 image/ 和 label/
IMAGE_ROOT = OLD_ROOT / "image"
LABEL_ROOT = OLD_ROOT / "label"

SPLITS = ["train", "val", "test"]

def move_dir(src: Path, dst: Path):
    if not src.exists():
        print(f"[跳过] 源目录不存在: {src}")
        return
    dst.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        if item.is_file():
            target = dst / item.name
            shutil.move(str(item), str(target))
        elif item.is_dir():
            target_dir = dst / item.name
            target_dir.mkdir(parents=True, exist_ok=True)
            move_dir(item, target_dir)

def main():
    # 创建新根目录
    IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    LABEL_ROOT.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        old_split = OLD_ROOT / split
        img_src = old_split / "image"
        lbl_src = old_split / "label"

        img_dst = IMAGE_ROOT / split
        lbl_dst = LABEL_ROOT / split

        print(f"处理 split = {split}")

        move_dir(img_src, img_dst)
        move_dir(lbl_src, lbl_dst)

        # 如果原来的 split 下已经空了，可以选择删除
        # 尽量只删空目录，避免误删
        try:
            if img_src.exists():
                img_src.rmdir()
            if lbl_src.exists():
                lbl_src.rmdir()
            # 如果 train/val/test 目录已经空，则删除
            old_split.rmdir()
        except OSError:
            # 目录非空就忽略
            pass

    print("完成目录重排。新结构：")
    print(f"- 图片：{IMAGE_ROOT} 下的 {', '.join(SPLITS)}")
    print(f"- 标签：{LABEL_ROOT} 下的 {', '.join(SPLITS)}")

if __name__ == "__main__":
    main()