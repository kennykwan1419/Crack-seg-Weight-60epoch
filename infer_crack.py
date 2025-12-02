import os
import cv2
import argparse
from ultralytics import YOLO
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Crack Segmentation Inference (YOLOv11-Seg)")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to model .pt file")
    parser.add_argument("--source", type=str, required=True, help="Image or folder to infer")
    parser.add_argument("--save-dir", type=str, default="runs/infer_crack", help="Output save directory")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


def load_images(path):
    """支持文件 & 文件夹输入"""
    if os.path.isfile(path):
        return [path]
    imgs = []
    for f in os.listdir(path):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            imgs.append(os.path.join(path, f))
    return imgs


def visualize_result(img, masks, save_path):
    """可视化 mask 并保存"""
    overlay = img.copy()

    if masks is not None:
        for m in masks:
            m = (m * 255).astype(np.uint8)
            m_colored = cv2.applyColorMap(m, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(overlay, 0.6, m_colored, 0.4, 0)

    cv2.imwrite(save_path, overlay)


def main():
    args = parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"[INFO] Loading model: {args.model}")
    model = YOLO(args.model)

    images = load_images(args.source)

    print(f"[INFO] Total images: {len(images)}")

    for img_path in images:
        print(f"\n[INFO] Inference: {img_path}")
        results = model.predict(
            source=img_path,
            imgsz=args.imgsz,
            save=False,
            conf=args.conf,
            verbose=False
        )

        for r in results:
            img = cv2.imread(img_path)
            h, w = img.shape[:2]

            save_name = os.path.basename(img_path)
            save_path = os.path.join(args.save_dir, f"vis_{save_name}")

            if r.masks is not None:
                masks = r.masks.data.cpu().numpy()  # N x H x W
                print(f"[INFO] Masks detected: {masks.shape[0]}")

                # 输出每个 mask 的面积
                for i, m in enumerate(masks):
                    area = m.sum()
                    print(f" - Mask {i+1}: area={area}")

                # 叠加可视化
                visualize_result(img, masks, save_path)

            else:
                print("[INFO] No crack mask detected.")
                visualize_result(img, None, save_path)

            print(f"[INFO] Saved: {save_path}")


if __name__ == "__main__":
    main()
