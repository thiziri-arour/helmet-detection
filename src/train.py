# Fine-tunes yolov8n on the PPE dataset.
# On CPU 50 epochs took me ~6h, the colab notebook is faster.
# To just check it runs: --epochs 1 --fraction 0.1

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ppe/data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="starting weights (n, s, m...)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="0 for GPU, cpu for CPU")
    parser.add_argument("--fraction", type=float, default=1.0, help="use part of the train set")
    parser.add_argument("--name", default="ppe")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        fraction=args.fraction,
        patience=15,
        name=args.name,
        exist_ok=True,
    )

    best = Path(model.trainer.best)
    Path("weights").mkdir(exist_ok=True)
    shutil.copy(best, "weights/ppe_best.pt")
    print(f"Best model copied to weights/ppe_best.pt (from {best})")


if __name__ == "__main__":
    main()
