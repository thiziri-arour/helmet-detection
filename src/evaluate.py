# Test set metrics -> results/metrics.json

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="weights/ppe_best.pt")
    parser.add_argument("--data", default="data/ppe/data.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default="results/metrics.json")
    args = parser.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, split=args.split, name="eval", exist_ok=True)

    per_class = {}
    for i, cls_id in enumerate(metrics.box.ap_class_index):
        p, r, ap50, ap = metrics.box.class_result(i)
        per_class[model.names[int(cls_id)]] = {
            "precision": round(float(p), 3),
            "recall": round(float(r), 3),
            "mAP50": round(float(ap50), 3),
            "mAP50-95": round(float(ap), 3),
        }

    results = {
        "split": args.split,
        "mAP50": round(float(metrics.box.map50), 3),
        "mAP50-95": round(float(metrics.box.map), 3),
        "per_class": per_class,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))

    print(f"\nmAP50: {results['mAP50']}   mAP50-95: {results['mAP50-95']}")
    for name, m in per_class.items():
        print(f"  {name:<10} P={m['precision']:.3f}  R={m['recall']:.3f}  mAP50={m['mAP50']:.3f}")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
