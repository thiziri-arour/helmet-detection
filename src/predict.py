# python src/predict.py --source site.jpg
# python src/predict.py --source site.mp4
#
# on videos, objects are tracked with bytetrack so each worker is counted once

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}


def predict_image(model, source, conf, out_dir):
    result = model.predict(source, conf=conf, verbose=False)[0]
    counts = Counter(model.names[int(c)] for c in result.boxes.cls)

    out_path = out_dir / f"{Path(source).stem}_pred.jpg"
    cv2.imwrite(str(out_path), result.plot())
    return dict(counts), out_path


def track_video(model, source, conf, out_dir, min_frames=5):
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    out_path = out_dir / f"{Path(source).stem}_tracked.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # track id -> Counter of the labels it got
    tracks = defaultdict(Counter)

    for result in model.track(source, conf=conf, tracker="bytetrack.yaml", stream=True, persist=True, verbose=False):
        writer.write(result.plot())
        if result.boxes.id is None:
            continue
        for cls, track_id in zip(result.boxes.cls.tolist(), result.boxes.id.tolist()):
            tracks[int(track_id)][model.names[int(cls)]] += 1

    writer.release()
    return summarize_tracks(tracks, model.names.values(), min_frames), out_path


def summarize_tracks(tracks, class_names, min_frames=5):
    # the label can flip between helmet and no_helmet for a few frames,
    # so each track gets the label it had most often.
    # very short tracks are mostly false detections
    counts = {name: 0 for name in class_names}
    for history in tracks.values():
        if sum(history.values()) < min_frames:
            continue
        label = history.most_common(1)[0][0]
        counts[label] += 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="image or video path")
    parser.add_argument("--weights", default="weights/ppe_best.pt")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--min-frames", type=int, default=5, help="ignore tracks shorter than this")
    parser.add_argument("--out", default="results/predictions")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)

    if Path(args.source).suffix.lower() in VIDEO_EXT:
        counts, out_path = track_video(model, args.source, args.conf, out_dir, args.min_frames)
        print("Unique objects tracked in the video:")
    else:
        counts, out_path = predict_image(model, args.source, args.conf, out_dir)
        print("Objects found in the image:")

    print(json.dumps(counts, indent=2))
    if counts.get("no_helmet"):
        print(f"WARNING: {counts['no_helmet']} worker(s) without a helmet")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
