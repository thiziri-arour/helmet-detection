# Downloads Construction-PPE and keeps only helmet, no_helmet, vest and person
# (the dataset has 11 classes, the others are dropped)

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip"

# old id -> new id
KEEP = {
    0: 0,  # helmet
    7: 1,  # no_helmet
    2: 2,  # vest
    6: 3,  # Person
}
NAMES = ["helmet", "no_helmet", "vest", "person"]
SPLITS = ["train", "val", "test"]


def download(raw_dir: Path) -> None:
    zip_path = raw_dir / "construction-ppe.zip"
    if not zip_path.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {URL} ...")
        urllib.request.urlretrieve(URL, zip_path)
    if not (raw_dir / "images").exists():
        print("Extracting ...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(raw_dir)


def remap_labels(src: Path, dst: Path) -> int:
    lines = []
    for line in src.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        cls = int(parts[0])
        if cls in KEEP:
            lines.append(" ".join([str(KEEP[cls])] + parts[1:]))
    dst.write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def build(raw_dir: Path, out_dir: Path) -> None:
    counts = {name: 0 for name in NAMES}
    for split in SPLITS:
        img_out = out_dir / "images" / split
        lbl_out = out_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        images = sorted((raw_dir / "images" / split).glob("*"))
        for img in images:
            shutil.copy2(img, img_out / img.name)
            label = raw_dir / "labels" / split / f"{img.stem}.txt"
            new_label = lbl_out / f"{img.stem}.txt"
            if label.exists():
                remap_labels(label, new_label)
            else:
                new_label.write_text("")
        print(f"{split}: {len(images)} images")

    for txt in (out_dir / "labels").rglob("*.txt"):
        for line in txt.read_text().splitlines():
            counts[NAMES[int(line.split()[0])]] += 1
    print("Boxes per class:", counts)

    # path has to be absolute or ultralytics looks in its own datasets folder
    yaml = [f"path: {out_dir.resolve()}"]
    yaml += [f"{s}: images/{s}" for s in SPLITS]
    yaml += ["", "names:"] + [f"  {i}: {n}" for i, n in enumerate(NAMES)]
    (out_dir / "data.yaml").write_text("\n".join(yaml) + "\n")
    print(f"Wrote {out_dir / 'data.yaml'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw", help="where the zip is downloaded")
    parser.add_argument("--out", default="data/ppe", help="where the cleaned dataset goes")
    args = parser.parse_args()

    raw_dir, out_dir = Path(args.raw), Path(args.out)
    download(raw_dir)
    build(raw_dir, out_dir)


if __name__ == "__main__":
    main()
