from pathlib import Path

label_dir = Path("data/processed/manga109_yolo/labels/train")
label_files = sorted(label_dir.glob("*.txt"))

for f in label_files[:5]:
    lines = f.read_text().strip().split("\n")
    print(f"{f.name}: {len(lines)} annotations")
    print(f"  first line: {lines[0]}")
    print()