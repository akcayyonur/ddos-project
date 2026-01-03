import argparse
from pathlib import Path
import sys

def unique(paths):
    # de-duplicate while preserving order
    seen = set()
    out = []
    for p in paths:
        key = str(p.resolve()).lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out

def classify_files(src: Path):
    # topla ve isimden train/test ayır
    all_txt  = list(src.glob("*.txt"))  + list(src.glob("*.TXT"))
    all_arff = list(src.glob("*.arff")) + list(src.glob("*.ARFF"))

    all_txt  = unique(all_txt)
    all_arff = unique(all_arff)

    train_txt = []
    test_txt  = []
    for p in all_txt:
        name = p.name.lower()
        if "train" in name:
            train_txt.append(p)
        elif "test" in name:
            test_txt.append(p)

    train_arff = []
    test_arff  = []
    for p in all_arff:
        name = p.name.lower()
        if "train" in name:
            train_arff.append(p)
        elif "test" in name:
            test_arff.append(p)

    return train_txt, test_txt, train_arff, test_arff

def normalize_line_42cols(line: str):
    # 43 kolonsa (difficulty Level) son kolonu at
    parts = [x for x in line.strip().split(",")]
    if len(parts) == 43:
        parts = parts[:-1]
    return ",".join(parts)

def read_txt_file(p: Path):
    out = []
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            if not raw.strip():
                continue
            out.append(normalize_line_42cols(raw))
    return out

def read_arff_file(p: Path):
    out, in_data = [], False
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            low = line.lower()
            if low.startswith("@data"):
                in_data = True
                continue
            if not in_data or line.startswith("%"):
                continue
            out.append(normalize_line_42cols(line))
    return out

def write_lines(lines, outpath: Path, label):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open("w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln.rstrip() + "\n")
    print(f"[ok] {label}: {len(lines)} lines -> {outpath}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Kaynak klasör (NSL-KDD dosyaları)")
    ap.add_argument("--dst", default="data", help="Hedef klasör (data/)")
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    if not src.exists():
        print(f"[err] source not found: {src}", file=sys.stderr)
        return 2

    train_txt, test_txt, train_arff, test_arff = classify_files(src)

    print("== picked files ==")
    for p in train_txt:  print(" train-txt :", p.name)
    for p in train_arff: print(" train-arff:", p.name)
    for p in test_txt:   print(" test-txt  :", p.name)
    for p in test_arff:  print(" test-arff :", p.name)

    all_train, all_test = [], []

    # 1) TRAIN: öncelik txt, yoksa arff
    if train_txt:
        for p in train_txt:
            all_train.extend(read_txt_file(p))
    elif train_arff:
        for p in train_arff:
            all_train.extend(read_arff_file(p))
    else:
        print("[warn] no train files found (txt/arff)")

    # 2) TEST: öncelik arff, yoksa txt
    if test_arff:
        for p in test_arff:
            all_test.extend(read_arff_file(p))
    elif test_txt:
        for p in test_txt:
            all_test.extend(read_txt_file(p))
    else:
        print("[warn] no test files found (txt/arff)")

    # yaz ve örnek kolon sayıları
    if all_train:
        write_lines(all_train, dst / "nsl_kdd_train.csv", "train")
        for i, ln in enumerate(all_train[:3]):
            print(" sample train", i+1, "cols:", len(ln.split(",")))
    if all_test:
        write_lines(all_test, dst / "nsl_kdd_test.csv", "test")
        for i, ln in enumerate(all_test[:3]):
            print(" sample test", i+1, "cols:", len(ln.split(",")))

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
