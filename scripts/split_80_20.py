import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_ORDER = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "count",
    "srv_count",
    "serror_rate",
    "bytes_ratio",
    "count_ratio",
    "log_src_bytes",
    "log_dst_bytes",
]

# Her feature için olası kolon adları (CIC export/sürüm farklarını yakalamak için)
CANDIDATES = {
    "duration": [
        "Flow Duration", "flow_duration", "Flow_Duration"
    ],
    "src_bytes": [
        "Total Fwd Bytes", "TotLen Fwd Pkts", "Total Length of Fwd Packets",
        "Fwd Packets Length Total", "Fwd Bytes", "total_fwd_bytes"
    ],
    "dst_bytes": [
        "Total Bwd Bytes", "TotLen Bwd Pkts", "Total Length of Bwd Packets",
        "Bwd Packets Length Total", "Bwd Bytes", "total_bwd_bytes"
    ],
    "count": [
        "Total Fwd Packets", "Tot Fwd Pkts", "Fwd Packet Count",
        "Fwd Packets", "total_fwd_packets"
    ],
    # SYN flood sinyali için (yoksa 0’a düşeceğiz)
    "syn": [
        "SYN Flag Count", "Syn Flag Cnt", "SYN Flag Cnt", "syn_flag_count"
    ],
    "label": [
        "Label", "label", "Attack", "Class"
    ],
}

def pick_col(df: pd.DataFrame, want: str) -> str | None:
    """df kolonları içinde want için en uygun kolon adını bulur."""
    cols = list(df.columns)

    # 1) Birebir adaylar
    for c in CANDIDATES.get(want, []):
        if c in cols:
            return c

    # 2) Case-insensitive + boşluk/altçizgi normalize ederek yakala
    norm = {normalize(x): x for x in cols}
    for c in CANDIDATES.get(want, []):
        key = normalize(c)
        if key in norm:
            return norm[key]

    return None

def normalize(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_")

def build_features(df: pd.DataFrame):
    # Zorunlu kolonları seç
    col_duration = pick_col(df, "duration")
    col_srcbytes = pick_col(df, "src_bytes")
    col_dstbytes = pick_col(df, "dst_bytes")
    col_count    = pick_col(df, "count")
    col_label    = pick_col(df, "label")
    col_syn      = pick_col(df, "syn")  # opsiyonel

    missing = [k for k, v in {
        "duration": col_duration,
        "src_bytes": col_srcbytes,
        "dst_bytes": col_dstbytes,
        "count": col_count,
        "label": col_label,
    }.items() if v is None]

    if missing:
        raise KeyError(
            "Missing required columns: "
            + ", ".join(missing)
            + "\nAvailable columns sample:\n"
            + ", ".join(list(map(str, df.columns[:40])))
        )

    X = pd.DataFrame()
    X["duration"] = pd.to_numeric(df[col_duration], errors="coerce").fillna(0.0)
    X["src_bytes"] = pd.to_numeric(df[col_srcbytes], errors="coerce").fillna(0.0)
    X["dst_bytes"] = pd.to_numeric(df[col_dstbytes], errors="coerce").fillna(0.0)

    pkt_count = pd.to_numeric(df[col_count], errors="coerce").fillna(0.0)
    X["count"] = pkt_count
    X["srv_count"] = pkt_count

    if col_syn is not None:
        syn = pd.to_numeric(df[col_syn], errors="coerce").fillna(0.0)
        X["serror_rate"] = syn / (pkt_count.replace(0, 1))
    else:
        # SYN sayacı yoksa, 0 kabul et (kaba ama çalışır)
        X["serror_rate"] = 0.0

    X["bytes_ratio"] = (X["src_bytes"] + 1.0) / (X["dst_bytes"] + 1.0)
    X["count_ratio"] = (X["srv_count"] + 1.0) / (X["count"] + 1.0)
    X["log_src_bytes"] = np.log1p(X["src_bytes"])
    X["log_dst_bytes"] = np.log1p(X["dst_bytes"])

    X = X[FEATURE_ORDER].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # Label: BENIGN -> 0, diğerleri -> 1
    y_raw = df[col_label].astype(str).str.upper()
    y = (y_raw != "BENIGN").astype(int)

    picked = {
        "duration": col_duration,
        "src_bytes": col_srcbytes,
        "dst_bytes": col_dstbytes,
        "count": col_count,
        "syn": col_syn,
        "label": col_label,
    }

    return X, y, picked

def main():
    data_dir = Path("data_cic2019")
    out_dir = Path("data_cic2019_out")
    out_dir.mkdir(exist_ok=True)

    X_all = []
    y_all = []
    picked_any = None

    files = sorted(list(data_dir.glob("*-training.parquet")))
    if not files:
        raise FileNotFoundError(f"No *-training.parquet found in {data_dir.resolve()}")

    for p in files:
        print(f"Loading {p.name}")
        df = pd.read_parquet(p)

        X, y, picked = build_features(df)
        if picked_any is None:
            picked_any = picked
            print("[picked columns on first file]", picked_any)

        X_all.append(X)
        y_all.append(y)

    X_all = pd.concat(X_all, ignore_index=True)
    y_all = pd.concat(y_all, ignore_index=True)

    X_all.to_csv(out_dir / "X.csv", index=False, header=False)
    y_all.to_csv(out_dir / "y.csv", index=False, header=False)

    meta = {
        "feature_order": FEATURE_ORDER,
        "threshold": 0.5,
        "dataset": "CIC-DDoS-2019 (parquet)",
        "rows": int(len(X_all)),
        "column_mapping_used": picked_any,
        "label_map": {"BENIGN": 0, "OTHER": 1},
    }

    (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("DONE:", X_all.shape)
    print("Output:", out_dir.resolve())

if __name__ == "__main__":
    main()
