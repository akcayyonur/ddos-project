import pandas as pd
import numpy as np

def load_nsl_kdd(csv_path: str):
    """
    NSL-KDD dosyasını okur:
      - Etiketi ikili (0=normal, 1=attack) hale getirir.
      - Seçili temel 6 özelliği alır.
      - Basit türev özellikler ekler.
    Çıktı: (X, y)
    """
    from .config import KDD_COLUMNS, FEATURES, LABEL_COL, ATTACK_LABELS_POSITIVE

    df = pd.read_csv(csv_path, names=KDD_COLUMNS, header=None)

    # label -> 0(normal) / 1(attack)
    y = (
        df[LABEL_COL]
        .astype(str)
        .str.lower()
        .apply(lambda x: 0 if "normal" in x else 1 if any(a in x for a in ATTACK_LABELS_POSITIVE) else 1)
    )

    # Temel sayısal özellikler
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # ---- Basit türev özellikler (eğitime fayda sağlar) ----
    # bytes oranı
    X["bytes_ratio"] = (X["src_bytes"] + 1.0) / (X["dst_bytes"] + 1.0)
    # sayım oranı
    X["count_ratio"] = (X["srv_count"] + 1.0) / (X["count"] + 1.0)
    # (opsiyonel) log dönüşümleri
    X["log_src_bytes"] = np.log1p(X["src_bytes"])
    X["log_dst_bytes"] = np.log1p(X["dst_bytes"])
    # -------------------------------------------------------

    # NaN temizlik (teoride kalmamalı ama emniyet)
    X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return X, y

def make_synthetic(n_rows=5000):
    """
    Çevrimdışı/ilk test için sentetik veri üretimi (özellik isimleri aynı).
    """
    rng = np.random.default_rng(42)
    X = pd.DataFrame({
        "duration": rng.exponential(2.0, n_rows),
        "src_bytes": rng.lognormal(5.0, 1.0, n_rows),
        "dst_bytes": rng.lognormal(5.0, 1.2, n_rows),
        "count": rng.integers(1, 100, n_rows),
        "srv_count": rng.integers(1, 100, n_rows),
        "serror_rate": rng.uniform(0, 1, n_rows),
    })
    # türevler
    X["bytes_ratio"] = (X["src_bytes"] + 1.0) / (X["dst_bytes"] + 1.0)
    X["count_ratio"] = (X["srv_count"] + 1.0) / (X["count"] + 1.0)
    X["log_src_bytes"] = np.log1p(X["src_bytes"])
    X["log_dst_bytes"] = np.log1p(X["dst_bytes"])

    # kaba kural: yüksek count & serror_rate saldırı
    y = ((X["count"] > 60) & (X["serror_rate"] > 0.4)).astype(int)
    return X, y
