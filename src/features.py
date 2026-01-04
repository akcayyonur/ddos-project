import pandas as pd
import numpy as np

def load_cicids2019(csv_path: str):
    """
    CIC-IDS2019 dosyasını okur:
      - Etiketi ikili (0=benign, 1=attack) hale getirir.
      - Seçili önemli özellikleri alır.
      - Türev özellikler ekler.
    Çıktı: (X, y)
    """
    from .config import FEATURES, LABEL_COL, ATTACK_LABELS_POSITIVE
    
    # CIC-IDS2019 dosyaları genellikle başlıklı gelir
    df = pd.read_csv(csv_path, encoding='utf-8', low_memory=False)
    
    # Sütun isimlerini temizle (boşluk, büyük/küçük harf)
    df.columns = df.columns.str.strip()
    
    # Label sütununu bul (farklı isimlendirme olabilir)
    label_candidates = ['Label', 'label', ' Label', 'Label ']
    label_col = None
    for candidate in label_candidates:
        if candidate in df.columns:
            label_col = candidate
            break
    
    if label_col is None:
        raise ValueError(f"Label column not found. Available columns: {df.columns.tolist()}")
    
    # Label temizliği ve ikili sınıflandırma
    df[label_col] = df[label_col].astype(str).str.strip().str.lower()
    
    # Benign/Attack sınıflandırması
    y = df[label_col].apply(lambda x: 0 if 'benign' in x.lower() else 1)
    
    # Özellik seçimi - eksik sütunlar varsa 0 ile doldur
    X = pd.DataFrame()
    for feat in FEATURES:
        if feat in df.columns:
            X[feat] = pd.to_numeric(df[feat], errors='coerce')
        else:
            print(f"[WARNING] Feature '{feat}' not found in dataset, filling with 0")
            X[feat] = 0.0
    
    # NaN ve Inf değerleri temizle
    X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    
    # ---- Türev özellikler (ML performansını artırır) ----
    # Bytes oranı (forward/backward)
    X["bytes_ratio"] = (X["Total Length of Fwd Packets"] + 1.0) / (X["Total Length of Bwd Packets"] + 1.0)
    
    # Packet oranı
    X["packet_ratio"] = (X["Total Fwd Packets"] + 1.0) / (X["Total Backward Packets"] + 1.0)
    
    # Log dönüşümleri (büyük değerleri normalize et)
    X["log_flow_duration"] = np.log1p(X["Flow Duration"])
    X["log_fwd_bytes"] = np.log1p(X["Total Length of Fwd Packets"])
    X["log_bwd_bytes"] = np.log1p(X["Total Length of Bwd Packets"])
    
    # Flag yoğunluğu (SYN/RST/ACK oranı)
    total_flags = X["SYN Flag Count"] + X["RST Flag Count"] + X["ACK Flag Count"] + 1.0
    X["syn_ratio"] = X["SYN Flag Count"] / total_flags
    X["rst_ratio"] = X["RST Flag Count"] / total_flags
    X["ack_ratio"] = X["ACK Flag Count"] / total_flags
    
    # Paket/saniye * süre = toplam paket tahmini
    X["estimated_packets"] = X["Flow Packets/s"] * X["Flow Duration"]
    # -------------------------------------------------------
    
    # Final temizlik
    X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    
    return X, y


def make_synthetic_cicids(n_rows=5000):
    """
    CIC-IDS2019 benzeri sentetik veri üretimi (ilk testler için)
    """
    rng = np.random.default_rng(42)
    
    # Normal trafik (benign)
    n_benign = int(n_rows * 0.7)
    benign_data = {
        "Flow Duration": rng.exponential(scale=5000, size=n_benign),
        "Total Length of Fwd Packets": rng.lognormal(mean=8, sigma=1.5, size=n_benign),
        "Total Length of Bwd Packets": rng.lognormal(mean=8, sigma=1.5, size=n_benign),
        "Total Fwd Packets": rng.integers(5, 50, size=n_benign),
        "Total Backward Packets": rng.integers(5, 50, size=n_benign),
        "Flow Packets/s": rng.uniform(10, 100, size=n_benign),
        "Flow Bytes/s": rng.uniform(1000, 10000, size=n_benign),
        "SYN Flag Count": rng.integers(0, 2, size=n_benign),
        "RST Flag Count": rng.integers(0, 2, size=n_benign),
        "ACK Flag Count": rng.integers(1, 10, size=n_benign),
    }
    
    # Attack trafik (DDoS/DoS)
    n_attack = n_rows - n_benign
    attack_data = {
        "Flow Duration": rng.exponential(scale=500, size=n_attack),  # Kısa süreli
        "Total Length of Fwd Packets": rng.lognormal(mean=6, sigma=0.5, size=n_attack),  # Küçük paketler
        "Total Length of Bwd Packets": rng.lognormal(mean=5, sigma=0.5, size=n_attack),
        "Total Fwd Packets": rng.integers(100, 1000, size=n_attack),  # Çok paket
        "Total Backward Packets": rng.integers(1, 10, size=n_attack),  # Az yanıt
        "Flow Packets/s": rng.uniform(1000, 10000, size=n_attack),  # Yüksek hız
        "Flow Bytes/s": rng.uniform(50000, 500000, size=n_attack),
        "SYN Flag Count": rng.integers(10, 100, size=n_attack),  # SYN flood
        "RST Flag Count": rng.integers(0, 5, size=n_attack),
        "ACK Flag Count": rng.integers(0, 5, size=n_attack),
    }
    
    # DataFrames oluştur
    df_benign = pd.DataFrame(benign_data)
    df_attack = pd.DataFrame(attack_data)
    
    # Birleştir
    X = pd.concat([df_benign, df_attack], ignore_index=True)
    y = pd.Series([0] * n_benign + [1] * n_attack)
    
    # Türev özellikler ekle
    X["bytes_ratio"] = (X["Total Length of Fwd Packets"] + 1.0) / (X["Total Length of Bwd Packets"] + 1.0)
    X["packet_ratio"] = (X["Total Fwd Packets"] + 1.0) / (X["Total Backward Packets"] + 1.0)
    X["log_flow_duration"] = np.log1p(X["Flow Duration"])
    X["log_fwd_bytes"] = np.log1p(X["Total Length of Fwd Packets"])
    X["log_bwd_bytes"] = np.log1p(X["Total Length of Bwd Packets"])
    
    total_flags = X["SYN Flag Count"] + X["RST Flag Count"] + X["ACK Flag Count"] + 1.0
    X["syn_ratio"] = X["SYN Flag Count"] / total_flags
    X["rst_ratio"] = X["RST Flag Count"] / total_flags
    X["ack_ratio"] = X["ACK Flag Count"] / total_flags
    X["estimated_packets"] = X["Flow Packets/s"] * X["Flow Duration"]
    
    X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    
    return X, y


# NSL-KDD compatibility (eski kodlarla uyumluluk için)
def load_nsl_kdd(csv_path: str):
    """
    ESKİ NSL-KDD fonksiyonu - geriye dönük uyumluluk için
    """
    print("[WARNING] Using load_nsl_kdd is deprecated. Use load_cicids2019 instead.")
    from .config import KDD_COLUMNS, FEATURES as OLD_FEATURES, LABEL_COL, ATTACK_LABELS_POSITIVE
    
    df = pd.read_csv(csv_path, names=KDD_COLUMNS, header=None)
    y = (
        df[LABEL_COL]
        .astype(str)
        .str.lower()
        .apply(lambda x: 0 if "normal" in x else 1 if any(a in x for a in ATTACK_LABELS_POSITIVE) else 1)
    )
    
    X = df[OLD_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X["bytes_ratio"] = (X["src_bytes"] + 1.0) / (X["dst_bytes"] + 1.0)
    X["count_ratio"] = (X["srv_count"] + 1.0) / (X["count"] + 1.0)
    X["log_src_bytes"] = np.log1p(X["src_bytes"])
    X["log_dst_bytes"] = np.log1p(X["dst_bytes"])
    X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    
    return X, y


def make_synthetic(n_rows=5000):
    """
    ESKİ sentetik veri fonksiyonu - geriye dönük uyumluluk için
    """
    print("[WARNING] Using make_synthetic is deprecated. Use make_synthetic_cicids instead.")
    return make_synthetic_cicids(n_rows)