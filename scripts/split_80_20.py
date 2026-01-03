# scripts/split_80_20.py
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Ayarlar
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DELIVER_DIR = PROJECT_ROOT / "deliver"
DELIVER_DIR.mkdir(parents=True, exist_ok=True)

# KDD kolon adları (config.py ile uyumlu)
KDD_COLUMNS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"
]

# Proje odağındaki 6 temel özellik
FEATURES = ["duration","src_bytes","dst_bytes","count","srv_count","serror_rate"]
# (Eğer türev istiyorsan ilave edebilirsin; şu an Emirhan için sadece bu 6 özellik isteniyor.)

# Girdi dosyaları (temizlenmiş halinizi kullanıyoruz)
train_path = DATA_DIR / "nsl_kdd_train.csv"
test_path  = DATA_DIR / "nsl_kdd_test.csv"

# OK: her iki dosyayı da okuyup tek bir DataFrame yapalım (bazı kayıtlar zaten temizlenmiş)
def read_csv_no_header(p: Path):
    if not p.exists():
        return pd.DataFrame(columns=KDD_COLUMNS)
    return pd.read_csv(p, names=KDD_COLUMNS, header=None, encoding="utf-8", low_memory=False)

df_train = read_csv_no_header(train_path)
df_test  = read_csv_no_header(test_path)

# Birleştir (tüm kullanılabilir örnekler)
df_all = pd.concat([df_train, df_test], ignore_index=True)

# Etiketleri ikili yap (0 = normal, 1 = attack) — aynı mantık features.py ile tutarlı olmalı
def map_label(lbl):
    s = str(lbl).lower()
    if "normal" in s:
        return 0
    else:
        return 1

df_all["label_bin"] = df_all["label"].apply(map_label)

# Sadece 6 özellik + label alalım
cols_out = FEATURES + ["label_bin"]
df_sub = df_all[cols_out].copy()

# Basit temizleme: sayısal olmayanları numeric yap, NaN -> 0
for c in FEATURES:
    df_sub[c] = pd.to_numeric(df_sub[c], errors="coerce").fillna(0.0)

# Stratified split %80 train / %20 test
train_df, test_df = train_test_split(
    df_sub,
    test_size=0.20,
    stratify=df_sub["label_bin"],
    random_state=42
)

# Dosyaya yaz (başlıksız, index yok — proje formatıyla uyumlu)
train_out = DELIVER_DIR / "train_80.csv"
test_out  = DELIVER_DIR / "test_20.csv"
train_df.to_csv(train_out, index=False, header=False)
test_df.to_csv(test_out, index=False, header=False)

# Meta / teslimat dosyaları
meta = {
    "num_total": int(len(df_sub)),
    "train_rows": int(len(train_df)),
    "test_rows": int(len(test_df)),
    "feature_order": FEATURES,
    "label_map": {"normal": 0, "attack": 1},
    "note": "80/20 stratified split by label; no header; numeric features only."
}
(DELIVER_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

# Kısa README
readme = f"""
NSL-KDD 80/20 teslim paketi
---------------------------
Toplam örnek: {meta['num_total']}
Eğitim (80%): {meta['train_rows']}
Test (20%): {meta['test_rows']}

Özellikler (sıra): {', '.join(FEATURES)}
Etiketleme: label_bin (0 = normal, 1 = attack)

Dosyalar:
 - train_80.csv   (başlıksız, index yok)
 - test_20.csv    (başlıksız, index yok)
 - meta.json
"""
(DELIVER_DIR / "README.txt").write_text(readme.strip() + "\n", encoding="utf-8")

print("[ok] created deliver package at:", DELIVER_DIR)
print(" - train:", train_out, "(", len(train_df), "rows )")
print(" - test :", test_out,  "(", len(test_df),  "rows )")
