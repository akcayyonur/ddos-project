from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

MODEL_PATH = MODEL_DIR / "rf_ddos_model.joblib"
META_PATH = MODEL_DIR / "model_meta.json"

# CIC-IDS2019 veri setinde bulunan temel kolonlar
# Not: CIC-IDS2019'da 80+ özellik var, burada en önemlilerini seçiyoruz
CICIDS_COLUMNS = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
    "Label"
]

# DDoS tespiti için en önemli özellikler (NSL-KDD benzerleri)
FEATURES = [
    "Flow Duration",           # duration benzeri
    "Total Length of Fwd Packets",  # src_bytes benzeri
    "Total Length of Bwd Packets",  # dst_bytes benzeri
    "Total Fwd Packets",       # count benzeri
    "Total Backward Packets",  # srv_count benzeri
    "Flow Packets/s",          # yoğunluk ölçüsü
    "Flow Bytes/s",            # bant genişliği kullanımı
    "SYN Flag Count",          # SYN flood tespiti için
    "RST Flag Count",          # anormal bağlantı sonlandırma
    "ACK Flag Count",          # ACK flood tespiti için
]

LABEL_COL = "Label"

# CIC-IDS2019'da DDoS saldırı türleri
ATTACK_LABELS_POSITIVE = {
    "ddos", "syn", "udp", "ldap", "mssql", "netbios", "ntp", "snmp", "ssdp", "udplag",
    "dos", "hulk", "goldeneye", "slowloris", "slowhttptest", "portscan", "bot",
    "brute", "force", "infiltration", "web", "attack", "ssh", "ftp", "patator"
}

# Etiket normalleştirme için mapping
LABEL_MAPPING = {
    "benign": 0,
    "ddos": 1,
    "dos": 1,
    "syn": 1,
    "udp": 1,
    "ldap": 1,
    "mssql": 1,
    "netbios": 1,
    "ntp": 1,
    "snmp": 1,
    "ssdp": 1,
    "udplag": 1,
}