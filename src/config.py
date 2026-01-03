from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

MODEL_PATH = MODEL_DIR / "rf_ddos_model.joblib"
META_PATH = MODEL_DIR / "model_meta.json"

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

FEATURES = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "count",
    "srv_count",
    "serror_rate"
]

LABEL_COL = "label"

ATTACK_LABELS_POSITIVE = {"attack", "anomaly", "malicious",
    "neptune","smurf","teardrop","back","land","pod","apache2","udpstorm",
    "processtable","worm","mailbomb","ipsweep","nmap","portsweep","satan",
    "mscan","ps","saint","guess_passwd","ftp_write","imap","phf","multihop",
    "spy","warezclient","warezmaster","snmpgetattack","snmpguess",
    "httptunnel","named","sendmail","xlock","xsnoop","buffer_overflow",
    "loadmodule","perl","rootkit","xterm","sqlattack"
}
