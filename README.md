# DDoS Detection System using Machine Learning and SDN

A machine learning-based DDoS (Distributed Denial of Service) detection system integrated with **Ryu SDN controller** for real-time attack mitigation. This project uses the **CIC-IDS2019** dataset and Random Forest classifier to detect and mitigate DDoS attacks at the network level.

## 🎯 Features

- **ML-Based Detection**: Random Forest classifier trained on CIC-IDS2019 dataset
- **Real-Time Analysis**: Packet-level traffic analysis and threat classification
- **SDN Integration**: Seamless integration with Ryu SDN controller for automatic mitigation
- **Queue-Based Mitigation**: Separates benign and attack traffic using QoS queues
- **Auto-Reset Mechanism**: Periodic statistics reset for continuous attack detection
- **High Accuracy**: 99.9% test accuracy with F1-score of 0.9993

## 📊 Dataset & Model

- **Dataset**: CIC-IDS2019 (Canadian Institute for Cybersecurity)
- **Algorithm**: Random Forest Classifier (100-200 estimators)
- **Features**: 19 network flow features (including derived features)
- **Train/Test Split**: 80/20 with stratification
- **Threshold Optimization**: F1-score based threshold selection

### Key Features Used

| Feature | Description |
|---------|-------------|
| Flow Duration | Duration of the flow in microseconds |
| Total Fwd/Bwd Packets | Number of forward/backward packets |
| Total Fwd/Bwd Bytes | Total bytes sent forward/backward |
| Flow Packets/s | Packet rate per second |
| Flow Bytes/s | Bytes transmitted per second |
| SYN/RST/ACK Flag Count | TCP flag statistics |
| bytes_ratio | Ratio of forward to backward bytes (derived) |
| log_flow_duration | Log transformation of flow duration (derived) |
| syn_ratio, rst_ratio, ack_ratio | Flag ratios (derived) |

## 🚀 Quick Start

### Prerequisites

```bash
Python >= 3.9
pip >= 21.0
Ryu SDN Controller (for deployment)
```

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ddos-detection
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Training the Model

#### Option 1: Using Your Own Data

```bash
python -m src.train \
  --train data/cicids2019_train.csv \
  --test data/cicids2019_test.csv \
  --n-estimators 200 \
  --balanced \
  --show-importance
```

#### Option 2: Using Synthetic Data (Testing)

```bash
python -m src.train --synthetic
```

#### Windows Batch File

```bash
./run_train.bat
```

### Testing the Model (Offline)

```bash
python scripts/test_model_offline.py
```

This will:
- Load the trained model and metadata
- Perform feature engineering on test data
- Evaluate on random benign/attack samples
- Display prediction accuracy and statistics



## 🔧 Configuration

Edit `src/config.py` to customize:

```python
# Model paths
MODEL_PATH = "models/rf_ddos_model.joblib"
META_PATH = "models/model_meta.json"

# Features selection
FEATURES = [
    "Flow Duration",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    # ... more features
]

# Attack labels (customize for your dataset)
ATTACK_LABELS_POSITIVE = {
    "ddos", "syn", "udp", "dos", "hulk", "bot", ...
}
```

## 🧠 Model Training Details

### Training Pipeline

1. **Data Loading**: Read CIC-IDS2019 CSV with automatic column detection
2. **Feature Engineering**: 
   - Base features: 10 network flow statistics
   - Derived features: 9 engineered features (log transforms, ratios)
3. **Label Encoding**: Benign=0, All attacks=1
4. **Train/Validation/Test Split**: 60/20/20 with stratification
5. **Threshold Optimization**: F1-score maximization on validation set
6. **Evaluation**: Accuracy, F1-score, confusion matrix on test set

### Hyperparameters

```python
RandomForestClassifier(
    n_estimators=200,          # Number of trees
    max_depth=None,            # Unlimited depth
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight='balanced',   # Handle class imbalance
    n_jobs=-1,                 # Parallel processing
    random_state=42
)
```

## 🕵️ Ryu SDN Integration

### Running the DDoS Mitigator

```bash
# Start Ryu with the DDoS mitigator
ryu-manager src/ryu_ddos_mitigator.py
```

### How It Works

1. **Packet Capture**: Intercepts all packets at the switch
2. **Flow Tracking**: Maintains per-flow statistics (source IP, destination IP)
3. **Feature Engineering**: Builds feature vector from flow statistics
4. **Prediction**: Runs ML model to classify flow as benign/attack
5. **Mitigation**: 
   - Benign traffic → Queue 0 (normal priority)
   - Attack traffic → Queue 1 (low priority/rate-limited)
6. **Auto-Reset**: Clears flow stats every 100 packets for continuous detection

### Configuration in Ryu

```python
# Threshold for attack classification
self.threshold = 0.45  # From model_meta.json

# Reset statistics every N packets
self.reset_interval = 100

# Minimum packets before ML prediction
self.min_pkts_for_ml = 3
```

## 📈 Performance Metrics

```
Test Accuracy:  99.91%
F1-Score:       99.93%
Precision:      High true positive rate
Recall:         Catches most attacks

Confusion Matrix:
                Predicted Benign  Predicted Attack
Actual Benign   [True Negatives]  [False Positives]
Actual Attack   [False Negatives] [True Positives]
```

## 🛠️ Data Preparation Scripts

### Convert NSL-KDD Dataset

```bash
python scripts/convert_and_merge.py \
  --src /path/to/nsl_kdd_files \
  --dst data/
```

### Convert CIC-IDS2019 Parquet to CSV

```bash
python scripts/parquet_to_csv.py
```

### Split Data 80/20 with Stratification

```bash
python scripts/split_custom_data.py
```

### Check Train/Test Overlap

```bash
python scripts/check_overlap.py
```

## 📝 Feature Engineering Details

The system uses 19 features for classification:

**Base Features (10)**
- Flow Duration
- Total Fwd/Bwd Packets & Bytes
- Flow Packets/s & Bytes/s
- SYN/RST/ACK Flag Counts

**Derived Features (9)**
- bytes_ratio = (fwd_bytes + 1) / (bwd_bytes + 1)
- packet_ratio = (fwd_packets + 1) / (bwd_packets + 1)
- log_flow_duration = log(Flow Duration + 1)
- log_fwd_bytes, log_bwd_bytes
- syn_ratio, rst_ratio, ack_ratio = flag_count / (total_flags + 1)
- estimated_packets = Flow Packets/s × Flow Duration

This transformation helps the model:
- Handle skewed distributions (log transforms)
- Capture asymmetric traffic patterns (ratios)
- Normalize extreme values

## ⚠️ Important Notes

### Feature Order Consistency

The Ryu mitigator builds feature vectors in the exact order specified in `model_meta.json`:

```json
{
  "feature_order": [
    "Flow Duration",
    "Total Length of Fwd Packets",
    ...
    "estimated_packets"
  ],
  "threshold": 0.45,
  "test_accuracy": 0.9990812495006791,
  "test_f1": 0.9992695862047064
}
```

**Critical**: The Ryu controller must produce features in this exact order.

### Handling Missing Columns

If a dataset has different column names, edit the column mapping in:
- `src/features.py` - Feature extraction logic
- `scripts/split_80_20.py` - Data conversion utilities

## 🔄 Troubleshooting

### Model Not Found
```
[ERROR] ML model not found: models/rf_ddos_model.joblib
```
**Solution**: Train the model first with `python -m src.train`

### Feature Mismatch
```
[ERROR] Missing required columns: Flow Duration, ...
```
**Solution**: Check CSV column names match CIC-IDS2019 format or update mapping

### Prediction Issues
```
[WARNING] X does not have valid feature names
```
**Solution**: Ensure feature vector length matches `feature_order` in meta file

## 📊 Dataset Information

### CIC-IDS2019
- **Source**: Canadian Institute for Cybersecurity
- **Size**: 80+ GB of network traffic
- **Duration**: 5 days of capture
- **Features**: 80+ network flow metrics
- **Labels**: BENIGN, DDoS, DoS (various types), Port Scan, Bot, etc.
- **Format**: CSV with headers

### Supported Attack Types
```
DDoS: LDAP, MSSQL, NetBIOS, NTP, SNMP, SSDP, UDP, UDPLag
DoS: HULK, GoldenEye, Slowloris, SlowHTTPTest
Other: Port Scan, Bot, Web Attack, SSH/FTP Brute Force
```

## 🚀 Future Improvements

- [ ] Multi-class classification (identify specific attack types)
- [ ] Real-time model updates with incremental learning
- [ ] Feature selection optimization
- [ ] Ensemble methods (XGBoost, LightGBM)
- [ ] Explainability (SHAP, LIME)
- [ ] Web dashboard for monitoring
- [ ] Kubernetes deployment support

## 📜 License

This project is available under the MIT License.

## 👤 Author

Created for network security and SDN research.

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📚 References

- [CIC-IDS2019 Dataset](https://www.unb.ca/cic/datasets/ids-2019.html)
- [Ryu SDN Controller](https://ryu-sdn.org/)
- [scikit-learn RandomForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [OpenFlow 1.3 Specification](https://opennetworking.org/software-defined-standards/specifications/)

## ❓ FAQ

**Q: Can I use a different dataset?**
A: Yes, but ensure similar network flow features. Update column mapping in `features.py`.

**Q: What's the inference latency?**
A: ~1-5ms per packet classification on modern hardware (model prediction only).

**Q: Can the model detect unknown attack types?**
A: Yes, anomaly detection capability depends on training diversity. CIC-IDS2019 covers 14+ attack types.

**Q: How do I deploy this in production?**
A: See Ryu SDN integration section. Requires OpenFlow-compatible switch and controller placement.

---

**Last Updated**: February 2026  
**Model Version**: 1.0 (CIC-IDS2019, Random Forest)
