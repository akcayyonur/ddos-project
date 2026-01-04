import pandas as pd
import joblib
import numpy as np
from pathlib import Path
import sys
import json

# Proje ana dizinini yola ekle
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import MODEL_PATH, META_PATH

def main():
    # 1. Dosya Yolları
    test_csv_path = Path("data/cicids2019_test.csv")
    
    print(f"[BILGI] Model yukleniyor: {MODEL_PATH}")
    if not MODEL_PATH.exists():
        print("[HATA] Model dosyasi bulunamadi! Once egitim yapin.")
        return

    model = joblib.load(MODEL_PATH)
    
    # Meta dosyasindan feature sirasini al (Egitimdeki sirayla ayni olmali)
    if META_PATH.exists():
        with open(META_PATH, "r") as f:
            meta = json.load(f)
        feature_order = meta.get("feature_order", [])
        threshold = meta.get("threshold", 0.5)
        print(f"[BILGI] Model Eşigi (Threshold): {threshold}")
    else:
        print("[UYARI] Meta dosyasi yok, varsayilanlar kullanilacak.")
        feature_order = []
        threshold = 0.5

    print(f"[BILGI] Test verisi yukleniyor: {test_csv_path}")
    if not test_csv_path.exists():
        print("[HATA] Test CSV dosyasi bulunamadi!")
        return

    # Veriyi oku
    df = pd.read_csv(test_csv_path)
    
    # Kolon isimlerini temizle
    df.columns = df.columns.str.strip()
    
    # Gerekli özellikler (src/features.py ile uyumlu olmali)
    # Eger feature_order boşsa, modelin bekledigi kolonları tahmin etmeye calisalim
    # Genellikle egitim sirasinda olusan features.json veya meta.json kritiktir.
    # Burada manuel olarak türetilmiş özellikleri hesaplamamiz gerekebilir 
    # EGER test verisi zaten islenmis (engineered) degilse.
    
    # NOT: data/cicids2019_test.csv dosyaniz muhtemelen 'scripts/split_custom_data.py' 
    # ile olusturuldu ve sadece ham kolonlari iceriyor. 
    # Model ise turetilmis ozellikler (log_..., ratio_...) bekliyor olabilir.
    # Bu yuzden features.py icindeki donusumleri burada da yapmaliyiz.
    
    # --- Feature Engineering (features.py mantigi) ---
    # Bu kisim modelin bekledigi formata cevirmek icindir
    X = pd.DataFrame()
    
    # Temel kolon eşleşmeleri (Config.py'deki FEATURES listesine gore)
    # Mevcut CSV'deki isimler -> Modelin bekledigi isimler
    req_cols = [
        "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
        "Total Length of Fwd Packets", "Total Length of Bwd Packets",
        "Flow Packets/s", "Flow Bytes/s",
        "SYN Flag Count", "RST Flag Count", "ACK Flag Count"
    ]
    
    for col in req_cols:
        if col in df.columns:
            X[col] = df[col]
        else:
            X[col] = 0.0 # Eksikse 0
            
    # Türev Özellikler (Model bunlarla egitildiyse zorunludur)
    X["bytes_ratio"] = (X["Total Length of Fwd Packets"] + 1.0) / (X["Total Length of Bwd Packets"] + 1.0)
    X["packet_ratio"] = (X["Total Fwd Packets"] + 1.0) / (X["Total Backward Packets"] + 1.0)
    X["log_flow_duration"] = np.log1p(X["Flow Duration"])
    X["log_fwd_bytes"] = np.log1p(X["Total Length of Fwd Packets"])
    X["log_bwd_bytes"] = np.log1p(X["Total Length of Bwd Packets"])
    
    total_flags = X["SYN Flag Count"] + X["RST Flag Count"] + X["ACK Flag Count"] + 1.0
    X["syn_ratio"] = X["SYN Flag Count"] / total_flags
    X["rst_ratio"] = X["RST Flag Count"] / total_flags
    X["ack_ratio"] = X["ACK Flag Count"] / total_flags
    X["estimated_packets"] = X["Flow Packets/s"] * X["Flow Duration"] / 1e6 # Duration us ise
    
    # NaN temizligi
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    
    # Etiketleri al
    y_true = df["Label"].apply(lambda x: 1 if str(x).strip().upper() != "BENIGN" else 0)
    
    # Modelin bekledigi siraya gore kolonlari diz
    if feature_order:
        # Sadece mevcut olanlari al, eksikleri 0 yap
        for f in feature_order:
            if f not in X.columns:
                X[f] = 0.0
        X = X[feature_order]
    
    # --- TEST AŞAMASI ---
    print("\n" + "="*60)
    print("TEST SONUÇLARI (Rastgele 20 Örnek)")
    print("="*60)
    print(f"{'DURUM':<10} | {'GERÇEK':<10} | {'TAHMİN':<10} | {'OLASILIK':<10} | {'DETAY'}")
    print("-" * 70)
    
    # Rastgele 10 Benign, 10 Attack sec
    benign_idx = y_true[y_true == 0].sample(n=min(10, sum(y_true==0))).index
    attack_idx = y_true[y_true == 1].sample(n=min(10, sum(y_true==1))).index
    test_indices = list(benign_idx) + list(attack_idx)
    
    correct = 0
    total = 0
    
    for idx in test_indices:
        row = X.iloc[[idx]] # Tek satir DataFrame
        true_label = "ATTACK" if y_true.iloc[idx] == 1 else "BENIGN"
        
        # Tahmin
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(row)[0, 1]
            pred_val = 1 if prob >= threshold else 0
        else:
            pred_val = int(model.predict(row)[0])
            prob = 1.0 if pred_val == 1 else 0.0
            
        pred_label = "ATTACK" if pred_val == 1 else "BENIGN"
        
        # Sonuc
        is_correct = (pred_val == y_true.iloc[idx])
        status = "✅ OK" if is_correct else "❌ HATA"
        if is_correct: correct += 1
        total += 1
        
        # Detay bilgisi (Hangi saldiri turu oldugu csv'den)
        original_label = df.iloc[idx]["Label"]
        
        print(f"{status:<10} | {true_label:<10} | {pred_label:<10} | {prob:.4f}     | {original_label}")

    print("-" * 70)
    print(f"Örneklem Doğruluğu: {correct}/{total} ({(correct/total)*100:.1f}%)")

if __name__ == "__main__":
    main()
