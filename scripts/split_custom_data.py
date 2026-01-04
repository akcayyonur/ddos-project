# scripts/split_custom_data.py
import pandas as pd
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split

def main():
    # Sizin belirttiginiz klasör
    source_dir = Path("/home/akcayonur/ddos-project/datacic2019")
    
    # Hedef klasör (train.py buraya bakıyor)
    dest_dir = Path("data")
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"[BILGI] Klasor taraniyor: {source_dir}")
    
    if not source_dir.exists():
        print(f"[HATA] Klasor bulunamadi: {source_dir}")
        print("Lutfen yolu kontrol edin.")
        sys.exit(1)

    # Klasördeki ilk CSV dosyasını bul
    csv_files = list(source_dir.glob("*.csv"))
    
    if not csv_files:
        print("[HATA] Bu klasorde hic .csv dosyasi yok!")
        sys.exit(1)
        
    # İlk bulunan dosyayı alıyoruz (örn: cicids2019_merged.csv)
    target_file = csv_files[0]
    print(f"[BILGI] Bulunan dosya: {target_file.name}")
    print("Okunuyor (biraz zaman alabilir)...")
    
    try:
        df = pd.read_csv(target_file, low_memory=False)
    except Exception as e:
        print(f"[HATA] Dosya okunamadi: {e}")
        sys.exit(1)

    # Label temizliği
    # Sütun isimlerindeki boşlukları temizle
    df.columns = df.columns.str.strip()
    
    if "Label" in df.columns:
        df["Label"] = df["Label"].astype(str).str.strip()
    else:
        print("[HATA] 'Label' sütunu bulunamadi! Kolonlar:", list(df.columns))
        sys.exit(1)
        
    print(f"[BILGI] Toplam Satir: {len(df)}")
    
    # %80 Train / %20 Test Ayrımı
    print("[BILGI] Egitim (%80) ve Test (%20) olarak ayriliyor...")
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=df["Label"]
    )
    
    # Dosyaları data/ klasörüne kaydet
    train_path = dest_dir / "cicids2019_train.csv"
    test_path = dest_dir / "cicids2019_test.csv"
    
    print(f"[BILGI] Train kaydediliyor: {train_path}")
    train_df.to_csv(train_path, index=False)
    
    print(f"[BILGI] Test kaydediliyor: {test_path}")
    test_df.to_csv(test_path, index=False)
    
    print("\n[TAMAM] Dosyalar hazir! Simdi egitim komutunu calistirabilirsiniz.")

if __name__ == "__main__":
    main()
