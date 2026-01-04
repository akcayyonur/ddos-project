# scripts/parquet_to_csv.py

import pandas as pd
from pathlib import Path
import sys
import numpy as np

def main():
    # Dosya yolları
    src_dir = Path("/home/akcayonur/ddos-project/data_cic2019")
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_csv = out_dir / "cicids2019_merged.csv"
    
    print(f"[INFO] Parquet dosyalari taraniyor: {src_dir}")
    
    # *-training.parquet dosyalarini bul
    files = sorted(list(src_dir.glob("*-training.parquet")))
    
    if not files:
        print("[ERROR] Hicbir .parquet dosyasi bulunamadi!")
        print(f"       Aranan yol: {src_dir}")
        sys.exit(1)
        
    print(f"[INFO] {len(files)} adet dosya bulundu. Birlestiriliyor...")
    
    df_list = []
    
    # Ryu ve Egitim kodlarinin bekledigi kritik kolonlar
    # Bellek hatasi almamak icin sadece bunlari secebiliriz
    REQUIRED_COLS = [
        "Flow Duration", 
        "Total Fwd Packets", 
        "Total Backward Packets", 
        "Total Length of Fwd Packets", 
        "Total Length of Bwd Packets", 
        "Flow Packets/s", 
        "Flow Bytes/s", 
        "SYN Flag Count", 
        "RST Flag Count", 
        "ACK Flag Count",
        "Label"
    ]

    for p in files:
        print(f"  -> Okunuyor: {p.name}")
        try:
            df = pd.read_parquet(p)
            
            # Sütun isimlerindeki boşlukları temizle (Garanti olsun)
            df.columns = df.columns.str.strip()
            
            # Label düzeltmeleri (örn: 'UDP ' -> 'UDP')
            if "Label" in df.columns:
                df["Label"] = df["Label"].astype(str).str.strip()
            
            # Sadece gerekli kolonlar varsa onlari al, yoksa hepsini al
            # (Kolon ismi uyusmazligi riskine karsi kontrol)
            available_cols = [c for c in REQUIRED_COLS if c in df.columns]
            
            if len(available_cols) == len(REQUIRED_COLS):
                df_subset = df[available_cols]
                df_list.append(df_subset)
            else:
                # Kolonlar eksikse tümünü ekleyip sonra bakarız
                df_list.append(df)
                
        except Exception as e:
            print(f"  [HATA] Dosya okunamadi {p.name}: {e}")

    if not df_list:
        print("[HATA] Veri yüklenemedi.")
        sys.exit(1)

    print("[INFO] Tablolar birlestiriliyor (Concat)...")
    full_df = pd.concat(df_list, ignore_index=True)
    
    # Inf ve NaN temizliği
    full_df = full_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    print(f"[INFO] Toplam Satir: {len(full_df)}")
    print(f"[INFO] Etiket Dagilimi:\n{full_df['Label'].value_counts()}")
    
    print(f"[INFO] CSV kaydediliyor: {out_csv} ...")
    full_df.to_csv(out_csv, index=False)
    print("[INFO] İŞLEM TAMAMLANDI.")

if __name__ == "__main__":
    main()