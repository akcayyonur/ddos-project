import argparse
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from .config import DATA_DIR, MODEL_PATH, FEATURES
from .features import load_cicids2019, make_synthetic_cicids
from .utils import ensure_dirs, save_meta


def load_data(train_csv=None, test_csv=None, use_synthetic=False):
    """
    train/test CSV verilmişse onları yükler; yoksa sentetik veri üretir.
    """
    if use_synthetic:
        print("[INFO] Using synthetic CIC-IDS2019 data for testing")
        X, y = make_synthetic_cicids(n_rows=6000)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True, random_state=42, stratify=y
        )
        return X_train, X_test, y_train, y_test
    
    if train_csv and Path(train_csv).exists():
        print(f"[INFO] Loading training data from: {train_csv}")
        X_train, y_train = load_cicids2019(train_csv)
    else:
        raise FileNotFoundError(f"Training file not found: {train_csv}")
    
    if test_csv and Path(test_csv).exists():
        print(f"[INFO] Loading test data from: {test_csv}")
        X_test, y_test = load_cicids2019(test_csv)
    else:
        raise FileNotFoundError(f"Test file not found: {test_csv}")
    
    return X_train, X_test, y_train, y_test


def best_threshold_for_f1(clf, X_val, y_val):
    """
    predict_proba ile [0,1] aralığında gezen bir eşik için F1'i maksimize et.
    """
    if not hasattr(clf, "predict_proba"):
        return 0.5, None
    
    p = clf.predict_proba(X_val)[:, 1]
    best_f1, best_t = -1.0, 0.5
    
    print("\n[INFO] Finding optimal threshold...")
    for t in np.linspace(0.05, 0.95, 19):
        y_hat = (p >= t).astype(int)
        f1 = f1_score(y_val, y_hat, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    
    return float(best_t), float(best_f1)


def main(args):
    ensure_dirs()
    
    # Veri yükleme
    if args.synthetic:
        X_train, X_test, y_train, y_test = load_data(use_synthetic=True)
    else:
        train_csv = args.train if args.train else (DATA_DIR / "cicids2019_train.csv")
        test_csv = args.test if args.test else (DATA_DIR / "cicids2019_test.csv")
        X_train, X_test, y_train, y_test = load_data(str(train_csv), str(test_csv))
    
    print(f"\n[INFO] Dataset loaded:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Training label distribution: {np.bincount(y_train)}")
    print(f"  Test label distribution: {np.bincount(y_test)}")
    
    # --- Train/Validation split (threshold seçimi için) ---
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    
    print(f"\n[INFO] Training model...")
    print(f"  Training set: {len(X_tr)} samples")
    print(f"  Validation set: {len(X_val)} samples")
    
    # --- Model: RandomForest with balanced class weights ---
    # CIC-IDS2019 genellikle dengesiz olduğu için class_weight='balanced' kullanıyoruz
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced' if args.balanced else None,
        verbose=1 if args.verbose else 0
    )
    
    clf.fit(X_tr, y_tr)
    
    # --- Feature importance ---
    if args.show_importance:
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n[INFO] Top 10 Feature Importances:")
        print(feature_importance.head(10).to_string(index=False))
    
    # --- En iyi threshold'u doğrulamada seç ---
    best_t, best_f1 = best_threshold_for_f1(clf, X_val, y_val)
    if best_f1 is not None:
        print(f"\n[INFO] Optimal threshold: {best_t:.4f} (validation F1: {best_f1:.4f})")
    
    # --- Test değerlendirme ---
    print(f"\n[INFO] Evaluating on test set...")
    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= best_t).astype(int)
    else:
        y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\n{'='*60}")
    print(f"TEST RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Benign', 'Attack'],
                                digits=4))
    
    # --- Model + meta kaydet ---
    joblib.dump(clf, MODEL_PATH)
    
    # Feature names (türevler dahil)
    feat_names = list(X_train.columns)
    save_meta(
        feature_names=feat_names,
        extra={
            "threshold": best_t,
            "test_accuracy": float(acc),
            "test_f1": float(f1),
            "n_estimators": args.n_estimators,
            "dataset": "CIC-IDS2019"
        }
    )
    
    print(f"\n{'='*60}")
    print(f"✓ Model saved: {MODEL_PATH}")
    print(f"✓ Meta saved: {MODEL_PATH.parent / 'model_meta.json'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import pandas as pd
    
    parser = argparse.ArgumentParser(description='Train DDoS detection model on CIC-IDS2019')
    parser.add_argument("--train", type=str, default=None, 
                       help="CIC-IDS2019 train CSV path")
    parser.add_argument("--test", type=str, default=None, 
                       help="CIC-IDS2019 test CSV path")
    parser.add_argument("--synthetic", action='store_true',
                       help="Use synthetic data for testing")
    parser.add_argument("--n-estimators", type=int, default=200,
                       help="Number of trees in Random Forest (default: 200)")
    parser.add_argument("--max-depth", type=int, default=None,
                       help="Maximum tree depth (default: None)")
    parser.add_argument("--balanced", action='store_true', default=True,
                       help="Use balanced class weights (default: True)")
    parser.add_argument("--show-importance", action='store_true',
                       help="Show feature importances")
    parser.add_argument("--verbose", action='store_true',
                       help="Verbose output during training")
    
    args = parser.parse_args()
    main(args)