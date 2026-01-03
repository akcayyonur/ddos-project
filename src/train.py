import argparse
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

from .config import DATA_DIR, MODEL_PATH, FEATURES
from .features import load_nsl_kdd, make_synthetic
from .utils import ensure_dirs, save_meta


def load_data(train_csv=None, test_csv=None):
    """
    train/test CSV verilmişse onları yükler; yoksa sentetik veri üretir.
    """
    if train_csv and Path(train_csv).exists() and test_csv and Path(test_csv).exists():
        X_train, y_train = load_nsl_kdd(train_csv)
        X_test, y_test   = load_nsl_kdd(test_csv)
    else:
        X, y = make_synthetic(n_rows=6000)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True, random_state=42, stratify=y
        )
    return X_train, X_test, y_train, y_test


def best_threshold_for_f1(clf, X_val, y_val):
    """
    predict_proba ile [0,1] aralığında gezen bir eşik için F1'i maksimize et.
    """
    if not hasattr(clf, "predict_proba"):
        return 0.5, None
    p = clf.predict_proba(X_val)[:, 1]
    best_f1, best_t = -1.0, 0.5
    for t in np.linspace(0.05, 0.95, 19):
        y_hat = (p >= t).astype(int)
        f1 = f1_score(y_val, y_hat, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)


def main(args):
    ensure_dirs()

    train_csv = args.train if args.train else (DATA_DIR / "nsl_kdd_train.csv")
    test_csv  = args.test  if args.test  else (DATA_DIR / "nsl_kdd_test.csv")

    X_train, X_test, y_train, y_test = load_data(str(train_csv), str(test_csv))

    # --- Train/Validation split (threshold seçimi için) ---
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    # --- Model: dengesiz veri için class_weight='balanced' ---
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(X_tr, y_tr)

    # --- En iyi threshold'u doğrulamada seç ---
    best_t, best_f1 = best_threshold_for_f1(clf, X_val, y_val)
    if best_f1 is not None:
        print(f"[threshold] best_t={best_t:.2f} (val F1={best_f1:.4f})")

    # --- Test değerlendirme (threshold uygulayarak) ---
    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= best_t).astype(int)
    else:
        y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, zero_division=0)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

    # --- Model + meta kaydet ---
    joblib.dump(clf, MODEL_PATH)

    # meta: kullanılan gerçek özellik isimleri + threshold
    feat_names = list(getattr(X_train, "columns", FEATURES))
    save_meta(feature_names=feat_names, extra={"threshold": best_t})

    print(f"\n[OK] Model kaydedildi: {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default=None, help="NSL-KDD train CSV yolu")
    parser.add_argument("--test", type=str, default=None, help="NSL-KDD test CSV yolu")
    main(parser.parse_args())
