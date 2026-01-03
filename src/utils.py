from pathlib import Path
import json
from .config import MODEL_DIR, META_PATH, FEATURES


def ensure_dirs():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def save_meta(feature_names=None, extra=None):
    """
    Model meta bilgisini kaydeder.
      - feature_order: eğitimde kullanılan gerçek sütun listesi (türevler dahil)
      - threshold: (varsa) seçilen karar eşiği
      - label_*: etiket sözleşmesi
      - notes: entegrasyon notu
    """
    feats = list(feature_names) if feature_names is not None else list(FEATURES)
    meta = {
        "feature_order": feats,
        "label_positive": 1,
        "label_negative": 0,
        "notes": "Ryu tarafında feature_order sırasıyla numerik vektör üretmelisin."
    }

    if extra:
        meta.update(extra)

    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

