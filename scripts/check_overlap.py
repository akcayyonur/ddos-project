from pathlib import Path

t = Path("data/nsl_kdd_train.csv").read_text(encoding="utf-8").splitlines()
s = set(t)
u = len(t) - len(s)
print("train total:", len(t), "unique:", len(s), "duplicates in train:", u)

tt = set(Path("data/nsl_kdd_test.csv").read_text(encoding="utf-8").splitlines())
inter = len(s.intersection(tt))
print("train ∩ test overlap lines:", inter)
