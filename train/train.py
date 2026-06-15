"""
离线训练「数据模型」：从 ~4.9 万场历史国际比赛学习 期望进球(λ) 映射。
小型 MLP（7→16→8→2，泊松 NLL），纯 numpy 手写。产出 core/model.json（端内纯 JS 推理用）。
运行：python train/train.py
"""
import json, math, datetime, os, sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ---------- 1. 读数据 ----------
df = pd.read_csv(os.path.join(HERE, "results.csv"))
df = df.dropna(subset=["home_score", "away_score"])
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
df["home_score"] = df["home_score"].astype(int)
df["away_score"] = df["away_score"].astype(int)
df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")

def importance(t):
    t = str(t).lower()
    if "friendly" in t: return 0
    if "qualif" in t: return 1
    if any(k in t for k in ["fifa world cup", "uefa euro", "copa am", "african cup",
                            "confederations", "asian cup", "gold cup", "nations league"]):
        return 2 if "qualif" not in t else 1
    return 1

df["imp"] = df["tournament"].map(importance)

# ---------- 2. 历史 Elo（全史预热，从 1998 起采样）----------
elo = {}
recent = {}   # team -> list of recent points (last 5)
BASE = 1500.0
def get_elo(t): return elo.get(t, BASE)
def get_form(t):
    r = recent.get(t, [])
    return sum(r) / (len(r) * 3) if r else 0.5   # 0~1, 默认 0.5

K_BY_IMP = [20.0, 35.0, 55.0]
HA = 65.0   # 主场 Elo 加成（非中立）

rows = []
SAMPLE_FROM = pd.Timestamp("1998-01-01")
RATINGS_ASOF = pd.Timestamp("2026-06-11")   # 导出 ratings 的赛前截止（WC2026 开赛前；防把已赛 WC 比分算进基线）
elo_asof, form_asof = {}, {}                 # 各队截至赛前的 Elo / 近期状态率快照

for _, m in df.iterrows():
    h, a = m["home_team"], m["away_team"]
    eh, ea = get_elo(h), get_elo(a)
    fh, fa = get_form(h), get_form(a)
    ha = 0.0 if m["neutral"] else HA
    # 采样（赛前特征 → 该场比分）；A2: 剔除友谊赛(imp==0)减噪，仅取正式比赛
    if m["date"] >= SAMPLE_FROM and m["imp"] >= 1:
        rows.append({
            "elo_h": eh, "elo_a": ea, "diff": (eh + ha) - ea,
            "neutral": 1.0 if m["neutral"] else 0.0,
            "imp": float(m["imp"]), "form_h": fh, "form_a": fa,
            "gh": m["home_score"], "ga": m["away_score"],
            "date": m["date"]
        })
    # 更新 Elo
    gh, ga = m["home_score"], m["away_score"]
    We = 1.0 / (1.0 + 10 ** (-((eh + ha) - ea) / 400.0))
    S = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)
    gd = abs(gh - ga)
    gdmul = 1.0 if gd <= 1 else (1.5 if gd == 2 else (1.75 if gd == 3 else 1.75 + (gd - 3) / 8.0))
    K = K_BY_IMP[int(m["imp"])]
    delta = K * gdmul * (S - We)
    elo[h] = eh + delta
    elo[a] = ea - delta
    for t, pts in ((h, 3 if gh > ga else (1 if gh == ga else 0)),
                   (a, 3 if ga > gh else (1 if gh == ga else 0))):
        r = recent.get(t, []); r.append(pts); recent[t] = r[-5:]
    # 赛前快照（截至 WC2026 开赛前的最新一场）
    if m["date"] < RATINGS_ASOF:
        elo_asof[h] = elo[h]; elo_asof[a] = elo[a]
        form_asof[h] = get_form(h); form_asof[a] = get_form(a)

data = pd.DataFrame(rows)
print(f"采样比赛数(1998+): {len(data)}")

# ---------- 3. 特征 / 目标 ----------
FEATS = ["elo_h", "elo_a", "diff", "neutral", "imp", "form_h", "form_a"]
X = data[FEATS].values.astype(np.float64)
Yh = data["gh"].values.astype(np.float64)
Ya = data["ga"].values.astype(np.float64)
Y = np.stack([Yh, Ya], axis=1)

# 时间留出：<2022 训练，>=2022 测试
test_mask = (data["date"] >= pd.Timestamp("2022-01-01")).values
Xtr, Xte = X[~test_mask], X[test_mask]
Ytr, Yte = Y[~test_mask], Y[test_mask]
print(f"训练 {len(Xtr)} / 测试 {len(Xte)}")

mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1
Xtr_n = (Xtr - mu) / sd
Xte_n = (Xte - mu) / sd

# ---------- 4. MLP（numpy + Adam, 泊松 NLL）----------
rng = np.random.default_rng(42)
def init(shape): return rng.standard_normal(shape) * math.sqrt(2.0 / shape[0])
P = {
    "W1": init((7, 16)), "b1": np.zeros(16),
    "W2": init((16, 8)), "b2": np.zeros(8),
    "W3": init((8, 2)) * 0.1, "b3": np.array([0.2, 0.0]),  # 主场略偏多
}
def relu(z): return np.maximum(0, z)
def forward(Xn):
    z1 = Xn @ P["W1"] + P["b1"]; a1 = relu(z1)
    z2 = a1 @ P["W2"] + P["b2"]; a2 = relu(z2)
    z3 = a2 @ P["W3"] + P["b3"]                # log λ
    z3 = np.clip(z3, -2.5, 2.0)
    return z1, a1, z2, a2, z3
def poisson_nll(z3, Yb):
    lam = np.exp(z3)
    return np.mean(lam - Yb * z3)

m = {k: np.zeros_like(v) for k, v in P.items()}
v = {k: np.zeros_like(v) for k, v in P.items()}
b1f, b2f, eps, lr = 0.9, 0.999, 1e-8, 0.01
EPOCHS, BS = 260, 512
n = len(Xtr_n); t = 0
for ep in range(EPOCHS):
    idx = rng.permutation(n)
    for s in range(0, n, BS):
        bi = idx[s:s + BS]; Xb = Xtr_n[bi]; Yb = Ytr[bi]
        z1, a1, z2, a2, z3 = forward(Xb)
        lam = np.exp(z3)
        g3 = (lam - Yb) / len(bi)             # dL/dz3
        gW3 = a2.T @ g3; gb3 = g3.sum(0)
        ga2 = g3 @ P["W3"].T; gz2 = ga2 * (z2 > 0)
        gW2 = a1.T @ gz2; gb2 = gz2.sum(0)
        ga1 = gz2 @ P["W2"].T; gz1 = ga1 * (z1 > 0)
        gW1 = Xb.T @ gz1; gb1 = gz1.sum(0)
        grads = {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2, "W3": gW3, "b3": gb3}
        t += 1
        for k in P:
            m[k] = b1f * m[k] + (1 - b1f) * grads[k]
            v[k] = b2f * v[k] + (1 - b2f) * grads[k] ** 2
            mh = m[k] / (1 - b1f ** t); vh = v[k] / (1 - b2f ** t)
            P[k] -= lr * mh / (np.sqrt(vh) + eps)
    if ep % 40 == 0 or ep == EPOCHS - 1:
        _, _, _, _, z3tr = forward(Xtr_n)
        _, _, _, _, z3te = forward(Xte_n)
        print(f"  ep{ep:3d} train NLL {poisson_nll(z3tr, Ytr):.4f} | test NLL {poisson_nll(z3te, Yte):.4f}")

# ---------- 5. 回测：模型 vs 基线手调公式（W/D/L logloss）----------
DC_RHO = -0.13
def score_matrix(lh, la, maxg=8):
    fac = np.array([math.factorial(k) for k in range(maxg + 1)])
    ph = np.exp(-lh) * lh ** np.arange(maxg + 1) / fac
    pa = np.exp(-la) * la ** np.arange(maxg + 1) / fac
    M = np.outer(ph, pa)
    M[0, 0] *= 1 - lh * la * DC_RHO; M[0, 1] *= 1 + lh * DC_RHO
    M[1, 0] *= 1 + la * DC_RHO; M[1, 1] *= 1 - DC_RHO
    return M / M.sum()
def wdl(M):
    w = np.tril(M, -1).sum(); d = np.trace(M); l = np.triu(M, 1).sum()
    return w, d, l
def logloss_set(lam_fn):
    tot = 0.0
    for i in range(len(Xte)):
        lh, la = lam_fn(i)
        w, d, l = wdl(score_matrix(lh, la))
        gh, ga = Yte[i]
        p = w if gh > ga else (d if gh == ga else l)
        tot += -math.log(max(p, 1e-9))
    return tot / len(Xte)
# 模型 λ
_, _, _, _, z3te = forward(Xte_n)
lam_te = np.exp(z3te)
model_ll = logloss_set(lambda i: (lam_te[i, 0], lam_te[i, 1]))
# 基线：λ = 1.30 * exp(0.0011 * diff)（与现 engine 同）
def baseline(i):
    d = Xte[i, 2]  # diff (含主场)
    return 1.30 * math.exp(0.0011 * d), 1.30 * math.exp(-0.0011 * d)
base_ll = logloss_set(baseline)
print(f"\n回测 W/D/L logloss —— 数据模型: {model_ll:.4f} | 基线公式: {base_ll:.4f} | "
      f"{'✅ 模型更优' if model_ll <= base_ll else '⚠️ 未超基线'}")

# ---------- 6. 导出 core/model.json ----------
out = {
    "feats": FEATS,
    "mean": mu.tolist(), "std": sd.tolist(),
    "layers": [
        {"W": P["W1"].T.tolist(), "b": P["b1"].tolist(), "act": "relu"},
        {"W": P["W2"].T.tolist(), "b": P["b2"].tolist(), "act": "relu"},
        {"W": P["W3"].T.tolist(), "b": P["b3"].tolist(), "act": "linear"},
    ],
    "clip": [-2.5, 2.0],
    "meta": {
        "trained": datetime.date.today().isoformat(),
        "samples": int(len(data)),
        "test_logloss": round(float(model_ll), 4),
        "baseline_logloss": round(float(base_ll), 4),
    },
}
payload = json.dumps(out, ensure_ascii=False, separators=(",", ":"))
js = ("// 数据模型权重（离线训练自约5万场历史国际比赛，端内推理，无外部调用）\n"
      "var WC_MODEL=" + payload + ";\n"
      "if(typeof module!==\"undefined\"&&module.exports)module.exports=WC_MODEL;"
      "else if(typeof window!==\"undefined\")window.WC_MODEL=WC_MODEL;")
with open(os.path.join(ROOT, "core", "model.js"), "w", encoding="utf-8") as f:
    f.write(js)
sz = os.path.getsize(os.path.join(ROOT, "core", "model.js"))
print(f"✅ 导出 core/model.js （{sz} 字节，{len(data)} 场训练）")

# ---------- 7. 导出 core/ratings.js（赛前同源 Elo/form，apply-ratings.js 写回 data.js）----------
# 与训练同一套时序 Elo 同标度，消除「线上 eloratings 标度 ≠ 训练标度」导致的标准化错位。
ratings = {}
for t in sorted(elo.keys()):
    e = elo_asof.get(t, elo.get(t, BASE))
    fr = form_asof.get(t, 0.5)
    ratings[t] = {"elo": round(float(e), 1), "form": round(float(fr), 4)}
rj = json.dumps(ratings, ensure_ascii=False, separators=(",", ":"))
rmeta = json.dumps({"asof": RATINGS_ASOF.date().isoformat(), "teams": len(ratings),
                    "source_rows": int(len(df))}, ensure_ascii=False, separators=(",", ":"))
rjs = ("// 训练同源评分（赛前 Elo + 近期状态率 form 0~1）——离线产出，apply-ratings.js 写回 core/data.js\n"
       "var WC_RATINGS=" + rj + ";\nvar WC_RATINGS_META=" + rmeta + ";\n"
       "if(typeof module!==\"undefined\"&&module.exports)module.exports={ratings:WC_RATINGS,meta:WC_RATINGS_META};\n"
       "else if(typeof window!==\"undefined\"){window.WC_RATINGS=WC_RATINGS;window.WC_RATINGS_META=WC_RATINGS_META;}")
with open(os.path.join(ROOT, "core", "ratings.js"), "w", encoding="utf-8") as f:
    f.write(rjs)
asof_n = sum(1 for t in elo if t in elo_asof)
print(f"✅ 导出 core/ratings.js （{len(ratings)} 队，赛前截止 {RATINGS_ASOF.date()}，含赛前快照 {asof_n} 队）")
