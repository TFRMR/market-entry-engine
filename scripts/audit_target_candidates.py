import numpy as np
import pandas as pd

df = pd.read_csv("data/processed/momentum_training_1r_10.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

e = df[df.is_momentum_candle].copy()
e["direction"] = np.where(e.is_bullish, "LONG", "SHORT")
e["period"] = pd.qcut(e.timestamp.rank(method="first"), 4, labels=["P1","P2","P3","P4"])

rows = []
for _, r in e.iterrows():
    i = int(r.name)
    f = df.iloc[i+1:i+6]
    if len(f) < 5:
        continue
    risk = float(r.barrier_risk)
    entry = float(r.barrier_entry)
    close5 = float(f.iloc[-1].close)
    value = (close5-entry)/risk if r.direction == "LONG" else (entry-close5)/risk
    rows.append({"period":r.period,"direction":r.direction,"r5":value})

x = pd.DataFrame(rows)

def s(g):
    r = g.r5
    return pd.Series({
        "n":len(r),
        "mean_R":r.mean(),
        "median_R":r.median(),
        "positive":(r>0).mean(),
        "ge_0.5R":(r>=0.5).mean(),
        "ge_1R":(r>=1.0).mean(),
        "le_-0.5R":(r<=-0.5).mean(),
        "le_-1R":(r<=-1.0).mean()
    })

print("=== OVERALL ===")
print(x.groupby("direction",observed=True).apply(s,include_groups=False).reset_index().to_string(index=False,float_format=lambda v:f"{v:.4f}"))

print("\n=== P1-P4 x DIRECTION ===")
print(x.groupby(["period","direction"],observed=True).apply(s,include_groups=False).reset_index().to_string(index=False,float_format=lambda v:f"{v:.4f}"))

print("\n=== P1-P4 ALL ===")
print(x.groupby("period",observed=True).apply(s,include_groups=False).reset_index().to_string(index=False,float_format=lambda v:f"{v:.4f}"))

print("\n=== TARGET BALANCE ===")
for t in [0.0,0.5,1.0]:
    pos=(x.r5>=t).mean()
    neg=(x.r5<0).mean() if t==0 else (x.r5<=-t).mean()
    print(f"{t:.1f}R: positive={pos:.4f} negative={neg:.4f}")

print("\n=== CHECKS ===")
if x.empty or not np.isfinite(x.r5).all() or not x.direction.isin(["LONG","SHORT"]).all():
    raise SystemExit("Audit check failed.")
print(f"Events evaluated: {len(x)}")
print("Checks passed.")
