"""Generate a synthetic customer + transaction dataset for a Baku coffee-chain loyalty app.

Customers sign up (with their first purchase) between 2026-01-01 and 2026-08-31.
Transactions are observed until 2026-09-30.

Built-in behaviour (the "truth" the analysis should uncover):
  * acquisition channel drives lifetime: referral > organic > instagram ads > free-coffee voucher
  * a "loyalty stamps" feature launched on 2026-05-01 makes later cohorts stickier
  * engaged customers (high visit rate) also churn more slowly
  * slight summer dip in visit frequency (Jul–Aug)
Deliberate data-quality problems: duplicate rows, refunds, staff test accounts,
missing channel values and transactions time-stamped before signup.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2026
rng = np.random.default_rng(SEED)
OUT = Path(__file__).parent

START, LAST_SIGNUP, END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-08-31"), pd.Timestamp("2026-09-30")
FEATURE_LAUNCH = pd.Timestamp("2026-05-01")

CHANNELS = ["organic", "referral", "instagram_ads", "free_coffee_voucher"]
CH_SHARE = [0.33, 0.14, 0.31, 0.22]
CH_LIFETIME = {"organic": 150, "referral": 200, "instagram_ads": 95, "free_coffee_voucher": 42}  # mean days
CH_RATE = {"organic": 1.25, "referral": 1.35, "instagram_ads": 1.05, "free_coffee_voucher": 0.95}  # visits / week
DISTRICTS = ["Nasimi", "Yasamal", "Narimanov", "Sabail", "Khatai", "Binagadi"]
DIST_SHARE = [0.25, 0.19, 0.17, 0.15, 0.14, 0.10]

# --- signups: growing month over month --------------------------------------
days = pd.date_range(START, LAST_SIGNUP, freq="D")
t = np.arange(len(days))
daily_signups = rng.poisson(48 + 0.17 * t + 6 * (days.dayofweek >= 5))
signup_dates = np.repeat(days.values, daily_signups)
n = len(signup_dates)

cust = pd.DataFrame({
    "customer_id": [f"C{100000 + i}" for i in range(n)],
    "signup_date": pd.to_datetime(signup_dates),
    "channel": rng.choice(CHANNELS, n, p=CH_SHARE),
    "district": rng.choice(DISTRICTS, n, p=DIST_SHARE),
    "age": np.clip(rng.normal(29, 7, n).round(), 17, 65).astype(int),
})

# --- latent behaviour ---------------------------------------------------------
rate = cust.channel.map(CH_RATE).values * rng.gamma(2.2, 1 / 2.2, n)          # visits / week
engagement = np.log(rate / cust.channel.map(CH_RATE).values)
life_mean = cust.channel.map(CH_LIFETIME).values * np.exp(0.45 * engagement)
life_mean *= np.where(cust.signup_date >= FEATURE_LAUNCH, 1.40, 1.0)
lifetime = rng.exponential(life_mean * rng.gamma(0.9, 1 / 0.9, n))             # days until churn
basket = rng.lognormal(np.log(6.2), 0.18, n)                                    # avg AZN per visit

# --- transactions -------------------------------------------------------------
rows = []
for i in range(n):
    s = cust.signup_date.iat[i]
    active_end = min(s + pd.Timedelta(days=float(lifetime[i])), END)
    span = (active_end - s).days
    k = rng.poisson(rate[i] * span / 7) if span > 0 else 0
    offs = np.sort(rng.uniform(0, span, k)) if k else np.array([])
    dates = [s] + [s + pd.Timedelta(days=int(o)) for o in offs]
    # summer dip: drop ~15% of Jul–Aug visits
    dates = [d for j, d in enumerate(dates) if j == 0 or d.month not in (7, 8) or rng.random() > 0.15]
    amt = rng.lognormal(np.log(basket[i]), 0.35, len(dates))
    if cust.channel.iat[i] == "free_coffee_voucher":
        amt[0] = max(amt[0] - 4.5, 0.0)     # first coffee free
    for d, a in zip(dates, amt):
        rows.append((cust.customer_id.iat[i], d, round(float(a), 2)))

tx = pd.DataFrame(rows, columns=["customer_id", "txn_date", "amount_azn"])
tx["store"] = tx.customer_id.map(cust.set_index("customer_id").district) + "-" + rng.choice(["01", "02"], len(tx))

# --- data-quality problems ----------------------------------------------------
# 1) staff test accounts with lots of transactions
test = pd.DataFrame({
    "customer_id": [f"TEST{i:03d}" for i in range(25)],
    "signup_date": pd.Timestamp("2026-01-02"), "channel": "organic",
    "district": "Nasimi", "age": 30,
})
test_tx = pd.DataFrame({
    "customer_id": rng.choice(test.customer_id, 1500),
    "txn_date": pd.Timestamp("2026-01-02") + pd.to_timedelta(rng.integers(0, 270, 1500), "D"),
    "amount_azn": rng.choice([0.01, 1.0, 0.0], 1500), "store": "Nasimi-01",
})
cust = pd.concat([cust, test], ignore_index=True)
tx = pd.concat([tx, test_tx], ignore_index=True)
tx.insert(0, "txn_id", [f"T{i:07d}" for i in range(len(tx))])
# 2) missing channel
cust.loc[rng.choice(n, int(0.015 * n), replace=False), "channel"] = np.nan
# 3) refunds (negative copies of real purchases)
ref = tx[tx.amount_azn > 1].sample(320, random_state=SEED).copy()
ref["txn_id"] = [f"R{i:07d}" for i in range(len(ref))]
ref["amount_azn"] *= -1
ref["txn_date"] = (ref.txn_date + pd.to_timedelta(rng.integers(0, 3, len(ref)), "D")).clip(upper=END)
# 4) exact duplicate rows (double-posted by POS sync)
dup = tx.sample(int(0.006 * len(tx)), random_state=SEED + 1)
# 5) clock errors: dates before signup
bad = tx.sample(150, random_state=SEED + 2).copy()
bad["txn_id"] = [f"X{i:07d}" for i in range(len(bad))]
bad["txn_date"] -= pd.Timedelta(days=400)
tx = pd.concat([tx, ref, dup, bad], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)

cust.to_csv(OUT / "customers.csv", index=False, date_format="%Y-%m-%d")
tx.to_csv(OUT / "transactions.csv", index=False, date_format="%Y-%m-%d")
print(f"customers: {len(cust):,}  transactions: {len(tx):,}")
