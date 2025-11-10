import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Sales Forecasting")

st.title("Sales Forecasting with SARIMAX")

st.write("Upload file CSV berisi kolom Order Date dan Sales. Atau pakai sample bawaan.")

uploaded = st.file_uploader("Upload train.csv", type=["csv"])

# baca file
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    try:
        df = pd.read_csv(os.path.join("data", "raw", "train.csv"))
    except FileNotFoundError:
        st.warning("File train.csv tidak ditemukan, silakan upload di bawah.")
        uploaded = st.file_uploader("Upload train.csv", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
        else:
            st.stop()

# normalisasi kolom
cols = {c: c.strip() for c in df.columns}
df.rename(columns=cols, inplace=True)

if "Order Date" not in df.columns or "Sales" not in df.columns:
    st.error("Kolom yang dibutuhkan: Order Date dan Sales")
    st.stop()

# parse tanggal dan numerik
df["date"] = pd.to_datetime(df["Order Date"], errors="coerce")
df = df.dropna(subset=["date"])
df["Sales"] = (
    df["Sales"]
    .astype(str)
    .str.replace(r"[^\d\.\-]", "", regex=True)
    .astype(float)
)

# agregasi bulanan
monthly = (
    df.groupby(pd.Grouper(key="date", freq="ME"))["Sales"]
    .sum()
    .reset_index()
    .rename(columns={"Sales": "sales"})
)

st.subheader("Monthly sales trend")
fig1, ax1 = plt.subplots(figsize=(10,4))
ax1.plot(monthly["date"], monthly["sales"], marker="o")
ax1.set_xlabel("Date")
ax1.set_ylabel("Total Sales")
ax1.grid(True)
st.pyplot(fig1)

# train test split
h = st.slider("Horizon bulan untuk test", 3, 12, 6)
ts = monthly.set_index("date")["sales"].asfreq("ME").fillna(0)
if len(ts) <= h + 15:
    st.warning("Data terlalu sedikit untuk evaluasi, kurangi horizon atau gunakan dataset lebih panjang")
train_ts = ts.iloc[:-h]
test_ts  = ts.iloc[-h:]

# fit sarimax
order_p = st.number_input("AR order p", min_value=0, value=1, step=1)
order_d = st.number_input("Diff d", min_value=0, value=1, step=1)
order_q = st.number_input("MA order q", min_value=0, value=1, step=1)
seasonal = st.checkbox("Seasonal monthly", value=True)

if seasonal:
    model = SARIMAX(train_ts, order=(order_p, order_d, order_q),
                    seasonal_order=(1,1,1,12),
                    enforce_stationarity=False, enforce_invertibility=False)
else:
    model = SARIMAX(train_ts, order=(order_p, order_d, order_q),
                    enforce_stationarity=False, enforce_invertibility=False)

res = model.fit(disp=False)

# forecast
pred = res.get_forecast(steps=h)
pred_mean = pred.predicted_mean
pred_ci = pred.conf_int()

mae = float(mean_absolute_error(test_ts, pred_mean))
rmse = float(np.sqrt(mean_squared_error(test_ts, pred_mean)))

st.subheader("Forecast vs actual")
fig2, ax2 = plt.subplots(figsize=(10,4))
ax2.plot(train_ts.index, train_ts, label="Train")
ax2.plot(test_ts.index, test_ts, label="Actual")
ax2.plot(pred_mean.index, pred_mean, label="Forecast")
ax2.fill_between(pred_ci.index, pred_ci.iloc[:,0], pred_ci.iloc[:,1], alpha=0.2, label="95% CI")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

st.write(f"MAE: {mae:,.2f}   RMSE: {rmse:,.2f}")

st.subheader("Comparison table")
cmp = pd.DataFrame({"actual": test_ts, "forecast": pred_mean})
st.dataframe(cmp)
