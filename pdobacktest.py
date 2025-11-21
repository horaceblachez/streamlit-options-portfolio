import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import timedelta
import altair as alt

st.set_page_config(page_title="PDO backtest (FV)", layout="wide")

@st.cache_data(ttl=6*3600)
def get_sp500():
    try:
        t = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        s = t["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
        if "SPY" not in s:
            s.append("SPY")
        return sorted(list(dict.fromkeys(s)))
    except:
        return ["SPY", "AAPL", "MSFT", "NVDA", "META", "AMZN", "GOOGL", "SPY"]

@st.cache_data(ttl=6*3600)
def get_history(ticker, start, end):
    d = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if d is None or d.empty:
        return pd.DataFrame()
    d = d.rename(columns=str.title)
    return d[["Open", "High", "Low", "Close"]].dropna().copy()

@st.cache_data(ttl=6*3600)
def get_irx_curve(start, end):
    r = yf.download("^IRX", start=start, end=end, progress=False, auto_adjust=False)
    if r is None or r.empty:
        idx = pd.bdate_range(start.normalize(), end.normalize())
        s = pd.Series(0.02, index=idx, name="r_ann")
        cum = (s / 365).cumsum()
        return pd.DataFrame({"r_ann": s, "cum": cum})
    c = r.get("Close", None)
    if c is None:
        c = r.get("Adj Close", None)
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    c = pd.to_numeric(c, errors="coerce") / 100.0
    c.name = "r_ann"
    c = c.dropna()
    c.index = pd.to_datetime(c.index).tz_localize(None)
    idx = pd.bdate_range(start.normalize(), end.normalize())
    c = c.reindex(idx).ffill().fillna(0.02)
    cum = (c / 365).cumsum()
    return pd.DataFrame({"r_ann": c, "cum": cum})

def roll_windows_index(dates, tenor_days):
    idx = pd.Index(dates)
    for i, d0 in enumerate(dates):
        j = int(idx.searchsorted(d0 + timedelta(days=tenor_days), side="left"))
        if j < len(dates):
            yield i, j

def compute(df, strike_pct=0.05, barrier_pct=0.10, tenor_days=90, contract_size=100):
    dates = df.index.to_list()
    out = []
    for i0, i1 in roll_windows_index(dates, tenor_days):
        d0, d1 = dates[i0], dates[i1]
        S0, ST = float(df["Close"].iloc[i0]), float(df["Close"].iloc[i1])
        K, B = S0 * (1 - strike_pct), S0 * (1 - barrier_pct)
        knocked = bool(np.any(df["Low"].iloc[i0:i1 + 1].to_numpy() <= B))
        v, o = max(K - ST, 0.0), 0.0 if knocked else max(K - ST, 0.0)
        out.append({
            "Start": d0,
            "End": d1,
            "S0": S0,
            "K": K,
            "B": B,
            "ST": ST,
            "KO": knocked,
            "Vanilla_USD": contract_size * v,
            "OutPut_USD": contract_size * o
        })
    return pd.DataFrame(out).set_index("End")

st.title("%premium of down-out put backtester")

st.markdown(
    "The point of this backtest is to find historic ratio of payoffs from Down and Out Put compared to vanilla puts to give appropriate price ratio of DOP compared to put today. It simulates buying everyday a DOP and vanilla put for a certain tenor, strike offset and barrier offset. It then calculates payoffs for each and brings payoff to current value using 13 week Treasury bill historic rates over time of payoff to today. It then gives average payoff of vanilla put and of DOP in todays value to get appropriate price ratio of DOP to vanilla put today. (one contract here contains 100 stock shares underlying)"
)

tickers = get_sp500()
c0, c1, c2 = st.columns([1.4, 1, 1])
ticker = c0.selectbox("Underlying", tickers, index=tickers.index("SPY") if "SPY" in tickers else 0)
years_back = c1.slider("History window (years)", 1, 15, 5)
tenor_days = c2.slider("Tenor (days)", 30, 365, 90, step=5)
c3, c4 = st.columns(2)
strike_pct = c3.number_input(
    "Strike offset (0.05=5%)",
    value=0.05,
    min_value=0.0,
    max_value=0.9,
    step=0.01,
    format="%.2f"
)
barrier_pct = c4.number_input(
    "Barrier offset (0.10=10%)",
    value=0.10,
    min_value=0.0,
    max_value=0.95,
    step=0.01,
    format="%.2f"
)

today = pd.Timestamp.today(tz="UTC").normalize().tz_localize(None)
start = (today - pd.DateOffset(years=years_back)).tz_localize(None)
hist = get_history(ticker, start=start, end=today)
if hist.empty:
    st.error("No data.")
    st.stop()

res = compute(hist, strike_pct, barrier_pct, tenor_days, 100)
if res.empty:
    st.warning("Not enough data.")
    st.stop()

irx = get_irx_curve(start, today)
cum = irx["cum"]
cum_T = float(cum.iloc[-1])
cum_on_end = cum.reindex(res.index, method="ffill")
factor_to_today = np.exp(cum_T - cum_on_end)
res["FV_OutPut_USD"] = res["OutPut_USD"] * factor_to_today.values
res["FV_Vanilla_USD"] = res["Vanilla_USD"] * factor_to_today.values

avg_out = res["FV_OutPut_USD"].mean()
avg_van = res["FV_Vanilla_USD"].mean()
ratio = (avg_out / avg_van) if avg_van > 0 else np.nan
summ = pd.DataFrame({
    "Avg Down-out Put (USD/contract)": [avg_out],
    "Avg Vanilla (USD/contract)": [avg_van],
    "DOP/Vanilla Ratio": [ratio]
}).T

st.subheader("Rolling Payoffs (today's USD per contract)")
left, right = st.columns(2)

with left:
    ts = res[["FV_OutPut_USD", "FV_Vanilla_USD"]].rename(
        columns={"FV_OutPut_USD": "Down-and-Out Put", "FV_Vanilla_USD": "Vanilla"}
    )
    plot_df = ts.copy()
    plot_df["End"] = plot_df.index
    long_df = plot_df.melt(id_vars="End", var_name="Type", value_name="Payoff")
    base = alt.Chart(long_df).encode(
        x=alt.X("End:T", title="Maturity date"),
        y=alt.Y("Payoff:Q", title="Value in today's USD / contract"),
        color=alt.Color(
            "Type:N",
            scale=alt.Scale(range=["#ff7f0e", "#1f77b4"])
        ),
        tooltip=[alt.Tooltip("End:T"), "Type:N", alt.Tooltip("Payoff:Q", format=",.0f")]
    )
    chart = (
        base.transform_filter(alt.datum.Type == "Vanilla").mark_line(size=2) +
        base.transform_filter(alt.datum.Type == "Down-and-Out Put").mark_line(size=2, strokeDash=[6, 3]) +
        base.transform_filter(alt.datum.Type == "Down-and-Out Put").mark_circle(size=26, opacity=0.6)
    ).properties(height=280).interactive()
    st.altair_chart(chart, use_container_width=True)

    st.markdown("**Average Relative Value (today's USD)**")
    st.table(summ.style.format("{:,.2f}"))

with right:
    stats = pd.DataFrame({
        "Mean": ts.mean(),
        "Median": ts.median(),
        "Std": ts.std(),
        "Hit rate (>0)": (ts > 0).mean()
    })
    st.table(
        stats.style.format({
            "Mean": "{:.0f}",
            "Median": "{:.0f}",
            "Std": "{:.0f}",
            "Hit rate (>0)": "{:.4f}"
        })
    )

    
    m1, m2 = st.columns(2)
    m1.metric("frequency DOP hit barrier (%)", f"{100 * res['KO'].mean():.1f}%")
    m2.metric("DOP/Vanilla Ratio", f"{ratio:.2f}" if np.isfinite(ratio) else "N/A")

st.subheader("Recent Trades (last 10)")
show = res.tail(10).copy()
st.dataframe(
    show[[
        "Start", "S0", "K", "B", "ST", "KO",
        "FV_OutPut_USD", "FV_Vanilla_USD"
    ]].style.format({
        "S0": "{:.2f}",
        "K": "{:.2f}",
        "B": "{:.2f}",
        "ST": "{:.2f}",
        "FV_OutPut_USD": "{:.0f}",
        "FV_Vanilla_USD": "{:.0f}"
    }),
    use_container_width=True
)
