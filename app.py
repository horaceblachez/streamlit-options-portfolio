# app.py — Simple S&P 500 Options Calculator & Portfolio (clean greeks + %move P&L)
import math
from dataclasses import dataclass
from typing import Literal, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.stats import norm

st.set_page_config(page_title="S&P 500 Options — Simple Calculator & Portfolio", layout="wide")

# ===================== Black–Scholes =====================
OptionType = Literal["Call", "Put"]

def _d1(S, K, r, q, sigma, T):
    if any(v <= 0 for v in [S, K, sigma, T]): return np.nan
    return (math.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))

def _d2(d1, sigma, T): return d1 - sigma*math.sqrt(T)

def bs_price(S, K, r, q, sigma, T, kind: OptionType) -> float:
    d1 = _d1(S,K,r,q,sigma,T); d2 = _d2(d1,sigma,T)
    if not np.isfinite(d1): return float("nan")
    if kind == "Call":
        return S*math.exp(-q*T)*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
    else:
        return K*math.exp(-r*T)*norm.cdf(-d2) - S*math.exp(-q*T)*norm.cdf(-d1)

def bs_greeks(S, K, r, q, sigma, T, kind: OptionType) -> dict:
    d1 = _d1(S,K,r,q,sigma,T); d2 = _d2(d1,sigma,T)
    if not np.isfinite(d1):
        return {"Delta": np.nan, "Gamma": np.nan, "Vega_1pct": np.nan, "Theta_day": np.nan, "Rho": np.nan}
    pdf = norm.pdf(d1); disc_q = math.exp(-q*T); disc_r = math.exp(-r*T); sqrtT = math.sqrt(T)
    delta = disc_q*norm.cdf(d1) if kind=="Call" else disc_q*(norm.cdf(d1)-1)
    gamma = disc_q*pdf/(S*sigma*sqrtT)
    vega_1pct = (S*disc_q*pdf*sqrtT)/100.0
    if kind=="Call":
        theta_pa = (-S*disc_q*pdf*sigma)/(2*sqrtT) - r*K*disc_r*norm.cdf(d2) + q*S*disc_q*norm.cdf(d1)
        rho = K*T*disc_r*norm.cdf(d2)
    else:
        theta_pa = (-S*disc_q*pdf*sigma)/(2*sqrtT) + r*K*disc_r*norm.cdf(-d2) - q*S*disc_q*norm.cdf(-d1)
        rho = -K*T*disc_r*norm.cdf(-d2)
    return {"Delta": delta, "Gamma": gamma, "Vega_1pct": vega_1pct, "Theta_day": theta_pa/365.0, "Rho": rho}

# ===================== Live data helpers =====================
@st.cache_data(ttl=6*3600)
def get_sp500() -> List[str]:
    try:
        df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        syms = df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
        syms.append("SPY")
        return sorted(list(dict.fromkeys(syms)))
    except Exception:
        return ["SPY","AAPL","MSFT","NVDA","AMZN","META","GOOGL"]

@st.cache_data(ttl=300)
def get_spot(ticker: str) -> Optional[float]:
    h = yf.Ticker(ticker).history(period="1d")
    if h.empty: return None
    return float(h["Close"].iloc[-1])

@st.cache_data(ttl=300)
def get_risk_free() -> float:
    try:
        x = yf.Ticker("^IRX").history(period="5d")["Close"].dropna()
        return float(x.iloc[-1])/100.0 if not x.empty else 0.02
    except Exception:
        return 0.02

@st.cache_data(ttl=3600)
def get_div_yield(ticker: str) -> float:
    try:
        fi = yf.Ticker(ticker).fast_info
        dy = fi.get("dividend_yield", 0.0) if isinstance(fi, dict) else 0.0
        return float(dy or 0.0)
    except Exception:
        return 0.0

@st.cache_data(ttl=120)
def get_expiries(ticker: str) -> List[str]:
    try:
        return yf.Ticker(ticker).options
    except Exception:
        return []

# Chain with IV normalization (fixes 0-ish IV leading to 0 greeks)
@st.cache_data(ttl=120)
def get_chain(ticker: str, expiry: str) -> pd.DataFrame:
    oc = yf.Ticker(ticker).option_chain(expiry)
    def _prep(df, typ):
        if df is None or df.empty: return pd.DataFrame()
        out = df.copy()
        out["type"] = typ
        out["mid"] = (out["bid"].fillna(0.0) + out["ask"].fillna(0.0))/2.0
        if "impliedVolatility" in out.columns:
            iv = pd.to_numeric(out["impliedVolatility"], errors="coerce")
            valid = iv.dropna()
            # if most are <2%, assume yfinance mis-scaled by 100
            if len(valid) >= 10 and (valid < 0.02).mean() > 0.6:
                iv = iv * 100.0
            out["impliedVolatility"] = iv.clip(lower=0.01, upper=5.0)
        return out
    return pd.concat([_prep(oc.calls,"Call"), _prep(oc.puts,"Put")], ignore_index=True)

@st.cache_data(ttl=120)
def get_atm_iv(ticker: str, spot: float) -> Optional[float]:
    exps = get_expiries(ticker)
    if not exps: return None
    for e in exps[:3]:
        ch = get_chain(ticker, e)
        if ch.empty: continue
        ch["dist"] = (ch["strike"] - spot).abs()
        ivs = ch.sort_values("dist")["impliedVolatility"].dropna()
        if not ivs.empty:
            return float(np.clip(ivs.iloc[0], 0.01, 5.0))
    return None

def years_to_expiry(exp_str: str) -> float:
    exp = pd.to_datetime(exp_str)
    now = pd.Timestamp.utcnow().tz_localize(None)
    dt = exp - now
    return max((dt.days + dt.seconds/86400.0) / 365.0, 1e-6)

# ===================== Portfolio lines =====================
@dataclass
class OptLine:
    ticker: str; type: OptionType; expiry: str; strike: float; qty: int
    S: float; r: float; q: float; iv: float; T: float

@dataclass
class StockLine:
    ticker: str; qty: int; S: float

if "portfolio" not in st.session_state:
    st.session_state.portfolio: List[object] = []

# ===================== UI =====================
tickers = get_sp500()
st.title("S&P 500 Options — Options Calculator & Portfolio")

tabs = st.tabs(["Options Calculator", "Portfolio greeks"])

# -------- Calculator (simple) --------
with tabs[0]:
    st.subheader("Option Calculator (BSM)")
    c0, c1, c2, c3, c4, c5 = st.columns([1.4, 1, 1, 1, 1, 1])
    ticker = c0.selectbox("Underlying", tickers)
    S0 = get_spot(ticker) or 100.0
    r0 = get_risk_free(); q0 = get_div_yield(ticker); iv0 = get_atm_iv(ticker, S0) or 0.20
    st.caption(f"Defaults → Spot={S0:.2f}, r={r0:.3f}, q={q0:.3f}, ATM IV={iv0:.2%}")

    kind = c1.selectbox("Type", ["Call", "Put"])
    S    = c2.number_input("Spot S", value=float(S0))
    K    = c3.number_input("Strike K", value=float(round(S0)))
    T    = c4.number_input("Time to Expiry (years)", value=0.25, step=0.05, min_value=1e-4)
    sigma= c5.number_input("Implied Vol σ", value=float(iv0), min_value=0.01, step=0.01)
    col_r, col_q = st.columns(2)
    r = col_r.number_input("Risk-free r", value=float(r0), step=0.002)
    q = col_q.number_input("Dividend/borrow yield q", value=float(q0), step=0.002)

    pv = bs_price(S, K, r, q, sigma, T, kind)
    g  = bs_greeks(S, K, r, q, sigma, T, kind)
    st.markdown(f"**Option Value (PV):** `{pv:,.4f}`")
    st.table(pd.DataFrame(g, index=["Value"]).T.style.format("{:,.6f}"))

# -------- Portfolio (lean) --------
with tabs[1]:
    st.subheader("Portfolio")
    left, right = st.columns([1.3, 1])

    # Add Option
    with left:
        pticker = st.selectbox("Option Ticker", tickers, key="opt_ticker")
        pspot = get_spot(pticker) or 100.0
        pr = get_risk_free(); pq = get_div_yield(pticker)
        pexps = get_expiries(pticker)
        if not pexps:
            st.warning("No expiries for this ticker.")
        else:
            pexp = st.selectbox("Expiry", pexps, key="opt_exp")
            chain = get_chain(pticker, pexp)
            if chain.empty:
                st.warning("No option chain for this expiry.")
            else:
                ptype = st.radio("Type", ["Call","Put"], horizontal=True, key="opt_type")
                strikes = np.sort(chain.loc[chain["type"]==ptype, "strike"].dropna().unique())
                def_idx = int(np.searchsorted(strikes, pspot)) if len(strikes) else 0
                pK = st.selectbox("Strike", strikes, index=min(def_idx, max(len(strikes)-1,0)), key="opt_strike")
                sel = chain[(chain["type"]==ptype) & (chain["strike"]==pK)]
                # robust IV read
                if not sel.empty and pd.notna(sel["impliedVolatility"].iloc[0]):
                    raw_iv = float(sel["impliedVolatility"].iloc[0])
                    piv = raw_iv*100.0 if (raw_iv < 0.02 and 0.02 <= raw_iv*100.0 <= 5.0) else raw_iv
                    piv = float(np.clip(piv, 0.01, 5.0))
                else:
                    piv = get_atm_iv(pticker, pspot) or 0.20
                pqty = st.number_input("Quantity (+long / −short)", value=1, step=1, key="opt_qty")
                st.caption(f"Spot={pspot:.2f}  r={pr:.3f}  q={pq:.3f}  IV={piv:.2%}")
                if st.button("➕ Add Option"):
                    Tyrs = years_to_expiry(pexp)
                    st.session_state.portfolio.append(
                        OptLine(pticker, ptype, pexp, float(pK), int(pqty),
                                float(pspot), float(pr), float(pq), float(piv), float(Tyrs))
                    )
                    st.success("Added option.")

    # Add Stock
    with right:
        stick = st.selectbox("Stock Ticker", tickers, key="stk_ticker")
        sspot = get_spot(stick) or 100.0
        sqty = st.number_input("Shares (+long / −short)", value=0, step=1, key="stk_qty")
        st.caption(f"Spot={sspot:.2f}")
        if st.button("➕ Add Stock"):
            st.session_state.portfolio.append(StockLine(stick, int(sqty), float(sspot)))
            st.success("Added stock.")

    st.divider()

    if st.session_state.portfolio:
        # Simple table (no mids, no PVs, just what matters)
        rows = []
        for i, line in enumerate(st.session_state.portfolio, 1):
            if hasattr(line, "iv"):  # option
                rows.append(dict(
                    idx=i, Kind=f"Option {line.type}", Ticker=line.ticker, Expiry=line.expiry,
                    Strike=line.strike, Qty=line.qty, S=line.S, IV=line.iv
                ))
            else:  # stock
                rows.append(dict(
                    idx=i, Kind="Stock", Ticker=line.ticker, Expiry="—",
                    Strike="—", Qty=line.qty, S=line.S, IV=np.nan
                ))
        df = pd.DataFrame(rows)
        # Format numerics only
        for c in ["Strike","Qty","S","IV"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        st.dataframe(
            df.style.format({"Strike":"{:.4f}","S":"{:.4f}","IV":"{:.4f}","Qty":"{:.0f}"}),
            use_container_width=True
        )

        # ---- Portfolio Greeks (totals) ----
        tot = {"Delta":0.0,"Gamma":0.0,"Vega_1pct":0.0,"Theta_day":0.0,"Rho":0.0}
        for l in st.session_state.portfolio:
            if hasattr(l, "iv"):
                g = bs_greeks(l.S, l.strike, l.r, l.q, l.iv, l.T, l.type)
                for k in tot: tot[k] += g[k]*l.qty
            else:
                # Stock: Δ=qty, others=0
                tot["Delta"] += 1.0*l.qty
        st.markdown("### Portfolio Greeks (sum)")
        st.table(pd.DataFrame(tot, index=["Total"]).T.style.format("{:,.6f}"))

        # ---- P&L vs uniform % move for ALL underlyings ----
        st.markdown("### Portfolio P&L vs Uniform % Move")
        col_a, col_b = st.columns(2)
        pct_min = col_a.number_input("Min move (%)", value=-50.0, step=5.0)
        pct_max = col_b.number_input("Max move (%)", value=50.0, step=5.0)
        if pct_max <= pct_min: pct_max = pct_min + 1.0
        grid_pct = np.linspace(pct_min/100.0, pct_max/100.0, 201)

        pnl = np.zeros_like(grid_pct)
        for l in st.session_state.portfolio:
            if hasattr(l, "iv"):  # options repriced with BSM at S*(1+p), hold vol/time/r/q constant
                pv_grid = np.array([
                    bs_price(l.S*(1+p), l.strike, l.r, l.q, l.iv, l.T, l.type) for p in grid_pct
                ])
                pv_base = bs_price(l.S, l.strike, l.r, l.q, l.iv, l.T, l.type)
                pnl += l.qty*(pv_grid - pv_base)
            else:
                pv_grid = (l.S*(1+grid_pct))*l.qty
                pv_base = l.S*l.qty
                pnl += (pv_grid - pv_base)

        chart = pd.DataFrame({"Underlying % Move": grid_pct*100.0, "Portfolio P&L": pnl})
        st.line_chart(
            chart.set_index("Underlying % Move"),
            x_label="Underlying % Move (%)", y_label="Portfolio P&L ($)",
            height=320, use_container_width=True
        )

        # Remove / Clear controls
        col1, col2 = st.columns(2)
        rm_idx = col1.number_input("Remove line #", min_value=1, max_value=len(df), value=1, step=1)
        if col1.button("Remove"): st.session_state.portfolio.pop(int(rm_idx) - 1); st.rerun()
        if col2.button("Clear Portfolio"): st.session_state.portfolio.clear(); st.rerun()

    else:
        st.info("Add options and/or stocks. Quantities: positive=long, negative=short.")
