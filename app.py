# app.py — S&P 500 Options Portfolio (with Cash Ledger & robust IV normalization)
# ------------------------------------------------------
# Three tabs:
# 1) Option Calculator — quick BSM pricing & greeks
# 2) Portfolio — add stocks/options; per-underlying greeks; CASH tracked at trade entry
# 3) P&L vs S&P% — scenario equity = mark-to-market of all lines + cash (beta-mapped)

import math
from dataclasses import dataclass
from typing import Literal, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.stats import norm

st.set_page_config(page_title="S&P 500 Options — Calculator & Portfolio", layout="wide")

# ========== Black–Scholes core ==========
OptionType = Literal["Call", "Put"]


def _d1(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    if any(v <= 0 for v in [S, K, sigma, T]):
        return np.nan
    return (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def _d2(d1: float, sigma: float, T: float) -> float:
    return d1 - sigma * math.sqrt(T)


def bs_price(S: float, K: float, r: float, q: float, sigma: float, T: float, kind: OptionType) -> float:
    d1 = _d1(S, K, r, q, sigma, T)
    d2 = _d2(d1, sigma, T)
    if not np.isfinite(d1):
        return float("nan")
    if kind == "Call":
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)


def bs_greeks(S: float, K: float, r: float, q: float, sigma: float, T: float, kind: OptionType) -> dict:
    d1 = _d1(S, K, r, q, sigma, T)
    d2 = _d2(d1, sigma, T)
    if not np.isfinite(d1):
        return {k: np.nan for k in ["Delta", "Gamma", "Vega_1pct", "Theta_day", "Rho"]}
    pdf = norm.pdf(d1)
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)
    sqrtT = math.sqrt(T)
    delta = disc_q * (norm.cdf(d1) if kind == "Call" else norm.cdf(d1) - 1)
    gamma = disc_q * pdf / (S * sigma * sqrtT)
    # vega per 1 vol point (0.01 absolute)
    vega_1pct = S * disc_q * pdf * sqrtT / 100.0
    if kind == "Call":
        theta_pa = (-S * disc_q * pdf * sigma) / (2 * sqrtT) - r * K * disc_r * norm.cdf(d2) + q * S * disc_q * norm.cdf(d1)
        rho = K * T * disc_r * norm.cdf(d2)
    else:
        theta_pa = (-S * disc_q * pdf * sigma) / (2 * sqrtT) + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1)
        rho = -K * T * disc_r * norm.cdf(-d2)
    return {"Delta": delta, "Gamma": gamma, "Vega_1pct": vega_1pct, "Theta_day": theta_pa / 365.0, "Rho": rho}


# ========== Data helpers ==========
TICKERS = [
    "SPY", "AAPL", "MSFT", "NVDA", "AMZN",
    "GOOGL", "META", "BRK-B", "LLY", "TSLA", "JPM",
]


@st.cache_data(ttl=300)
def get_spot(tkr: str) -> Optional[float]:
    h = yf.Ticker(tkr).history(period="1d")
    return float(h["Close"].iloc[-1]) if not h.empty else None


@st.cache_data(ttl=300)
def get_risk_free() -> float:
    try:
        x = yf.Ticker("^IRX").history(period="5d")["Close"].dropna()
        return float(x.iloc[-1]) / 100.0 if not x.empty else 0.02
    except Exception:
        return 0.02


@st.cache_data(ttl=3600)
def get_div_yield(tkr: str) -> float:
    try:
        fi = yf.Ticker(tkr).fast_info
        return float(fi.get("dividend_yield", 0.0) if isinstance(fi, dict) else 0.0)
    except Exception:
        return 0.0


@st.cache_data(ttl=120)
def get_expiries(tkr: str) -> List[str]:
    try:
        return yf.Ticker(tkr).options
    except Exception:
        return []


@st.cache_data(ttl=120)
def get_chain(tkr: str, expiry: str) -> pd.DataFrame:
    oc = yf.Ticker(tkr).option_chain(expiry)

    def _normalize_iv(iv: pd.Series) -> pd.Series:
        """Normalize IV to fraction units (0.20 = 20%). Handles 20 ↔ 0.20 and 0.002 ↔ 0.2% cases."""
        s = pd.to_numeric(iv, errors="coerce")
        v = s.dropna()
        if v.empty:
            return s
        med = float(v.median())
        q90 = float(v.quantile(0.90))
        # If typical IV looks like 20, 30, ... → divide by 100
        if med > 2.0:
            s = s / 100.0
        # If typical IV looks like 0.002, 0.005 and even the 90th pct < 0.05 → multiply by 100
        elif med < 0.02 and q90 < 0.05:
            s = s * 100.0
        return s.clip(lower=0.01, upper=5.0)

    def _prep(df: pd.DataFrame, typ: str) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        out["type"] = typ
        out["mid"] = (out["bid"].fillna(0.0) + out["ask"].fillna(0.0)) / 2.0
        if "impliedVolatility" in out.columns:
            out["impliedVolatility"] = _normalize_iv(out["impliedVolatility"])
        return out

    return pd.concat([_prep(oc.calls, "Call"), _prep(oc.puts, "Put")], ignore_index=True)


@st.cache_data(ttl=120)
def get_atm_iv(tkr: str, spot: float) -> Optional[float]:
    exps = get_expiries(tkr)
    if not exps:
        return None
    for e in exps[:3]:
        ch = get_chain(tkr, e)
        if ch.empty:
            continue
        ch["dist"] = (ch["strike"] - spot).abs()
        ivs = ch.sort_values("dist")["impliedVolatility"].dropna()
        if not ivs.empty:
            return float(np.clip(ivs.iloc[0], 0.01, 5.0))
    return None


def years_to_expiry(exp_str: str) -> float:
    exp = pd.to_datetime(exp_str)
    now = pd.Timestamp.utcnow().tz_localize(None)
    dt = exp - now
    return max((dt.days + dt.seconds / 86400.0) / 365.0, 1e-6)


@st.cache_data(ttl=3600)
def get_beta(tkr: str) -> float:
    if tkr == "SPY":
        return 1.0
    try:
        s = yf.Ticker(tkr).history(period="2y")["Close"].pct_change().dropna()
        m = yf.Ticker("SPY").history(period="2y")["Close"].pct_change().dropna()
        df = pd.concat([s.rename("S"), m.rename("M")], axis=1).dropna()
        if len(df) < 60:
            return 1.0
        cov = np.cov(df["S"], df["M"], ddof=1)[0, 1]
        var_m = np.var(df["M"], ddof=1)
        return float(cov / var_m) if var_m > 0 else 1.0
    except Exception:
        return 1.0


# ========== Portfolio models & cash ==========
@dataclass
class OptLine:
    ticker: str
    type: OptionType
    expiry: str
    strike: float
    qty: int
    S: float
    r: float
    q: float
    iv: float
    T: float
    premium: float
    contract_mult: int = 100


@dataclass
class StockLine:
    ticker: str
    qty: int
    S: float


if "portfolio" not in st.session_state:
    st.session_state.portfolio: List[object] = []
    st.session_state.cash: float = 0.0  # cumulative trade cashflows (negative = spent)


def record_cashflow_for_line(line) -> float:
    """Record trade cashflow into session cash and return it.
    Options: pay premium when buying (qty>0), receive when shorting (qty<0).
    Stocks: pay price when buying, receive when shorting.
    """
    if hasattr(line, "iv"):
        cf = - line.qty * line.contract_mult * line.premium
    else:
        cf = - line.qty * line.S
    st.session_state.cash += float(cf)
    return float(cf)


# ========== UI ==========
st.title("📈 S&P 500 Options Portfolio Tool")
tabs = st.tabs(["Option Calculator", "Portfolio", "P&L vs S&P%"])


# --- 1) Option Calculator ---
with tabs[0]:
    st.write("Quick Black–Scholes pricing & Greeks for a selected ticker. (IV handling is normalized internally; enter σ as a fraction, e.g., 0.20 = 20%.)")

    c0, c1, c2, c3, c4, c5 = st.columns([1.4, 1, 1, 1, 1, 1])
    tkr = c0.selectbox("Ticker", TICKERS)
    S0 = get_spot(tkr) or 100.0
    r0 = get_risk_free()
    q0 = get_div_yield(tkr)
    iv0 = float(get_atm_iv(tkr, S0) or 0.20)

    kind = c1.selectbox("Type", ["Call", "Put"])
    S = c2.number_input("Spot", value=float(S0))
    K = c3.number_input("Strike", value=float(round(S0)))
    T = c4.number_input("T (yrs)", value=0.25, step=0.05, min_value=1e-6)
    sigma = c5.number_input("Vol (σ)", value=float(iv0), step=0.01, min_value=0.01)

    colr, colq = st.columns(2)
    r = colr.number_input("r", value=float(r0), step=0.002)
    q = colq.number_input("q", value=float(q0), step=0.002)

    pv = bs_price(S, K, r, q, sigma, T, kind)
    g = bs_greeks(S, K, r, q, sigma, T, kind)

    st.markdown(f"**Price:** `{pv:,.4f}`")
    st.table(pd.DataFrame(g, index=["Value"]).T.style.format("{:,.6f}"))


# --- 2) Portfolio (stocks + options + CASH) ---
with tabs[1]:
    st.write("Build your portfolio. Trade cashflows go to a **cash** ledger so total P&L = cash + scenario marks − (cash + current marks). IV scaling fixed.")

    left, right = st.columns([1.3, 1])

    # Add Option
    with left:
        ptkr = st.selectbox("Option ticker", TICKERS, key="opt_t")
        pS = get_spot(ptkr) or 100.0
        pr = get_risk_free()
        pq = get_div_yield(ptkr)
        pexps = get_expiries(ptkr)

        if pexps:
            pexp = st.selectbox("Expiry", pexps)
            chain = get_chain(ptkr, pexp)
            if not chain.empty:
                ptype = st.radio("Type", ["Call", "Put"], horizontal=True)
                strikes = np.sort(chain.loc[chain["type"] == ptype, "strike"].dropna().unique())
                def_idx = int(np.searchsorted(strikes, pS)) if len(strikes) else 0
                idx = min(def_idx, max(len(strikes) - 1, 0))
                pK = st.selectbox("Strike", strikes, index=idx)

                piv = float(get_atm_iv(ptkr, pS) or 0.20)
                Tyrs = years_to_expiry(pexp)
                model_px = float(bs_price(pS, float(pK), pr, pq, piv, Tyrs, ptype))

                pqty = st.number_input("Qty (+long/−short)", value=1, step=1)
                st.caption(f"Spot={pS:.2f}  r={pr:.3f}  q={pq:.3f}  IV={piv:.2%}  Model={model_px:.4f}")

                if st.button("➕ Add Option"):
                    line = OptLine(
                        ticker=ptkr, type=ptype, expiry=pexp, strike=float(pK), qty=int(pqty),
                        S=float(pS), r=float(pr), q=float(pq), iv=float(piv), T=float(Tyrs), premium=float(model_px)
                    )
                    st.session_state.portfolio.append(line)
                    cf = record_cashflow_for_line(line)
                    st.success(f"Added option (BSM premium). Cash change: {cf:,.2f}")
        else:
            st.info("No expiries found.")

    # Add Stock
    with right:
        stck = st.selectbox("Stock ticker", TICKERS, key="stk_t")
        sS = get_spot(stck) or 100.0
        sqty = st.number_input("Shares (+long/−short)", value=0, step=1)
        st.caption(f"Spot={sS:.2f}")
        if st.button("➕ Add Stock"):
            line = StockLine(ticker=stck, qty=int(sqty), S=float(sS))
            st.session_state.portfolio.append(line)
            cf = record_cashflow_for_line(line)
            st.success(f"Added stock. Cash change: {cf:,.2f}")

    st.divider()

    st.info(f"**Cash ledger (since start):** {st.session_state.cash:,.2f}")

    if not st.session_state.portfolio:
        st.info("Add instruments above.")
    else:
        # Lines table
        rows = []
        for i, l in enumerate(st.session_state.portfolio, 1):
            if hasattr(l, "iv"):
                rows.append(
                    dict(idx=i, Type=f"Opt {l.type}", Ticker=l.ticker, Expiry=l.expiry,
                         Strike=l.strike, Qty=l.qty, S=l.S, IV=l.iv, Premium=l.premium)
                )
            else:
                rows.append(
                    dict(idx=i, Type="Stock", Ticker=l.ticker, Expiry="—",
                         Strike="—", Qty=l.qty, S=l.S, IV=np.nan, Premium=np.nan)
                )
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        # Greeks per underlying
        per_tkr: dict = {}
        for l in st.session_state.portfolio:
            t = l.ticker
            if t not in per_tkr:
                per_tkr[t] = {k: 0.0 for k in ["Delta", "Gamma", "Vega_1pct", "Theta_day", "Rho"]}
            if hasattr(l, "iv"):
                g = bs_greeks(l.S, l.strike, l.r, l.q, l.iv, l.T, l.type)
                per_tkr[t]["Delta"] += g["Delta"] * l.qty * l.contract_mult
                per_tkr[t]["Gamma"] += g["Gamma"] * l.qty * l.contract_mult
                per_tkr[t]["Vega_1pct"] += g["Vega_1pct"] * l.qty * l.contract_mult
                per_tkr[t]["Theta_day"] += g["Theta_day"] * l.qty * l.contract_mult
                per_tkr[t]["Rho"] += g["Rho"] * l.qty * l.contract_mult
            else:
                per_tkr[t]["Delta"] += 1.0 * l.qty

        st.markdown("### Greeks per Underlying")
        dfp = pd.DataFrame([{**{"Ticker": t}, **vals} for t, vals in per_tkr.items()]).set_index("Ticker")
        st.table(dfp.style.format("{:,.6f}"))

        c1, c2 = st.columns(2)
        rm = c1.number_input("Remove #", min_value=1, max_value=len(df), value=1, step=1)
        if c1.button("Remove"):
            line = st.session_state.portfolio.pop(int(rm) - 1)
            # reverse its cashflow
            if hasattr(line, "iv"):
                st.session_state.cash -= (- line.qty * line.contract_mult * line.premium)
            else:
                st.session_state.cash -= (- line.qty * line.S)
            st.rerun()
        if c2.button("Clear"):
            st.session_state.portfolio.clear()
            st.session_state.cash = 0.0
            st.rerun()


# --- 3) P&L vs S&P% (with CASH) ---
with tabs[2]:
    st.write("Simulate total equity vs S&P move. Equity = cash + marked stocks/options (beta-mapped). Premiums and short-sale proceeds are included via the cash ledger. 0% move ≈ 0.")

    if not st.session_state.portfolio:
        st.info("Add some positions first.")
    else:
        a, b = st.columns(2)
        mn = a.number_input("Min move %", value=-50.0, step=5.0)
        mx = b.number_input("Max move %", value=50.0, step=5.0)
        if mx <= mn:
            mx = mn + 1.0
        grid = np.linspace(mn / 100.0, mx / 100.0, 201)

        betas = {l.ticker: get_beta(l.ticker) for l in st.session_state.portfolio}

        # Baseline equity now = cash + current marks
        equity_now = st.session_state.cash
        for l in st.session_state.portfolio:
            if hasattr(l, "iv"):
                equity_now += l.qty * l.contract_mult * bs_price(l.S, l.strike, l.r, l.q, l.iv, l.T, l.type)
            else:
                equity_now += l.qty * l.S

        # Scenario equity for each grid point
        equity_scn = np.full_like(grid, fill_value=st.session_state.cash, dtype=float)
        for idx, p in enumerate(grid):
            total = st.session_state.cash
            for l in st.session_state.portfolio:
                beta = betas.get(l.ticker, 1.0)
                if hasattr(l, "iv"):
                    new_p = bs_price(l.S * (1.0 + beta * p), l.strike, l.r, l.q, l.iv, l.T, l.type)
                    total += l.qty * l.contract_mult * new_p
                else:
                    new_s = l.S * (1.0 + beta * p)
                    total += l.qty * new_s
            equity_scn[idx] = total

        pnl = equity_scn - equity_now
        chart = pd.DataFrame({"S&P %": grid * 100.0, "P&L": pnl})
        st.line_chart(
            chart.set_index("S&P %"),
            x_label="S&P move (%)",
            y_label="Portfolio P&L ($)",
            height=340,
            use_container_width=True,
        )
