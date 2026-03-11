import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="CDS Pricer", layout="wide")
st.title("CDS Pricer")

# --- Sidebar Inputs ---
st.sidebar.header("Contract")
notional = st.sidebar.number_input("Notional", value=1000, step=100)
maturity = st.sidebar.number_input("Maturity (years)", value=5.0, step=0.5, min_value=0.5)
freq_label = st.sidebar.selectbox("Payment Frequency", ["Quarterly", "Semi-Annual", "Annual"])
freq_map = {"Quarterly": 4, "Semi-Annual": 2, "Annual": 1}
freq = freq_map[freq_label]

st.sidebar.header("Market")
contractual_spread_bps = st.sidebar.number_input("Contractual Spread (bps)", value=100.0, step=1.0)
recovery = st.sidebar.slider("Recovery Rate", 0.0, 1.0, 0.40, 0.01)
risk_free_rate = st.sidebar.slider("Risk-Free Rate", 0.0, 0.15, 0.03, 0.001)

st.sidebar.header("Credit")
mode = st.sidebar.selectbox("Mode", ["Flat Hazard Rate", "Input Hazard Rate"])
if mode == "Flat Hazard Rate":
    hazard_rate = st.sidebar.slider("Hazard Rate (λ)", 0.001, 0.50, 0.02, 0.001)
else:
    hazard_rate = st.sidebar.number_input("Hazard Rate (λ)", value=0.02, step=0.001, format="%.4f")

# --- Calculations ---
dt = 1.0 / freq
n_periods = int(maturity * freq)
times = np.array([(i + 1) * dt for i in range(n_periods)])

# Survival and default probabilities
survival = np.exp(-hazard_rate * times)
# Marginal default probability in each period
survival_prev = np.exp(-hazard_rate * (times - dt))
marginal_pd = survival_prev - survival

# Discount factors
df = np.exp(-risk_free_rate * times)

# Premium Leg PV: sum of (spread * dt * notional * Q(t_i) * DF(t_i))
# We compute per 1bp first, then scale
premium_leg_pv01 = np.sum(dt * survival * df)  # risky PV01 (per 1 notional)
risky_pv01 = premium_leg_pv01

contractual_spread = contractual_spread_bps / 10000.0
premium_leg_pv = contractual_spread * notional * risky_pv01

# Protection Leg PV: sum of (LGD * marginal_pd_i * DF(t_mid_i))
lgd = 1.0 - recovery
# Discount at midpoint of each period
times_mid = times - dt / 2.0
df_mid = np.exp(-risk_free_rate * times_mid)
protection_leg_pv = lgd * notional * np.sum(marginal_pd * df_mid)

# Fair spread
fair_spread = (lgd * np.sum(marginal_pd * df_mid)) / risky_pv01 if risky_pv01 > 0 else 0.0
fair_spread_bps = fair_spread * 10000.0

# CDS MTM (mark-to-market) from protection buyer perspective
cds_mtm = protection_leg_pv - premium_leg_pv

# Terminal survival and default
survival_T = np.exp(-hazard_rate * maturity)
default_pd_T = 1.0 - survival_T

# --- Display Results ---
st.header("Pricing Results")
col1, col2, col3 = st.columns(3)
col1.metric("Fair Spread", f"{fair_spread_bps:.2f} bps")
col2.metric("CDS PV (MTM)", f"${cds_mtm:.2f}")
col3.metric("Risky PV01", f"${risky_pv01:.2f}")

col4, col5, col6, col7 = st.columns(4)
col4.metric("Premium Leg PV", f"${premium_leg_pv:.2f}")
col5.metric("Protection Leg PV", f"${protection_leg_pv:.2f}")
col6.metric("Survival Q(T)", f"{survival_T * 100:.2f}%")
col7.metric("Default PD(T)", f"{default_pd_T * 100:.2f}%")

# --- Plot ---
st.header("Survival & Default Probabilities")
t_plot = np.linspace(0, maturity, 200)
surv_plot = np.exp(-hazard_rate * t_plot)
pd_plot = 1.0 - surv_plot

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(t_plot, surv_plot, label="Survival Q(t)", linewidth=2)
ax.plot(t_plot, pd_plot, label="Default PD(t)", linewidth=2, linestyle="--")
ax.set_xlabel("Time (years)")
ax.set_ylabel("Probability")
ax.set_title("Survival & Default Probabilities")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
st.pyplot(fig)
