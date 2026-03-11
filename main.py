import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

st.set_page_config(page_title="CDS Pricer", layout="wide")

# ── Dark style for all charts ──
plt.rcParams.update({
    "figure.facecolor": "#0E1117",
    "axes.facecolor": "#0E1117",
    "axes.edgecolor": "#333",
    "axes.labelcolor": "#ccc",
    "text.color": "#ccc",
    "xtick.color": "#999",
    "ytick.color": "#999",
    "grid.color": "#222",
    "legend.facecolor": "#0E1117",
    "legend.edgecolor": "#333",
    "legend.labelcolor": "#ccc",
    "font.size": 10,
})

# ════════════════════════════════════════════
# SIDEBAR INPUTS
# ════════════════════════════════════════════
st.sidebar.markdown("## Contract")
notional = st.sidebar.number_input("Notional ($)", value=10_000_000, step=1_000_000, format="%d")
maturity = st.sidebar.number_input("Maturity (years)", value=5.0, step=0.5, min_value=0.5, max_value=30.0)
freq_label = st.sidebar.selectbox("Payment Frequency", ["Quarterly", "Semi-Annual", "Annual"])
freq_map = {"Quarterly": 4, "Semi-Annual": 2, "Annual": 1}
freq = freq_map[freq_label]

st.sidebar.markdown("## Market")
contractual_spread_bps = st.sidebar.number_input("Contractual Spread (bps)", value=100.0, step=5.0)
recovery = st.sidebar.slider("Recovery Rate", 0.0, 0.80, 0.40, 0.01)
risk_free_rate = st.sidebar.slider("Risk-Free Rate", 0.0, 0.15, 0.05, 0.005, format="%.3f")

st.sidebar.markdown("## Credit")
hazard_rate = st.sidebar.slider("Hazard Rate (λ)", 0.001, 0.30, 0.02, 0.001, format="%.3f")

# ════════════════════════════════════════════
# CORE ENGINE
# ════════════════════════════════════════════
dt = 1.0 / freq
n_periods = int(maturity * freq)
times = np.array([(i + 1) * dt for i in range(n_periods)])

survival = np.exp(-hazard_rate * times)
survival_prev = np.exp(-hazard_rate * (times - dt))
marginal_pd = survival_prev - survival
df = np.exp(-risk_free_rate * times)
times_mid = times - dt / 2.0
df_mid = np.exp(-risk_free_rate * times_mid)
lgd = 1.0 - recovery

# Risky PV01 & legs
risky_pv01 = np.sum(dt * survival * df)
contractual_spread = contractual_spread_bps / 10000.0
premium_leg_pv = contractual_spread * notional * risky_pv01
protection_leg_pv = lgd * notional * np.sum(marginal_pd * df_mid)

# Fair spread
fair_spread = (lgd * np.sum(marginal_pd * df_mid)) / risky_pv01 if risky_pv01 > 0 else 0.0
fair_spread_bps = fair_spread * 10000.0

# MTM & upfront
cds_mtm = protection_leg_pv - premium_leg_pv
upfront_pct = (fair_spread_bps - contractual_spread_bps) / 10000.0 * risky_pv01 / (maturity) * 100
upfront_fee = cds_mtm
upfront_pct_notional = (upfront_fee / notional) * 100.0

# Carry (annual premium)
annual_carry = contractual_spread * notional

# Breakeven spread
breakeven_spread_bps = fair_spread_bps

# Terminal
survival_T = np.exp(-hazard_rate * maturity)
default_pd_T = 1.0 - survival_T

# ── Sensitivity helpers ──
def compute_fair_spread(h, r_rate, rec, mat, f):
    d = 1.0 / f
    n = int(mat * f)
    t = np.array([(i + 1) * d for i in range(n)])
    s = np.exp(-h * t)
    sp = np.exp(-h * (t - d))
    mp = sp - s
    d_f = np.exp(-r_rate * t)
    d_fm = np.exp(-r_rate * (t - d / 2.0))
    pv01 = np.sum(d * s * d_f)
    if pv01 == 0:
        return 0.0
    return (1.0 - rec) * np.sum(mp * d_fm) / pv01 * 10000.0

def compute_mtm(h, r_rate, rec, mat, f, c_spread_bps, N):
    d = 1.0 / f
    n = int(mat * f)
    t = np.array([(i + 1) * d for i in range(n)])
    s = np.exp(-h * t)
    sp = np.exp(-h * (t - d))
    mp = sp - s
    d_f = np.exp(-r_rate * t)
    d_fm = np.exp(-r_rate * (t - d / 2.0))
    pv01 = np.sum(d * s * d_f)
    prem = (c_spread_bps / 10000.0) * N * pv01
    prot = (1.0 - rec) * N * np.sum(mp * d_fm)
    return prot - prem

# ════════════════════════════════════════════
# DISPLAY
# ════════════════════════════════════════════
st.markdown("# CDS Pricer")

# ── Row 1: Key metrics ──
c1, c2, c3, c4 = st.columns(4)
c1.metric("Fair Spread", f"{fair_spread_bps:.2f} bps")
c2.metric("CDS MTM", f"${cds_mtm:,.0f}")
c3.metric("Upfront Fee", f"{upfront_pct_notional:.2f}% of notional")
c4.metric("Risky PV01", f"${notional * risky_pv01 / 10000:,.0f} /bp")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Premium Leg PV", f"${premium_leg_pv:,.0f}")
c6.metric("Protection Leg PV", f"${protection_leg_pv:,.0f}")
c7.metric("Survival Q(T)", f"{survival_T * 100:.2f}%")
c8.metric("Annual Carry", f"${annual_carry:,.0f}")

st.markdown("---")

# ════════════════════════════════════════════
# CHARTS
# ════════════════════════════════════════════

# ── Row 2: Survival / Default + Term Structure of Fair Spread ──
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Survival & Default Curves")
    t_plot = np.linspace(0, maturity, 300)
    fig1, ax1 = plt.subplots(figsize=(6, 3.5))
    ax1.plot(t_plot, np.exp(-hazard_rate * t_plot), color="#4FC3F7", linewidth=2, label="Survival Q(t)")
    ax1.fill_between(t_plot, np.exp(-hazard_rate * t_plot), alpha=0.08, color="#4FC3F7")
    ax1.plot(t_plot, 1 - np.exp(-hazard_rate * t_plot), color="#EF5350", linewidth=2, linestyle="--", label="Cumulative PD(t)")
    ax1.fill_between(t_plot, 1 - np.exp(-hazard_rate * t_plot), alpha=0.08, color="#EF5350")
    ax1.set_xlabel("Time (years)")
    ax1.set_ylabel("Probability")
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax1.legend(framealpha=0.5)
    ax1.grid(True, alpha=0.15)
    fig1.tight_layout()
    st.pyplot(fig1)

with col_right:
    st.subheader("Fair Spread Term Structure")
    maturities_ts = np.arange(1, min(maturity * 2, 31) + 1, 1)
    spreads_ts = [compute_fair_spread(hazard_rate, risk_free_rate, recovery, m, freq) for m in maturities_ts]
    fig2, ax2 = plt.subplots(figsize=(6, 3.5))
    ax2.plot(maturities_ts, spreads_ts, color="#FFA726", linewidth=2, marker="o", markersize=4)
    ax2.axhline(y=contractual_spread_bps, color="#666", linestyle=":", linewidth=1, label=f"Contractual {contractual_spread_bps:.0f} bps")
    ax2.set_xlabel("Maturity (years)")
    ax2.set_ylabel("Fair Spread (bps)")
    ax2.legend(framealpha=0.5)
    ax2.grid(True, alpha=0.15)
    fig2.tight_layout()
    st.pyplot(fig2)

# ── Row 3: Sensitivity to Hazard Rate + Sensitivity to Recovery ──
col_left2, col_right2 = st.columns(2)

with col_left2:
    st.subheader("MTM vs Hazard Rate")
    h_range = np.linspace(0.001, 0.25, 80)
    mtm_h = [compute_mtm(h, risk_free_rate, recovery, maturity, freq, contractual_spread_bps, notional) for h in h_range]
    fig3, ax3 = plt.subplots(figsize=(6, 3.5))
    ax3.plot(h_range * 100, np.array(mtm_h) / 1e6, color="#AB47BC", linewidth=2)
    ax3.axhline(0, color="#555", linewidth=0.8, linestyle="--")
    ax3.axvline(hazard_rate * 100, color="#EF5350", linewidth=1, linestyle=":", label=f"Current λ={hazard_rate:.3f}")
    ax3.set_xlabel("Hazard Rate (%)")
    ax3.set_ylabel("MTM ($M)")
    ax3.legend(framealpha=0.5)
    ax3.grid(True, alpha=0.15)
    fig3.tight_layout()
    st.pyplot(fig3)

with col_right2:
    st.subheader("MTM vs Recovery Rate")
    rec_range = np.linspace(0.0, 0.80, 80)
    mtm_r = [compute_mtm(hazard_rate, risk_free_rate, r, maturity, freq, contractual_spread_bps, notional) for r in rec_range]
    fig4, ax4 = plt.subplots(figsize=(6, 3.5))
    ax4.plot(rec_range * 100, np.array(mtm_r) / 1e6, color="#26A69A", linewidth=2)
    ax4.axhline(0, color="#555", linewidth=0.8, linestyle="--")
    ax4.axvline(recovery * 100, color="#EF5350", linewidth=1, linestyle=":", label=f"Current R={recovery:.0%}")
    ax4.set_xlabel("Recovery Rate (%)")
    ax4.set_ylabel("MTM ($M)")
    ax4.legend(framealpha=0.5)
    ax4.grid(True, alpha=0.15)
    fig4.tight_layout()
    st.pyplot(fig4)

# ── Row 4: Cashflow waterfall + CS01 ladder ──
col_left3, col_right3 = st.columns(2)

with col_left3:
    st.subheader("Expected Cashflows per Period")
    prem_cf = contractual_spread * notional * dt * survival * df
    prot_cf = lgd * notional * marginal_pd * df_mid
    fig5, ax5 = plt.subplots(figsize=(6, 3.5))
    width = dt * 0.35
    ax5.bar(times - width / 2, -prem_cf / 1e3, width=width, color="#EF5350", alpha=0.85, label="Premium (paid)")
    ax5.bar(times + width / 2, prot_cf / 1e3, width=width, color="#4FC3F7", alpha=0.85, label="Protection (received)")
    ax5.axhline(0, color="#555", linewidth=0.8)
    ax5.set_xlabel("Time (years)")
    ax5.set_ylabel("Expected CF ($K)")
    ax5.legend(framealpha=0.5, fontsize=8)
    ax5.grid(True, alpha=0.15)
    fig5.tight_layout()
    st.pyplot(fig5)

with col_right3:
    st.subheader("CS01 by Tenor Bucket")
    bump = 1  # 1bp
    bucket_years = np.arange(1, int(maturity) + 1)
    cs01_buckets = []
    for yr in bucket_years:
        mtm_base = compute_mtm(hazard_rate, risk_free_rate, recovery, yr, freq, contractual_spread_bps, notional)
        mtm_up = compute_mtm(hazard_rate, risk_free_rate, recovery, yr, freq, contractual_spread_bps + bump, notional)
        cs01_buckets.append(mtm_up - mtm_base)
    marginal_cs01 = np.diff(np.array(cs01_buckets), prepend=0)
    fig6, ax6 = plt.subplots(figsize=(6, 3.5))
    colors = ["#4FC3F7" if v >= 0 else "#EF5350" for v in marginal_cs01]
    ax6.bar(bucket_years, marginal_cs01, color=colors, alpha=0.85, width=0.6)
    ax6.set_xlabel("Tenor Bucket (year)")
    ax6.set_ylabel("Marginal CS01 ($)")
    ax6.grid(True, alpha=0.15)
    fig6.tight_layout()
    st.pyplot(fig6)

# ── Row 5: Spread sensitivity heatmap ──
st.subheader("MTM Heatmap: Hazard Rate × Recovery")
h_grid = np.linspace(0.005, 0.15, 25)
r_grid = np.linspace(0.10, 0.70, 25)
mtm_matrix = np.zeros((len(r_grid), len(h_grid)))
for i, r in enumerate(r_grid):
    for j, h in enumerate(h_grid):
        mtm_matrix[i, j] = compute_mtm(h, risk_free_rate, r, maturity, freq, contractual_spread_bps, notional)

fig7, ax7 = plt.subplots(figsize=(10, 4.5))
im = ax7.imshow(
    mtm_matrix / 1e6,
    aspect="auto",
    origin="lower",
    extent=[h_grid[0] * 100, h_grid[-1] * 100, r_grid[0] * 100, r_grid[-1] * 100],
    cmap="RdYlGn",
)
cbar = fig7.colorbar(im, ax=ax7, label="MTM ($M)")
ax7.set_xlabel("Hazard Rate (%)")
ax7.set_ylabel("Recovery Rate (%)")
ax7.plot(hazard_rate * 100, recovery * 100, "x", color="white", markersize=12, markeredgewidth=2)
fig7.tight_layout()
st.pyplot(fig7)
