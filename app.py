# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Penyusutan Nilai Aset - Euler Decay",
    page_icon="🚗",
    layout="wide"
)

# =========================
# Custom CSS (warm & bright)
# =========================
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(1200px circle at 10% 10%, #1f2a44 0%, transparent 50%),
                    radial-gradient(1000px circle at 90% 0%, #3b1d5e 0%, transparent 45%),
                    linear-gradient(180deg, #0b1220 0%, #0f172a 100%);
        color: #EAEAEA;
    }
    .block-container { 
        padding-top: 1.4rem; 
        padding-bottom: 2rem; 
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f4b 0%, #2a145e 100%);
        border-right: 1px solid rgba(255,255,255,0.10);
    }
    section[data-testid="stSidebar"] * { color: #f5f5f5; }

    .sidebar-card {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 14px 14px 8px 14px;
        margin-bottom: 12px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
    }

    div[data-baseweb="slider"] > div {
        color: #ffb454 !important;
    }

    .metric-wrap {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 14px;
        margin-top: 10px;
        margin-bottom: 6px;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(255,137,90,0.18), rgba(255,196,110,0.12), rgba(255,90,160,0.10));
        border: 1px solid rgba(255,255,255,0.14);
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: 0 10px 28px rgba(0,0,0,0.45);
        backdrop-filter: blur(6px);
    }
    .metric-title {
        font-size: 0.9rem;
        color: #ffd7bf;
        margin-bottom: 6px;
        letter-spacing: 0.3px;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #ffffff;
    }
    .metric-sub {
        font-size: 0.82rem;
        color: #ffe7d6;
        margin-top: 4px;
        opacity: 0.9;
    }

    h1, h2, h3 { 
        letter-spacing: 0.4px; 
        font-weight: 800;
    }

    hr { 
        border: none; 
        height: 1px; 
        background: rgba(255,255,255,0.15); 
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Header
# =========================
st.markdown("## 🚗 Penyusutan Nilai Aset (Used Car Prices)")
st.caption("Dashboard simulasi Euler untuk model penyusutan nilai aset mobil bekas.")
st.divider()

# =========================
# Load & prepare data
# =========================
@st.cache_data
def load_and_prepare(csv_path="public_cars.csv", current_year=2019):
    df = pd.read_csv(csv_path)

    df["age"] = current_year - df["year_produced"]
    df = df[df["age"] >= 0].copy()

    age_price = (
        df.groupby("age", as_index=False)["price_usd"]
          .mean()
          .sort_values("age")
    )
    age_price = age_price[age_price["price_usd"] > 0].copy()

    return df, age_price

def estimate_r_from_logfit(ap_fit):
    """Estimasi r dari decay murni (sesuai notebook)."""
    x = ap_fit["age"].values
    y = np.log(ap_fit["price_usd"].values)
    b, a = np.polyfit(x, y, 1)
    r_est = -b
    V0_est = np.exp(a)
    return float(r_est), float(V0_est)

# =========================
# Euler methods
# =========================
def euler_decay_pure(t0, V0, r, h, t_end):
    """Decay murni: dV/dt = -rV (sesuai modul 6–7)."""
    ts = np.arange(t0, t_end + 1e-9, h)
    Vs = np.zeros_like(ts, dtype=float)
    Vs[0] = V0

    for i in range(1, len(ts)):
        V_prev = Vs[i-1]
        dVdt = -r * V_prev
        Vs[i] = max(V_prev + h * dVdt, 0.0)

    return ts, Vs

def euler_decay_floor(t0, V0, r, K, h, t_end):
    """Decay ke nilai sisa: dV/dt = -r(V-K). Dipakai jika K>0."""
    ts = np.arange(t0, t_end + 1e-9, h)
    Vs = np.zeros_like(ts, dtype=float)
    Vs[0] = V0

    for i in range(1, len(ts)):
        V_prev = Vs[i-1]
        dVdt = -r * (V_prev - K)
        Vs[i] = max(V_prev + h * dVdt, 0.0)

    return ts, Vs

# =========================
# Main
# =========================
try:
    df, age_price = load_and_prepare()
except FileNotFoundError:
    st.error("public_cars.csv tidak ditemukan. Pastikan satu folder dengan app.py.")
    st.stop()

# =========================
# Sidebar
# =========================
st.sidebar.markdown("### ⚙️ Input Parameter Model")
st.sidebar.markdown('<div class="sidebar-card">', unsafe_allow_html=True)

max_age_fit = st.sidebar.slider(
    "Batas umur data untuk fitting/simulasi (tahun)",
    min_value=5,
    max_value=int(age_price["age"].max()),
    value=35,   # default langsung sesuai final
    step=1
)

ap_fit = age_price[age_price["age"] <= max_age_fit].copy()
if len(ap_fit) < 2:
    st.sidebar.warning("Data terlalu sedikit untuk fitting. Naikkan batas umur.")
    st.stop()

r_est, V0_est = estimate_r_from_logfit(ap_fit)

# ====== PERBAIKAN UTAMA DI SINI ======
# Pakai number_input agar r bisa diisi tepat (0.111, dst)
# Rentang r dibuat realistis: max = max(0.5, 3*r_est)
r_max = float(max(0.5, r_est * 3))

r = st.sidebar.number_input(
    "Laju penyusutan r",
    min_value=0.0,
    max_value=r_max,
    value=float(r_est),
    step=0.001,
    format="%.3f",
    help="Isi nilai r sekitar 0.10–0.12 untuk dataset mobil bekas."
)
# ====================================

K = st.sidebar.number_input(
    "Nilai sisa K (USD) (opsional)",
    min_value=0.0,
    max_value=float(ap_fit["price_usd"].max()),
    value=0.0,
    step=100.0
)

h = st.sidebar.slider(
    "Step size h (tahun)",
    min_value=0.1,
    max_value=5.0,
    value=0.25,   # default final yang bagus
    step=0.05
)

st.sidebar.markdown("</div>", unsafe_allow_html=True)

st.sidebar.info(
    "Untuk hasil sesuai modul NIM 6–7 gunakan K = 0 (decay murni). "
    "Jika K > 0, simulasi memakai nilai sisa."
)

# =========================
# Data arrays
# =========================
ages = ap_fit["age"].values
prices = ap_fit["price_usd"].values

t0 = float(ages.min())
V0_data = float(prices[0])
t_end = float(ages.max())

# =========================
# Simulation
# =========================
if K == 0:
    ts, Vs = euler_decay_pure(t0, V0_data, r, h, t_end)
else:
    ts, Vs = euler_decay_floor(t0, V0_data, r, K, h, t_end)

pred_at_data = np.interp(ages, ts, Vs)
mse = float(np.mean((pred_at_data - prices) ** 2))

# =========================
# Metric cards
# =========================
st.markdown(
    f"""
    <div class="metric-wrap">
        <div class="metric-card">
            <div class="metric-title">r (laju penyusutan)</div>
            <div class="metric-value">{r:.4f}</div>
            <div class="metric-sub">besar r → turun lebih cepat</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">K (nilai sisa)</div>
            <div class="metric-value">{K:,.2f} USD</div>
            <div class="metric-sub">K=0 → decay murni</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">MSE</div>
            <div class="metric-value">{mse:,.2f}</div>
            <div class="metric-sub">error data vs simulasi</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# =========================
# Plotly chart
# =========================
fig = go.Figure()

COLOR_DATA = "#FF7A59"
COLOR_SIM  = "#FFC857"

fig.add_trace(go.Scatter(
    x=ages, y=prices,
    mode="lines+markers",
    name="Data Asli (rata-rata)",
    line=dict(width=4, color=COLOR_DATA),
    marker=dict(size=8, color=COLOR_DATA),
    hovertemplate="Umur: %{x} tahun<br>Harga: %{y:,.2f} USD<extra></extra>"
))

fig.add_trace(go.Scatter(
    x=ts, y=Vs,
    mode="lines",
    name=f"Simulasi Euler (h={h})",
    line=dict(width=4, color=COLOR_SIM),
    hovertemplate="Umur: %{x:.2f} tahun<br>Harga: %{y:,.2f} USD<extra></extra>"
))

fig.update_layout(
    template="plotly_dark",
    title="Data Asli vs Simulasi Euler",
    title_font=dict(size=22),
    xaxis_title="Umur mobil (tahun)",
    yaxis_title="Harga rata-rata (USD)",
    hovermode="closest",
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="right", x=1
    ),
    margin=dict(l=30, r=30, t=65, b=45),
    height=540,
)

fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.10)")
fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.10)")

st.plotly_chart(fig, use_container_width=True)

with st.expander("📋 Lihat tabel data rata-rata harga per umur"):
    st.dataframe(ap_fit.reset_index(drop=True), use_container_width=True)