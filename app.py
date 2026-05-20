# Segmentasi Pelanggan Grosir — LRFM-CLV + Fuzzy C-Means
# Antarmuka Streamlit 

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import io

import skfuzzy as fuzz
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# Konfigurasi halaman 
st.set_page_config(
    page_title="Segmentasi Pelanggan LRFM-CLV",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS kustom 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Header utama */
.main-header {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    padding: 2.5rem 2rem 2rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    color: white;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
}
.main-header h1 { font-size: 1.8rem; font-weight: 800; margin: 0 0 .4rem 0; letter-spacing: -0.5px; }
.main-header p  { font-size: .9rem; opacity: .75; margin: 0; }

/* Kartu metrik */
.metric-card {
    background: white;
    border: 1px solid #e8ecf0;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    transition: transform .15s ease, box-shadow .15s ease;
}
.metric-card:hover { transform: translateY(-2px); box-shadow: 0 6px 16px rgba(0,0,0,0.08); }
.metric-card .label { font-size: .75rem; font-weight: 600; color: #6b7280; text-transform: uppercase; letter-spacing: .8px; margin-bottom: .3rem; }
.metric-card .value { font-size: 1.9rem; font-weight: 800; color: #111827; font-family: 'JetBrains Mono', monospace; }
.metric-card .sub   { font-size: .75rem; color: #9ca3af; margin-top: .15rem; }
@media (prefers-color-scheme: dark) {
    .metric-card {
        background: #1e293b !important;
        border-color: #334155 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .metric-card .label { color: #94a3b8 !important; }
    .metric-card .value { color: #f1f5f9 !important; }
    .metric-card .sub   { color: #64748b !important; }
    /* teks inline di kartu landing page & kartu segmen */
    .metric-card div[style*="color:#6b7280"]  { color: #94a3b8 !important; }
    .metric-card div[style*="color:#374151"]  { color: #cbd5e1 !important; }
    .metric-card div[style*="font-weight:700"],
    .metric-card div[style*="font-weight:800"] { color: #f1f5f9 !important; }
    .metric-card div[style*="font-size:2rem"]  { color: #f1f5f9 !important; }
    .metric-card div[style*="font-size:1.6rem"] { color: #f1f5f9 !important; }
    .metric-card div[style*="font-size:1rem"]  { color: #e2e8f0 !important; }
    .metric-card div[style*="font-size:.8rem"] { color: #94a3b8 !important; }
    .metric-card div[style*="font-size:.78rem"] { color: #cbd5e1 !important; }
    .metric-card div[style*="font-size:.75rem"] { color: #64748b !important; }
    .metric-card table td { color: #cbd5e1 !important; }
    .metric-card table td strong { color: #f1f5f9 !important; }
}

/* Badge segmen */
.badge-high   { background:#d1fae5; color:#065f46; border-radius:20px; padding:3px 12px; font-size:.8rem; font-weight:600; }
.badge-medium { background:#fef3c7; color:#92400e; border-radius:20px; padding:3px 12px; font-size:.8rem; font-weight:600; }
.badge-low    { background:#fee2e2; color:#991b1b; border-radius:20px; padding:3px 12px; font-size:.8rem; font-weight:600; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f2027;
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stButton button {
    background: linear-gradient(135deg, #2dd4bf, #0891b2) !important;
    color: white !important;
    border: none !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    width: 100%;
    padding: .6rem 1rem !important;
}
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.1) !important; }

/* Tab style */
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: #f1f5f9; border-radius: 10px; padding: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 7px; font-weight: 600; font-size: .85rem; padding: .4rem 1rem; }
.stTabs [aria-selected="true"] { background: white !important; color: #0f2027 !important; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }
@media (prefers-color-scheme: dark) {
    .stTabs [data-baseweb="tab-list"] {
        background: #1e293b !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #cbd5e1 !important;
    }
    .stTabs [aria-selected="true"] {
        background: #334155 !important;
        color: #ffffff !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.4);
    }
}
/* Tabel */
.dataframe thead th { background: #0f2027 !important; color: white !important; font-weight: 600; }
.dataframe tbody tr:nth-child(even) { background: #f8fafc; }

/* Section title */
.section-title {
    font-size: 1rem; font-weight: 700; color: #0f2027;
    border-left: 4px solid #0891b2; padding-left: .7rem;
    margin: 1.5rem 0 1rem 0;
}
@media (prefers-color-scheme: dark) {
    .section-title {
        color: #e2e8f0 !important;
        border-left-color: #2dd4bf !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.4);
    }
}
/* Info box */
.info-box {
    background: #f0f9ff; border: 1px solid #bae6fd;
    border-radius: 10px; padding: 1rem 1.2rem;
    font-size: .875rem; color: #0369a1;
    margin-bottom: 1rem;
}
.warn-box {
    background: #fffbeb; border: 1px solid #fde68a;
    border-radius: 10px; padding: 1rem 1.2rem;
    font-size: .875rem; color: #92400e;
    margin-bottom: 1rem;
}
@media (prefers-color-scheme: dark) {
    .info-box {
        background: #0c2340 !important;
        border-color: #1e4976 !important;
        color: #7dd3fc !important;
    }
    .warn-box {
        background: #2d1f00 !important;
        border-color: #78450a !important;
        color: #fbbf24 !important;
    }
}
</style>
""", unsafe_allow_html=True)

# Konstanta 
ORDER       = ["High Value / Loyal", "Medium Value / Potential", "Low Value / Passive"]
COLOR_MAP   = {
    "High Value / Loyal":       "#2ecc71",
    "Medium Value / Potential": "#f39c12",
    "Low Value / Passive":      "#e74c3c",
}
FEATURES    = ["L_norm", "R_norm", "F_norm", "M_norm", "CLV"]
# CATATAN: M_FUZZY dihapus sebagai konstanta global karena sekarang
# diambil dari slider sidebar (m_fuzzy) agar UI dan komputasi sinkron.
MAX_ITER    = 300
ERROR       = 0.0001

# FUNGSI UTAMA
@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes) -> pd.DataFrame:
    buf = io.BytesIO(file_bytes)
    df1 = pd.read_excel(buf, sheet_name="Year 2009-2010", dtype={"Customer ID": str})
    buf.seek(0)
    df2 = pd.read_excel(buf, sheet_name="Year 2010-2011", dtype={"Customer ID": str})
    return pd.concat([df1, df2], ignore_index=True)

@st.cache_data(show_spinner=False)
def preprocess(raw_bytes: bytes):
    df_raw = load_data(raw_bytes)
    df = df_raw.copy()

    log = {}
    log["raw"]  = len(df)

    # Langkah 1: hapus duplikat
    df.drop_duplicates(inplace=True)
    log["dup"]  = log["raw"] - len(df)

    # Langkah 2: hapus baris tanpa Customer ID
    before_mis = len(df)
    df.dropna(subset=["Customer ID"], inplace=True)
    log["mis"]  = before_mis - len(df)

    # Langkah 3: hapus transaksi pembatalan (Invoice berawalan 'C')
    before_can = len(df)
    df = df[~df["Invoice"].astype(str).str.startswith("C")]
    log["can"]  = before_can - len(df)

    # Langkah 4: hapus Quantity <= 0
    before_qty = len(df)
    df = df[df["Quantity"] > 0]
    log["qty"]  = before_qty - len(df)

    # Langkah 5: hapus Price <= 0
    before_prc = len(df)
    df = df[df["Price"] > 0]
    log["prc"]  = before_prc - len(df)

    log["clean"]     = len(df)
    log["customers"] = df["Customer ID"].nunique()

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"]  = df["Quantity"] * df["Price"]
    df.reset_index(drop=True, inplace=True)
    return df, log

@st.cache_data(show_spinner=False)
def build_lrfm(clean_bytes: bytes):
    df, log = preprocess(clean_bytes)
    REF_DATE = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    lrfm = df.groupby("Customer ID").agg(
        first_purchase=("InvoiceDate", "min"),
        last_purchase =("InvoiceDate", "max"),
        Frequency     =("Invoice",     "nunique"),
        Monetary      =("TotalPrice",  "sum"),
    ).reset_index()

    lrfm["Length"]  = (lrfm["last_purchase"] - lrfm["first_purchase"]).dt.days
    lrfm["Recency"] = (REF_DATE - lrfm["last_purchase"]).dt.days
    lrfm = lrfm[["Customer ID", "Length", "Recency", "Frequency", "Monetary"]]

    # Normalisasi
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(lrfm[["Length", "Recency", "Frequency", "Monetary"]])
    df_norm = pd.DataFrame(scaled, columns=["L_norm", "R_norm", "F_norm", "M_norm"])
    df_norm["R_norm"] = 1 - df_norm["R_norm"]
    df_norm["CLV"]    = df_norm[["L_norm", "R_norm", "F_norm", "M_norm"]].mean(axis=1)
    df_norm.insert(0, "Customer ID", lrfm["Customer ID"].values)

    return lrfm, df_norm, REF_DATE, log

# [REVISI PRIORITAS TINGGI]
# evaluate_clusters sekarang menerima parameter m_fuzzy dari sidebar
# sehingga grafik evaluasi mencerminkan nilai fuzzifier yang dipilih user.
@st.cache_data(show_spinner=False)
def evaluate_clusters(raw_bytes: bytes, m_fuzzy: float = 2.0, k_range=(2, 9)):
    lrfm, df_norm, _, _ = build_lrfm(raw_bytes)
    X   = df_norm[FEATURES].values
    X_T = X.T
    results = {"k": [], "sse": [], "fpc": [], "sc": [], "dbi": []}

    for k in range(*k_range):
        cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
            X_T, c=k, m=m_fuzzy, error=ERROR, maxiter=MAX_ITER, init=None, seed=42
        )
        lbl = np.argmax(u, axis=0)
        sse = sum(np.sum((X[lbl == j] - cntr[j]) ** 2) for j in range(k))
        sc  = silhouette_score(X, lbl) if len(set(lbl)) > 1 else 0
        dbi = davies_bouldin_score(X, lbl) if len(set(lbl)) > 1 else 99
        results["k"].append(k)
        results["sse"].append(sse)
        results["fpc"].append(fpc)
        results["sc"].append(sc)
        results["dbi"].append(dbi)

    return pd.DataFrame(results)

# [REVISI PRIORITAS TINGGI]
# run_fcm sekarang menerima parameter m_fuzzy dari sidebar
# sehingga semua hasil clustering (profil, metrik, zona transisi)
# konsisten dengan fuzzifier yang dipilih user.
@st.cache_data(show_spinner=False)
def run_fcm(raw_bytes: bytes, k_optimal: int = 3, m_fuzzy: float = 2.0, threshold: float = 0.20):
    lrfm, df_norm, REF_DATE, log = build_lrfm(raw_bytes)
    X   = df_norm[FEATURES].values
    X_T = X.T

    cntr, u, _, _, _, p, fpc = fuzz.cluster.cmeans(
        X_T, c=k_optimal, m=m_fuzzy, error=ERROR, maxiter=MAX_ITER, init=None, seed=42
    )
    labels = np.argmax(u, axis=0)

    df_res = df_norm.copy()
    df_res["Cluster_Raw"] = labels
    for j in range(k_optimal):
        df_res[f"U_{j}"] = u[j]

    clv_mean        = df_res.groupby("Cluster_Raw")["CLV"].mean()
    sorted_clusters = clv_mean.sort_values(ascending=False).index.tolist()
    label_map = {
        sorted_clusters[0]: ORDER[0],
        sorted_clusters[1]: ORDER[1],
        sorted_clusters[2]: ORDER[2],
    }
    df_res["Segment"] = df_res["Cluster_Raw"].map(label_map)

    df_final = lrfm.copy()
    df_final["Segment"]  = df_res["Segment"].values
    df_final["CLV_norm"] = df_res["CLV"].values
    for j in range(k_optimal):
        df_final[f"U_Cluster_{j}"] = df_res[f"U_{j}"].values

    # Evaluasi
    sc_val  = silhouette_score(X, labels)
    dbi_val = davies_bouldin_score(X, labels)
    sse_val = sum(np.sum((X[labels == j] - cntr[j]) ** 2) for j in range(k_optimal))

    # Profil
    profile = df_final.groupby("Segment")[["Length","Recency","Frequency","Monetary","CLV_norm"]].mean()
    profile["Jumlah Pelanggan"] = df_final.groupby("Segment").size()
    profile["Persen (%)"]       = (profile["Jumlah Pelanggan"] / len(df_final) * 100).round(2)
    profile = profile.loc[[o for o in ORDER if o in profile.index]]

    # Centroid
    centroid_df = pd.DataFrame(cntr, columns=FEATURES)
    centroid_df["Segment"] = [label_map[j] for j in sorted_clusters]
    centroid_df = centroid_df.set_index("Segment")
    centroid_df = centroid_df.loc[[o for o in ORDER if o in centroid_df.index]]

    # Zona transisi
    u_max     = u.max(axis=0)
    u_gap     = u_max - np.sort(u, axis=0)[-2]
    df_final["U_gap"]    = u_gap
    df_final["Transisi"] = u_gap < threshold
    n_trans   = df_final["Transisi"].sum()
    pct_trans = n_trans / len(df_final) * 100

    metrics = {"fpc": fpc, "sc": sc_val, "dbi": dbi_val, "sse": sse_val,
               "iter": p, "n_trans": int(n_trans), "pct_trans": pct_trans}

    return df_final, df_res, profile, centroid_df, cntr, label_map, X, metrics, log


# SIDEBAR
with st.sidebar:
    st.markdown("## 📂 Upload Dataset")
    st.markdown("Dataset: **Online Retail II** (UCI ML Repository)")
    st.markdown("[🔗 Download dataset](https://archive.ics.uci.edu/dataset/502/online+retail+ii)",
                unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload `online_retail_II.xlsx`", type=["xlsx"])

    st.markdown("---")
    st.markdown("## ⚙️ Parameter FCM")
    k_optimal = st.slider("Jumlah Kluster (k)", min_value=2, max_value=8, value=3)
    m_fuzzy   = st.slider("Fuzziness (m)", min_value=1.1, max_value=3.0, value=2.0, step=0.1)
    threshold = st.slider("Ambang Transisi", min_value=0.05, max_value=0.40, value=0.20, step=0.05)

    st.markdown("---")
    st.markdown("## 📌 Tentang Aplikasi")
    st.markdown("""
Segmentasi pelanggan grosir menggunakan **Fuzzy C-Means (FCM)** berbasis model **LRFM-CLV**.

**Variabel:**
- **L** — Length (durasi hubungan)
- **R** — Recency (kebaruan transaksi)
- **F** — Frequency (frekuensi transaksi)
- **M** — Monetary (nilai pembelian)
- **CLV** — Customer Lifetime Value
    """)

# HEADER
st.markdown("""
<div class="main-header">
    <h1>Segmentasi Pelanggan Grosir</h1>
    <p>Fuzzy C-Means Clustering · Model LRFM-CLV · Online Retail II Dataset</p>
</div>
""", unsafe_allow_html=True)

# KONDISI: BELUM UPLOAD
if uploaded is None:
    st.markdown("""
    <div class="info-box">
        ⬅️ <strong>Upload file <code>online_retail_II.xlsx</code> di sidebar kiri</strong> untuk memulai analisis.
        File dapat diunduh dari UCI Machine Learning Repository.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    steps = [
        ("1️⃣", "Upload Dataset", "File Excel Online Retail II dari UCI ML Repository"),
        ("2️⃣", "Preprocessing & LRFM", "Cleaning data dan konstruksi 5 fitur LRFM-CLV"),
        ("3️⃣", "FCM Clustering", "Segmentasi pelanggan dengan Fuzzy C-Means"),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3], steps):
        col.markdown(f"""
        <div class="metric-card">
            <div style="font-size:2rem">{icon}</div>
            <div style="font-weight:700;margin:.5rem 0 .3rem">{title}</div>
            <div style="font-size:.8rem;color:#6b7280">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# PROSES DATA
raw_bytes = uploaded.read()

with st.spinner("🔄 Memproses data dan menjalankan FCM..."):
    try:
        # [REVISI PRIORITAS TINGGI] — teruskan m_fuzzy dari slider ke run_fcm
        df_final, df_res, profile, centroid_df, cntr, label_map, X, metrics, log = run_fcm(
            raw_bytes, k_optimal=k_optimal, m_fuzzy=m_fuzzy, threshold=threshold
        )
        lrfm, df_norm, REF_DATE, _ = build_lrfm(raw_bytes)
        # [REVISI PRIORITAS TINGGI] — teruskan m_fuzzy dari slider ke evaluate_clusters
        df_eval = evaluate_clusters(raw_bytes, m_fuzzy=m_fuzzy)
    except Exception as e:
        st.error(f"❌ Terjadi error: {e}")
        st.stop()

# TABS UTAMA
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Ringkasan",
    "Preprocessing",
    "Analisis LRFM",
    "Clustering",
    "Visualisasi",
])

# TAB 1 — RINGKASAN
with tab1:
    st.markdown('<div class="section-title">Ringkasan Eksekutif</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("Total Pelanggan", f"{len(df_final):,}", "setelah preprocessing"),
        ("Kluster (k)", str(k_optimal), "Fuzzy C-Means"),
        ("FPC", f"{metrics['fpc']:.4f}", "mendekati 1 = baik"),
        ("Silhouette", f"{metrics['sc']:.4f}", "mendekati 1 = baik"),
    ]
    for col, (label, val, sub) in zip([c1, c2, c3, c4], cards):
        col.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{val}</div>
            <div class="sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    c5, c6, c7, c8 = st.columns(4)
    cards2 = [
        ("DBI", f"{metrics['dbi']:.4f}", "mendekati 0 = baik"),
        ("SSE", f"{metrics['sse']:.2f}", "Elbow Method"),
        ("Iterasi FCM", str(metrics["iter"]), "hingga konvergen"),
        ("Zona Transisi", f"{metrics['n_trans']:,}", f"{metrics['pct_trans']:.1f}% pelanggan"),
    ]
    for col, (label, val, sub) in zip([c5, c6, c7, c8], cards2):
        col.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{val}</div>
            <div class="sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Distribusi Segmen Pelanggan</div>', unsafe_allow_html=True)

    badge = {"High Value / Loyal": "badge-high", "Medium Value / Potential": "badge-medium", "Low Value / Passive": "badge-low"}
    icon  = {"High Value / Loyal": "🟢", "Medium Value / Potential": "🟡", "Low Value / Passive": "🔴"}
    strat = {
        "High Value / Loyal":       "Prioritas pengiriman, jaga ketersediaan stok, pertahankan kualitas layanan.",
        "Medium Value / Potential": "Promosi khusus, penawaran paket produk, distribusi fleksibel.",
        "Low Value / Passive":      "Distribusi selektif, program reaktivasi, survei kepuasan.",
    }

    cols = st.columns(3)
    for col, seg in zip(cols, ORDER):
        if seg not in profile.index:
            continue
        n   = int(profile.loc[seg, "Jumlah Pelanggan"])
        pct = profile.loc[seg, "Persen (%)"]
        clv = profile.loc[seg, "CLV_norm"]
        col.markdown(f"""
        <div class="metric-card" style="text-align:left">
            <div style="font-size:1.5rem;margin-bottom:.5rem">{icon[seg]}</div>
            <div style="font-weight:800;font-size:1rem;margin-bottom:.3rem">{seg}</div>
            <div style="font-size:1.6rem;font-weight:800;font-family:'JetBrains Mono',monospace">{n:,}</div>
            <div style="font-size:.8rem;color:#6b7280;margin-bottom:.6rem">{pct}% dari total pelanggan</div>
            <div style="font-size:.78rem;color:#374151;line-height:1.5">{strat[seg]}</div>
            <div style="margin-top:.6rem;font-size:.75rem;color:#6b7280">CLV rata-rata: <strong>{clv:.4f}</strong></div>
        </div>
        """, unsafe_allow_html=True)

# TAB 2 — PREPROCESSING
with tab2:
    st.markdown('<div class="section-title">Hasil Pembersihan Data</div>', unsafe_allow_html=True)

    # [REVISI PRIORITAS SEDANG]
    # Menampilkan 7 kartu terpisah (konsisten dengan laporan & notebook):
    # Data Awal | Duplikat | Missing CustID | Pembatalan | Qty≤0 | Price≤0 | Data Bersih
    c1, c2, c3, c4 = st.columns(4)
    pre_cards_row1 = [
        ("Data Awal",      f"{log['raw']:,}", "total baris"),
        ("Duplikat",       f"{log['dup']:,}", "dihapus"),
        ("Missing CustID", f"{log['mis']:,}", "dihapus"),
        ("Pembatalan",     f"{log['can']:,}", "invoice 'C'"),
    ]
    for col, (label, val, sub) in zip([c1, c2, c3, c4], pre_cards_row1):
        col.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value" style="font-size:1.4rem">{val}</div>
            <div class="sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    c5, c6, c7 = st.columns(3)
    pre_cards_row2 = [
        ("Quantity ≤ 0", f"{log['qty']:,}", "dihapus"),
        ("Price ≤ 0",    f"{log['prc']:,}", "dihapus"),
        ("Data Bersih",  f"{log['clean']:,}", f"{log['customers']:,} pelanggan"),
    ]
    for col, (label, val, sub) in zip([c5, c6, c7], pre_cards_row2):
        col.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value" style="font-size:1.4rem">{val}</div>
            <div class="sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Statistik Deskriptif Fitur LRFM</div>', unsafe_allow_html=True)
    st.dataframe(
        lrfm[["Length","Recency","Frequency","Monetary"]].describe().round(2),
        use_container_width=True
    )

    st.markdown('<div class="section-title">Data LRFM Ternormalisasi (Min-Max) + CLV</div>', unsafe_allow_html=True)
    st.dataframe(df_norm.head(20), use_container_width=True)

    st.markdown('<div class="section-title">Distribusi Fitur (Histogram)</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
    cols_hist = ["L_norm","R_norm","F_norm","M_norm","CLV"]
    labels_h  = ["Length","Recency","Frequency","Monetary","CLV"]
    colors_h  = ["#0891b2","#8b5cf6","#10b981","#f59e0b","#ef4444"]
    for ax, col, lbl, clr in zip(axes, cols_hist, labels_h, colors_h):
        ax.hist(df_norm[col], bins=40, color=clr, alpha=.85, edgecolor="white")
        ax.set_title(lbl, fontweight="bold", fontsize=10)
        ax.set_xlabel("Nilai Normalisasi")
        ax.grid(axis="y", linestyle="--", alpha=.4)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

# TAB 3 — ANALISIS LRFM
with tab3:
    st.markdown('<div class="section-title">Evaluasi Jumlah Kluster Optimal (k = 2 – 8)</div>',
                unsafe_allow_html=True)
    st.dataframe(df_eval.set_index("k").round(4), use_container_width=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Penentuan Jumlah Kluster Optimal — Multi-Indikator",
                 fontsize=13, fontweight="bold", y=1.01)
    metrics_plot = [
        ("sse", "SSE (Elbow Method)",          "#0891b2"),
        ("fpc", "Fuzzy Partition Coefficient", "#10b981"),
        ("sc",  "Silhouette Coefficient",      "#f59e0b"),
        ("dbi", "Davies-Bouldin Index",        "#ef4444"),
    ]
    for ax, (col, title, clr) in zip(axes.flatten(), metrics_plot):
        ax.plot(df_eval["k"], df_eval[col], marker="o", color=clr, lw=2, ms=7)
        if col != "sse":
            ax.axvline(x=k_optimal, color="gray", linestyle="--", alpha=.5, label=f"k={k_optimal}")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Jumlah Kluster (k)")
        ax.set_xticks(list(range(2, 9)))
        ax.grid(True, linestyle="--", alpha=.4)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="section-title">Statistik CLV per Segmen</div>', unsafe_allow_html=True)
    clv_stats = df_final.groupby("Segment")["CLV_norm"].describe().round(4)
    clv_stats = clv_stats.loc[[o for o in ORDER if o in clv_stats.index]]
    st.dataframe(clv_stats, use_container_width=True)

# TAB 4 — CLUSTERING
with tab4:
    st.markdown('<div class="section-title">Evaluasi Kualitas Clustering</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    eval_cards = [
        ("FPC", f"{metrics['fpc']:.4f}", "≈ 1 = baik", "#10b981"),
        ("SC",  f"{metrics['sc']:.4f}",  "≈ 1 = baik", "#0891b2"),
        ("DBI", f"{metrics['dbi']:.4f}", "≈ 0 = baik", "#f59e0b"),
        ("SSE", f"{metrics['sse']:.2f}", "lebih kecil lebih baik", "#8b5cf6"),
    ]
    for col, (label, val, sub, clr) in zip([c1,c2,c3,c4], eval_cards):
        col.markdown(f"""
        <div class="metric-card" style="border-top:4px solid {clr}">
            <div class="label">{label}</div>
            <div class="value">{val}</div>
            <div class="sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Profil Rata-Rata Tiap Segmen</div>', unsafe_allow_html=True)
    st.dataframe(profile.round(3), use_container_width=True)

    st.markdown('<div class="section-title">Centroid Kluster (Ruang Ternormalisasi)</div>', unsafe_allow_html=True)
    st.dataframe(centroid_df.round(4), use_container_width=True)

    st.markdown('<div class="section-title">Analisis Zona Transisi</div>', unsafe_allow_html=True)
    trans_df = df_final[df_final["Transisi"]][["Customer ID","Segment","CLV_norm","U_gap"]].reset_index(drop=True)
    st.info(f"🔀 **{metrics['n_trans']:,} pelanggan ({metrics['pct_trans']:.1f}%)** berada di zona transisi "
            f"(U_gap < {threshold}) — keanggotaan tidak tegas ke satu segmen.")
    st.dataframe(trans_df.head(50), use_container_width=True)

    st.markdown('<div class="section-title">Unduh Hasil Segmentasi</div>', unsafe_allow_html=True)
    csv = df_final.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download CSV Hasil Segmentasi",
        data=csv,
        file_name="hasil_segmentasi_lrfm_clv.csv",
        mime="text/csv",
    )

    st.markdown('<div class="section-title">Pencarian Pelanggan</div>', unsafe_allow_html=True)
    search_id = st.text_input("Masukkan Customer ID:", placeholder="contoh: 12345")
    if search_id:
        result = df_final[df_final["Customer ID"] == search_id]
        if len(result) == 0:
            st.warning("Customer ID tidak ditemukan.")
        else:
            row = result.iloc[0]
            seg = row["Segment"]
            badge_cls = "badge-high" if "High" in seg else ("badge-medium" if "Medium" in seg else "badge-low")
            st.markdown(f"""
            <div class="metric-card" style="text-align:left;max-width:500px">
                <div style="margin-bottom:.8rem">
                    <strong>Customer ID:</strong> {row['Customer ID']} &nbsp;
                    <span class="{badge_cls}">{seg}</span>
                </div>
                <table style="width:100%;font-size:.88rem">
                    <tr><td style="color:#6b7280">Length</td><td><strong>{row['Length']:.0f} hari</strong></td></tr>
                    <tr><td style="color:#6b7280">Recency</td><td><strong>{row['Recency']:.0f} hari</strong></td></tr>
                    <tr><td style="color:#6b7280">Frequency</td><td><strong>{row['Frequency']:.0f} transaksi</strong></td></tr>
                    <tr><td style="color:#6b7280">Monetary</td><td><strong>£{row['Monetary']:,.2f}</strong></td></tr>
                    <tr><td style="color:#6b7280">CLV (norm)</td><td><strong>{row['CLV_norm']:.4f}</strong></td></tr>
                    <tr><td style="color:#6b7280">Zona Transisi</td><td><strong>{"Ya" if row['Transisi'] else "Tidak"}</strong></td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

# TAB 5 — VISUALISASI
with tab5:
    # Bar chart distribusi
    st.markdown('<div class="section-title">Distribusi Pelanggan per Segmen</div>', unsafe_allow_html=True)
    seg_counts = df_final["Segment"].value_counts().reindex(ORDER).dropna()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(seg_counts.index, seg_counts.values,
                  color=[COLOR_MAP[s] for s in seg_counts.index], edgecolor="white", width=.5)
    for bar, val in zip(bars, seg_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f"{val:,}", ha="center", va="bottom", fontweight="bold")
    ax.set_xlabel("Segmen Pelanggan")
    ax.set_ylabel("Jumlah Pelanggan")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="y", linestyle="--", alpha=.4)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Radar chart 
    st.markdown('<div class="section-title">Radar Chart Profil LRFM-CLV per Segmen</div>', unsafe_allow_html=True)
    features_r = ["L_norm","R_norm","F_norm","M_norm","CLV"]
    labels_r   = ["Length","Recency","Frequency","Monetary","CLV"]
    radar_mean = df_res.groupby("Segment")[features_r].mean()
    radar_mean = radar_mean.loc[[o for o in ORDER if o in radar_mean.index]]
    N = len(labels_r)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for seg in radar_mean.index:
        vals = radar_mean.loc[seg].tolist() + radar_mean.loc[seg].tolist()[:1]
        ax.plot(angles, vals, color=COLOR_MAP[seg], lw=2, label=seg)
        ax.fill(angles, vals, color=COLOR_MAP[seg], alpha=.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_r, fontsize=10)
    ax.set_yticklabels([])
    ax.set_title("Profil LRFM-CLV per Segmen", fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    col_l, col_r = st.columns(2)

    # Scatter F vs M 
    with col_l:
        st.markdown('<div class="section-title">Scatter: Frequency vs Monetary</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for seg in ORDER:
            mask = df_final["Segment"] == seg
            ax.scatter(df_final.loc[mask,"Frequency"], df_final.loc[mask,"Monetary"],
                       c=COLOR_MAP[seg], label=seg, alpha=.5, s=25, edgecolors="none")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Monetary")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=.35)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # PCA 2D 
    with col_r:
        st.markdown('<div class="section-title">PCA 2D — Distribusi Kluster</div>', unsafe_allow_html=True)
        pca   = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for seg in ORDER:
            mask = df_final["Segment"] == seg
            ax.scatter(X_pca[mask,0], X_pca[mask,1],
                       c=COLOR_MAP[seg], label=seg, alpha=.5, s=25, edgecolors="none")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=.35)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    #  Heatmap 
    st.markdown('<div class="section-title">Heatmap Rata-Rata LRFM-CLV per Segmen</div>', unsafe_allow_html=True)
    hmap = radar_mean.copy()
    hmap.columns = ["Length","Recency","Frequency","Monetary","CLV"]
    fig, ax = plt.subplots(figsize=(9, 3))
    sns.heatmap(hmap, annot=True, fmt=".3f", cmap="RdYlGn",
                linewidths=.5, ax=ax, cbar_kws={"label":"Nilai Normalisasi"})
    ax.set_ylabel("")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Box plot 
    st.markdown('<div class="section-title">Box Plot Distribusi LRFM per Segmen</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    palette = {s: COLOR_MAP[s] for s in ORDER}
    for ax, col in zip(axes, ["Length","Recency","Frequency","Monetary"]):
        sns.boxplot(data=df_final[df_final["Segment"].isin(ORDER)],
                    x="Segment", y=col, order=ORDER, palette=palette, ax=ax, width=.5)
        ax.set_title(col, fontweight="bold")
        ax.set_xlabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=.4)
        if col == "Monetary":
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()
