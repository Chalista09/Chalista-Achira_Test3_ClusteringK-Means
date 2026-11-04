import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples

# ---------------- KONFIGURASI ----------------
DATA_FILE = "Chalista_Achira_Test3_ClusteringK-Means.csv"
SEED = 42

st.set_page_config(page_title="Clustering Gempa K-Means", page_icon="🌍", layout="wide")
sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#FFFDF0", "figure.facecolor": "#FFF8DC",
                                      "grid.color": "#F0E0A0", "axes.edgecolor": "#B8860B"})

st.markdown("""
<style>
.stApp { background-color: #FFF8DC; }
[data-testid="stSidebar"] { background-color: #FFE680; }
h1, h2, h3 { color: #7A5C00 !important; }
h1 { border-bottom: 4px solid #F2B705; padding-bottom: .3rem; }
[data-testid="stMetric"] { background: #FFEB99; border-left: 6px solid #F2B705; padding: .6rem 1rem; border-radius: 8px; }
[data-testid="stAlert"] { background-color: #FFF3B0; border-left: 6px solid #F2B705; color: #3B2F00; }
[data-testid="stAlert"] svg { display: none; }
.stDownloadButton button { background-color: #F2B705; color: #3B2F00; border: none; font-weight: 600; }
.stDownloadButton button:hover { background-color: #D99E00; color: #fff; }
</style>
""", unsafe_allow_html=True)

WARNA = ["#B8860B", "#1F3A5F", "#D95F02", "#6A3D9A", "#2E7D32", "#C2185B", "#795548", "#00838F", "#F9A825", "#455A64"]

# ---------------- FUNGSI BANTU ----------------
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_data
def hitung_k(X, k_min, k_max, sample=5000):
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    ks, ine, sil = list(range(k_min, k_max + 1)), [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=5, random_state=SEED).fit(X)
        ine.append(km.inertia_)
        sil.append(silhouette_score(X[idx], km.labels_[idx]))
    return ks, ine, sil

def titik_siku(ks, ine):
    x = np.array(ks, float); y = np.array(ine, float)
    x = (x - x.min()) / (x.max() - x.min()); y = (y - y.min()) / (y.max() - y.min())
    p1, p2 = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
    v = p2 - p1
    w = np.c_[x, y] - p1
    d = np.abs(v[0] * w[:, 1] - v[1] * w[:, 0]) / np.linalg.norm(v)
    return ks[int(np.argmax(d))]

def kategori_sil(s):
    if s > 0.70: return "kuat"
    if s > 0.50: return "baik"
    if s > 0.25: return "lemah (ada tumpang tindih antar klaster)"
    return "tidak jelas"

# ---------------- LOAD DATA ----------------
try:
    df = load_data(DATA_FILE).copy()
except FileNotFoundError:
    st.error(f"File {DATA_FILE} tidak ditemukan. Jalankan notebook terlebih dahulu & taruh CSV di folder yang sama.")
    st.stop()

SCALED = [c for c in df.columns if c.endswith("_scaled")]
if SCALED:
    # Format 1: CSV punya kolom *_scaled (dan biasanya kolom aslinya)
    FITUR = [c[:-len("_scaled")] for c in SCALED]
    for f, sc in zip(FITUR, SCALED):
        if f not in df.columns:          # kolom asli tidak ada -> pakai versi scaled
            df[f] = df[sc]
    X = df[SCALED].fillna(0).values
else:
    # Format 2: CSV hanya berisi kolom numerik (mentah atau sudah dinormalisasi)
    FITUR = [c for c in df.select_dtypes(include=np.number).columns if c.lower() != "cluster"]
    if len(FITUR) < 2:
        st.error("Dataset harus punya minimal 2 kolom numerik untuk clustering.")
        st.stop()
    df[FITUR] = df[FITUR].fillna(df[FITUR].median())
    X = StandardScaler().fit_transform(df[FITUR].values)
    st.warning("Kolom *_scaled tidak ditemukan, jadi semua kolom numerik distandarisasi otomatis oleh dashboard.")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Pengaturan")
K = st.sidebar.slider("Jumlah klaster (K)", 2, 10, 4)
fx = st.sidebar.selectbox("Fitur sumbu X", FITUR, index=FITUR.index("depth") if "depth" in FITUR else 0)
fy = st.sidebar.selectbox("Fitur sumbu Y", FITUR, index=FITUR.index("mag") if "mag" in FITUR else 1)
n_plot = st.sidebar.slider("Jumlah titik pada plot", 1000, 20000, 8000, 1000)
st.sidebar.caption(f"Fitur clustering: {', '.join(FITUR)}")

# ---------------- CLUSTERING ----------------
km = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(X)
df["cluster"] = km.labels_
rng = np.random.RandomState(SEED)
idx_s = rng.choice(len(X), size=min(5000, len(X)), replace=False)
sil_k = silhouette_score(X[idx_s], km.labels_[idx_s])
palet = WARNA[:K]

# ---------------- 1. JUDUL ----------------
st.title("🌍 Clustering Gempa Bumi Global dengan K-Means")
st.markdown("""
Dashboard ini menganalisis **Global Earthquakes Seismic Telemetry** (Kaggle) menggunakan algoritma **K-Means**
untuk mengelompokkan kejadian gempa berdasarkan karakteristik seismiknya. Fitur sudah distandarisasi (Z-score)
sehingga setiap fitur berkontribusi seimbang pada perhitungan jarak.
""")

# ---------------- 2. RINGKASAN DATASET ----------------
st.header("1. Ringkasan Dataset")
c1, c2, c3 = st.columns(3)
c1.metric("Jumlah data", f"{len(df):,}")
c2.metric("Jumlah fitur clustering", len(FITUR))
c3.metric("Klaster (K)", K)
st.write("**Fitur:**", ", ".join(FITUR))
st.dataframe(df[FITUR].describe().T.round(3))
with st.expander("Lihat 20 baris pertama"):
    st.dataframe(df.head(20))
st.info("**Analisis:** Tabel di atas menunjukkan mean, std, min, kuartil, dan max tiap fitur. "
        "Perbedaan besar antara mean dan median (50%) menandakan distribusi miring dan potensi outlier.")

# ---------------- 3. DISTRIBUSI ----------------
st.header("2. Distribusi Fitur")
fd = st.selectbox("Pilih fitur untuk dilihat distribusinya", FITUR)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
sns.histplot(df[fd], bins=40, kde=True, ax=ax[0], color="#F2B705"); ax[0].set_title(f"Histogram {fd}")
bp = ax[1].boxplot(df[fd].dropna(), patch_artist=True, flierprops={"marker": ".", "markersize": 3, "alpha": .4},
                   medianprops={"color": "#7A5C00"})
bp["boxes"][0].set_facecolor("#FFD54F"); ax[1].set_xticks([]); ax[1].set_title(f"Boxplot {fd}")
plt.tight_layout(); st.pyplot(fig)
sk = df[fd].skew()
arah = "kanan (banyak nilai kecil, sedikit nilai besar)" if sk > 0.5 else "kiri" if sk < -0.5 else "relatif simetris"
st.info(f"**Analisis:** Fitur `{fd}` memiliki mean {df[fd].mean():.3f}, median {df[fd].median():.3f}, dan skewness {sk:.2f}, "
        f"sehingga distribusinya condong ke {arah}. Titik di luar whisker boxplot merupakan kandidat nilai ekstrem.")

# ---------------- 4. KORELASI ----------------
st.header("3. Korelasi Antar Fitur")
corr = df[FITUR].corr()
fig, ax = plt.subplots(figsize=(6.5, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="BrBG", center=0, square=True, ax=ax)
plt.tight_layout(); st.pyplot(fig)
pair = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack()
top = pair.abs().idxmax()
st.info(f"**Analisis:** Korelasi terkuat ada pada `{top[0]}` dan `{top[1]}` (r = {pair[top]:.2f}). "
        + ("Korelasi tinggi menandakan informasi yang tumpang tindih antar fitur." if abs(pair[top]) > 0.7
           else "Tidak ada pasangan fitur yang sangat berkorelasi (|r| < 0.7), sehingga tiap fitur membawa informasi yang relatif berbeda untuk clustering."))

# ---------------- 5. ELBOW & SILHOUETTE ----------------
st.header("4. Elbow Method & Silhouette Score")
ks, ine, sil = hitung_k(X, 2, 10)
k_elbow, k_sil = titik_siku(ks, ine), ks[int(np.argmax(sil))]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(ks, ine, "o-", color="#B8860B"); ax[0].axvline(k_elbow, color="red", ls="--"); ax[0].axvline(K, color="green", ls=":")
ax[0].set_title("Elbow (Inertia)"); ax[0].set_xlabel("K")
ax[1].plot(ks, sil, "s-", color="#F2B705"); ax[1].axvline(k_sil, color="red", ls="--"); ax[1].axvline(K, color="green", ls=":")
ax[1].set_title("Silhouette Score"); ax[1].set_xlabel("K")
plt.tight_layout(); st.pyplot(fig)
st.info(f"**Analisis:** Inertia turun terus ketika K bertambah; titik siku terdeteksi pada **K = {k_elbow}**. "
        f"Silhouette tertinggi ({max(sil):.3f}) ada pada **K = {k_sil}**. Garis hijau putus-putus adalah K pilihanmu (K = {K}) "
        f"dengan silhouette {sil_k:.3f} dan inertia {km.inertia_:,.1f}.")

# ---------------- 6. SCATTER ----------------
st.header("5. Visualisasi Hasil Clustering")
samp = df.sample(min(n_plot, len(df)), random_state=SEED)
cent = pd.DataFrame(km.cluster_centers_, columns=FITUR)
# kembalikan centroid ke skala asli
cent_asli = df.groupby("cluster")[FITUR].mean()   # centroid pada skala asli

col_a, col_b = st.columns(2)
with col_a:
    fig, ax = plt.subplots(figsize=(6, 4.8))
    sns.scatterplot(data=samp, x=fx, y=fy, hue="cluster", palette=palet, s=14, alpha=.6, ax=ax)
    ax.scatter(cent_asli[fx], cent_asli[fy], c="black", marker="X", s=180, edgecolor="white", label="Centroid")
    ax.set_title(f"{fx} vs {fy}"); ax.legend(title="Cluster", fontsize=7)
    plt.tight_layout(); st.pyplot(fig)
with col_b:
    pca = PCA(n_components=2, random_state=SEED); pc = pca.fit_transform(X)
    ids = samp.index.values
    fig, ax = plt.subplots(figsize=(6, 4.8))
    sns.scatterplot(x=pc[ids, 0], y=pc[ids, 1], hue=df.loc[ids, "cluster"], palette=palet, s=14, alpha=.6, ax=ax)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("Proyeksi PCA 2D"); ax.legend(title="Cluster", fontsize=7)
    plt.tight_layout(); st.pyplot(fig)
ukuran = df["cluster"].value_counts().sort_index()
st.info(f"**Analisis:** Scatter kiri menampilkan klaster pada fitur `{fx}` dan `{fy}` (X hitam = centroid), sedangkan scatter kanan "
        f"meringkas seluruh fitur ke 2 komponen utama (menjelaskan {pca.explained_variance_ratio_.sum()*100:.1f}% varians). "
        f"Klaster terbesar adalah cluster {ukuran.idxmax()} ({ukuran.max():,} data, {ukuran.max()/len(df)*100:.1f}%) "
        f"dan terkecil cluster {ukuran.idxmin()} ({ukuran.min():,} data). Silhouette {sil_k:.3f} menunjukkan struktur klaster {kategori_sil(sil_k)}.")

if {"latitude", "longitude"}.issubset(df.columns):
    st.subheader("Sebaran Geografis Klaster")
    fig, ax = plt.subplots(figsize=(11, 4.8))
    sns.scatterplot(data=samp, x="longitude", y="latitude", hue="cluster", palette=palet, s=8, alpha=.6, ax=ax)
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90); ax.legend(title="Cluster", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout(); st.pyplot(fig)
    st.info("**Analisis:** Peta menunjukkan apakah klaster memiliki pola spasial. Gempa umumnya mengikuti batas lempeng "
            "(mis. Cincin Api Pasifik). Klaster yang terkonsentrasi di zona tertentu menandakan karakteristik seismik khas wilayah tersebut, "
            "meskipun koordinat tidak dipakai sebagai fitur clustering.")

# ---------------- 7. TABEL KARAKTERISTIK ----------------
st.header("6. Karakteristik Tiap Klaster")
agg = st.radio("Statistik", ["Mean", "Median"], horizontal=True)
prof = (df.groupby("cluster")[FITUR].mean() if agg == "Mean" else df.groupby("cluster")[FITUR].median()).round(3)
prof.insert(0, "jumlah_data", ukuran)
st.dataframe(prof)

z = (df.groupby("cluster")[FITUR].mean() - df[FITUR].mean()) / df[FITUR].std()
fig, ax = plt.subplots(figsize=(10, 4))
z.plot(kind="bar", ax=ax, width=.8, color=WARNA[:len(FITUR)]); ax.axhline(0, color="black", lw=.8)
ax.set_ylabel("Selisih dari rata-rata global (std)"); ax.set_title("Profil Klaster (z-score)"); plt.xticks(rotation=0)
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left"); plt.tight_layout(); st.pyplot(fig)

ringkas = []
for c in z.index:
    t = z.loc[c][z.loc[c] > 0.5].sort_values(ascending=False).index.tolist()
    r = z.loc[c][z.loc[c] < -0.5].sort_values().index.tolist()
    ringkas.append(f"- **Cluster {c}** ({ukuran[c]:,} data): nilai tinggi pada {', '.join(t) if t else '-'}; nilai rendah pada {', '.join(r) if r else '-'}.")
st.info("**Analisis:** Bar chart menampilkan seberapa jauh rata-rata tiap klaster dari rata-rata global (satuan standar deviasi). "
        "Batang positif = di atas rata-rata, negatif = di bawah.\n\n" + "\n".join(ringkas))

# ---------------- 8. DOWNLOAD ----------------
st.header("7. Unduh Hasil")
st.download_button("⬇️ Unduh data + label klaster (CSV)", df.to_csv(index=False).encode("utf-8"),
                   file_name=f"hasil_clustering_K{K}.csv", mime="text/csv")
st.caption("Praktikum Clustering K-Means | Dataset: Global Earthquakes Seismic Telemetry (Kaggle)")