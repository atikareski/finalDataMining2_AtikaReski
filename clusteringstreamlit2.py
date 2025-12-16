import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import sys
from sklearn.cluster import KMeans

# --- KONSTANTA BARU UNTUK DATA SUPERSTORE ---
K_FIXED = 2 
SPENDING_COLS = ['Total_Sales', 'Total_Profit', 'Total_Quantity', 'Frequency']
MODEL_FILES = {
    'scaler': 'scaler_superstore_final.pkl',
    'model_logistic': 'model_logistic_superstore_final.pkl',
    'pca': 'pca_superstore_final.pkl',
    # Kita tidak menyimpan 'pca_data_historis.pkl' di kode terakhir, 
    # jadi kita akan membuat ulang PCA data historis dari file data asli
}

# Data profil kluster rata-rata (dibulatkan dari analisis K=2)
# Kluster 0: High-Value, Kluster 1: Standard
CLUSTER_PROFILES_DATA = {
    'Total_Sales': [4908.43, 1659.58], 
    'Total_Profit': [748.93, 122.65], 
    'Total_Quantity': [71.05, 33.43], 
    'Frequency': [8.62, 4.90]
}

CLUSTER_PROFILES_DF = pd.DataFrame(CLUSTER_PROFILES_DATA, index=['0', '1']).T 
warnings.filterwarnings('ignore', category=UserWarning) 

@st.cache_resource
def load_and_preprocess_models():
    """Memuat semua model lokal dan membuat ulang data historis PCA."""
    
    # Cek ketersediaan file model
    if not all(os.path.exists(f) for f in MODEL_FILES.values()):
        st.error(f"ERROR: Satu atau lebih file model (.pkl) tidak ditemukan secara lokal. Harap pastikan file berikut ada di direktori yang sama: {list(MODEL_FILES.values())}")
        st.stop()

    try:
        scaler = joblib.load(MODEL_FILES['scaler'])
        model_logistic = joblib.load(MODEL_FILES['model_logistic'])
        pca = joblib.load(MODEL_FILES['pca'])
    except Exception as e:
        st.error(f"Error memuat model: {e}")
        st.stop()
        
    # --- Membuat ulang X_means dan data historis PCA (karena tidak disimpan sebelumnya) ---
    df_raw = pd.read_csv("Sample - Superstore.csv", encoding='latin1')
    
    customer_df = df_raw.groupby('Customer ID').agg(
        Total_Sales=('Sales', 'sum'),
        Total_Profit=('Profit', 'sum'),
        Total_Quantity=('Quantity', 'sum'),
        Frequency=('Order ID', 'nunique')
    ).reset_index()
    
    X = customer_df[SPENDING_COLS]
    
    # Transformasi data historis menggunakan scaler dan pca yang dimuat
    X_scaled_historis = scaler.transform(X)
    principal_components_historis = pca.transform(X_scaled_historis)
    
    pca_data_historis = pd.DataFrame(data=principal_components_historis, columns=['PC1', 'PC2'])
    
    # Label Kluster harus dibuat ulang karena hanya model yang disimpan
    kmeans = KMeans(n_clusters=K_FIXED, random_state=42, n_init=10)
    pca_data_historis['Cluster'] = kmeans.fit_predict(X_scaled_historis)
    
    X_means = pd.Series(scaler.mean_, index=SPENDING_COLS).round(0).astype(int)

    return scaler, model_logistic, pca, pca_data_historis, X_means

# Panggil fungsi load model
scaler, model_logistic, pca, pca_data_historis, X_means = load_and_preprocess_models() 

st.set_page_config(layout="wide")
st.title("📦 Prediksi Segmen Pelanggan Superstore Baru (K=2)")
st.caption("Model Regresi Logistik dilatih untuk mengklasifikasikan pelanggan ke dalam Kluster High-Value atau Standard.")

def plot_pca_clusters(pca_data_historis, new_point=None, predicted_cluster=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(
        pca_data_historis['PC1'], 
        pca_data_historis['PC2'], 
        c=pca_data_historis['Cluster'],
        cmap='viridis', 
        marker='o', 
        s=50, 
        alpha=0.6,
        label='Data Historis'
    )
    
    if new_point is not None:
        ax.scatter(
            new_point[0, 0], 
            new_point[0, 1], 
            color='red', 
            marker='*', 
            s=500, 
            edgecolors='black', 
            linewidth=1.5,
            label=f'Pelanggan Baru (Kluster {predicted_cluster})'
        )
        ax.annotate(
            'Prediksi', 
            (new_point[0, 0], new_point[0, 1]),
            textcoords="offset points", 
            xytext=(10, 10), 
            ha='center', 
            fontsize=12, 
            color='red'
        )
        
    ax.set_title(f'Peta Segmentasi Pelanggan (K={K_FIXED})', fontsize=16)
    ax.set_xlabel('Principal Component 1 (PC1) - Variansi 59.22%', fontsize=12)
    ax.set_ylabel('Principal Component 2 (PC2) - Variansi 25.50%', fontsize=12)
    
    legend1 = ax.legend(*scatter.legend_elements(), title="Kluster", loc="lower left", title_fontsize=12, fontsize=10)
    ax.add_artist(legend1)
    
    if new_point is not None:
        ax.legend(loc="upper right")
        
    ax.grid(True, linestyle='--', alpha=0.6)
    return fig

# --- STREAMLIT UI ---

st.sidebar.header("Uji Prediksi Pelanggan Baru")
st.sidebar.markdown("Masukkan metrik agregat pelanggan (Total $):")

input_values = {}
for col_name in SPENDING_COLS:
    # Mengatur default value menggunakan mean (rata-rata) dari data historis
    default_mean = X_means[col_name] if X_means is not None else 300
    
    if 'Sales' in col_name or 'Profit' in col_name:
         input_values[col_name] = st.sidebar.number_input(
            f'{col_name} Rata-rata: {default_mean:,.0f}', 
            min_value=-50000.0, 
            value=float(default_mean),
            key=f'input_{col_name}',
            format='%.2f'
        )
    else:
        # Untuk Quantity dan Frequency
        input_values[col_name] = st.sidebar.number_input(
            f'{col_name} Rata-rata: {default_mean:,.0f}', 
            min_value=0, 
            value=int(default_mean),
            key=f'input_{col_name}'
        )

predict_button = st.sidebar.button("Prediksi Segmen Pelanggan")

st.header("1. Peta Segmentasi Pelanggan Historis")
col_pca_display, col_results_display = st.columns([2, 1])

with col_pca_display:
    st.subheader("Peta Segmentasi (PCA) Data Historis")
    fig_pca = plot_pca_clusters(pca_data_historis)
    pca_plot_area = st.pyplot(fig_pca)

with col_results_display:
    st.subheader("Tinjauan Segmen Historis Superstore")
    st.markdown("Rata-rata metrik untuk setiap kluster:")
    
    st.dataframe(
        CLUSTER_PROFILES_DF.style.format("{:,.2f}").set_caption("Nilai Rata-rata Kluster (Total USD)"), 
        use_container_width=True
    ) 
    st.info("Tekan tombol 'Prediksi Segmen Pelanggan' di sidebar untuk menguji pelanggan baru!")

if predict_button:
    # 1. Persiapan Data
    new_customer_data = pd.DataFrame([input_values])
    new_customer_data = new_customer_data[SPENDING_COLS] 
    X_new = new_customer_data.values

    # 2. Standarisasi dan Prediksi
    # Scaling manual karena scaler.transform(X_new) akan mengkonversi ke array 2D
    new_customer_scaled = (X_new - scaler.mean_) / scaler.scale_
    prediction = model_logistic.predict(new_customer_scaled)
    prediction_proba = model_logistic.predict_proba(new_customer_scaled)[0]
    predicted_cluster = prediction[0]
    new_point_pca = pca.transform(new_customer_scaled)

    # 3. Update Visualisasi dan Tampilkan Hasil
    with col_pca_display:
        st.subheader("Peta Segmentasi Pelanggan (PCA) - Hasil Prediksi")
        fig_updated = plot_pca_clusters(pca_data_historis, new_point_pca, predicted_cluster)
        pca_plot_area.pyplot(fig_updated) 

    with col_results_display:
        st.subheader("3. Hasil Prediksi")
        
        if predicted_cluster == 0:
            st.success(f"Segmen Diprediksi: **Kluster {predicted_cluster} (HIGH-VALUE)** 🎉")
            profile_key_str = '0'
        else: # predicted_cluster == 1
            st.info(f"Segmen Diprediksi: **Kluster {predicted_cluster} (STANDARD)** 📈")
            profile_key_str = '1'

        st.markdown(f"**Perbandingan dengan Pola Khas Kluster {predicted_cluster}:**")
        
        st.dataframe(
            CLUSTER_PROFILES_DF[[profile_key_str]].rename(columns={profile_key_str: "Rata-rata Kluster"}).style.format("{:,.2f}"),
            use_container_width=True
        )

        st.markdown("##### Probabilitas Keyakinan Model:")
        proba_df = pd.DataFrame(
            {'Kluster': [f'Kluster {i}' for i in range(K_FIXED)], 
             'Probabilitas': prediction_proba.round(4)
            }
        ).sort_values(by='Probabilitas', ascending=False)
        st.dataframe(proba_df.style.format({'Probabilitas': "{:.2%}"}), hide_index=True)
        
        st.markdown("---")
        st.markdown("##### Rekomendasi Tindakan Bisnis (Actionable Insight):")
        
        if predicted_cluster == 0:
            st.success("Tindakan: Pertahankan dan Kembangkan. Fokus pada **Program Loyalitas Eksklusif, Penawaran Produk Baru (Terutama Teknologi), dan Peningkatan Layanan Premium**.")
        else:
            st.warning("Tindakan: Tingkatkan Nilai. Fokus pada **Penjualan Silang (Cross-Selling) produk dengan marjin tinggi (Technology/Office Supplies)** dan insentif untuk meningkatkan frekuensi pembelian.")
