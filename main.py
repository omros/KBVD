# from turtle import clear
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Harga Komoditas",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data.csv')
    df = df.melt(id_vars=['No', 'Komoditas (Rp)'], var_name='Tanggal', value_name='Harga')
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], format='%d/%m/%Y')
    df['Harga'] = df['Harga'].replace('-', np.nan).str.replace(',', '').astype(float)
    return df

df = load_data()

# Prepare data for clustering
def prepare_clustering_data(df):
    # Pivot the data to get commodities as rows and dates as columns
    df_pivot = df.pivot(index='Komoditas (Rp)', 
                       columns='Tanggal', 
                       values='Harga')
    
    # Calculate features for clustering
    clustering_features = pd.DataFrame(index=df_pivot.index)
    clustering_features['mean_price'] = df_pivot.mean(axis=1)
    clustering_features['std_price'] = df_pivot.std(axis=1)
    clustering_features['price_range'] = df_pivot.max(axis=1) - df_pivot.min(axis=1)
    
    return clustering_features, df_pivot

# Perform clustering
@st.cache_data
def perform_clustering(features, n_clusters=3):
    # Scale the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features_scaled)
    
    return clusters

# Sidebar
st.sidebar.write("Anggota Kelompok :")
st.sidebar.write("10122081 - Fajar Gustiana")
st.sidebar.write("10122092 - Muhlas Putra Siswaji")
st.sidebar.write("10122097 - Ryan Bachtiar")

st.sidebar.header("Filter Data")
selected_commodities = st.sidebar.multiselect(
    "Pilih Komoditas",
    options=df['Komoditas (Rp)'].unique(),
    default=["Beras", "Daging Ayam", "Minyak Goreng"]
)

date_range = st.sidebar.date_input(
    "Rentang Tanggal",
    value=[df['Tanggal'].min(), df['Tanggal'].max()],
    min_value=df['Tanggal'].min(),
    max_value=df['Tanggal'].max()
)

# Filter data
filtered_df = df[
    (df['Komoditas (Rp)'].isin(selected_commodities)) &
    (df['Tanggal'].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])))
]

# Layout utama
title = "📊 Analisis Harga di Jawa Barat dari Komoditas"
if selected_commodities:
    title += f" - {', '.join(selected_commodities)}"
st.title(title)

# Create main tabs
tab_visualisasi, tab_clustering = st.tabs(["📈 Visualisasi Data", "🔍 Analisis Clustering"])
with tab_visualisasi:
    st.markdown("---")
    
    # Chart 1: Line Chart Tren Harga
    st.subheader("1. Tren Harga")
    
    # Buat objek fig1 menggunakan px.line
    fig1 = px.line(
        filtered_df,
        x='Tanggal',
        y='Harga',
        color='Komoditas (Rp)',
        labels={'Harga': 'Harga (Rp)', 'Tanggal': 'Tanggal'},
        height=500
    )
    
    # Cari harga tertinggi dan terendah untuk setiap komoditas
    for commodity in selected_commodities:
        commodity_data = filtered_df[filtered_df['Komoditas (Rp)'] == commodity]
        max_price = commodity_data['Harga'].max()
        min_price = commodity_data['Harga'].min()
        
        # Temukan tanggal untuk harga tertinggi dan terendah
        max_date = commodity_data.loc[commodity_data['Harga'].idxmax(), 'Tanggal']
        min_date = commodity_data.loc[commodity_data['Harga'].idxmin(), 'Tanggal']
        
        # Tambahkan titik untuk harga tertinggi
        fig1.add_trace(go.Scatter(
            x=[max_date],
            y=[max_price],
            mode='markers',
            marker=dict(color='red', size=10),
            name='Harga Tertinggi',  # Nama legend yang sama untuk semua titik tertinggi
            showlegend=True if commodity == selected_commodities[0] else False  # Hanya tampilkan sekali di legend
        ))
        
        # Tambahkan titik untuk harga terendah
        fig1.add_trace(go.Scatter(
            x=[min_date],
            y=[min_price],
            mode='markers',
            marker=dict(color='green', size=10),
            name='Harga Terendah',  # Nama legend yang sama untuk semua titik terendah
            showlegend=True if commodity == selected_commodities[0] else False  # Hanya tampilkan sekali di legend
        ))
    
    st.plotly_chart(fig1, use_container_width=True)
    
    st.markdown("""
    **Penjelasan:**
    - Menunjukkan pergerakan harga komoditas pilihan
    - Garis warna berbeda mewakili komoditas yang berbeda
    - Titik merah menunjukkan harga tertinggi, titik hijau menunjukkan harga terendah
    - Dapat melihat pola kenaikan/penurunan harga secara temporal
    - Interaktif: zoom, hover untuk nilai detail, toggle legend
    """)
    st.markdown("---")
    # Chart 2: Bar Chart Rata-rata Harga
    st.subheader("2. Perbandingan Harga Rata-rata")
    avg_prices = filtered_df.groupby('Komoditas (Rp)')['Harga'].mean().reset_index()
    fig2 = px.bar(
        avg_prices,
        x='Komoditas (Rp)',
        y='Harga',
        color='Komoditas (Rp)',
        labels={'Harga': 'Rata-rata Harga (Rp)'},
        height=500
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("""
    **Penjelasan:**
    - Menampilkan perbandingan harga rata-rata komoditas
    - Tinggi batang menunjukkan nilai rata-rata dalam periode terpilih
    - Warna berbeda untuk memudahkan identifikasi komoditas
    - Berguna untuk membandingkan tingkat harga antar komoditas
    """)
    st.markdown("---")

    # Chart 3: Area Chart Distribusi Harga
    st.subheader("3. Distribusi Harga Kumulatif")
    fig3 = px.area(
        filtered_df,
        x='Tanggal',
        y='Harga',
        color='Komoditas (Rp)',
        labels={'Harga': 'Total Kumulatif (Rp)'},
        height=500
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("""
    **Penjelasan:**
    - Menunjukkan akumulasi harga komoditas sepanjang waktu
    - Area berwarna menunjukkan kontribusi masing-masing komoditas
    - Berguna untuk melihat proporsi total belanja komoditas
    - Dapat mengidentifikasi periode kenaikan biaya kumulatif
    """)



with tab_clustering:
    st.markdown("---")
    st.subheader("Analisis Clustering Komoditas")

    # Prepare data and perform clustering
    features, df_pivot = prepare_clustering_data(df)
    clusters = perform_clustering(features)
    features['Cluster'] = clusters

    # Add cluster information
    cluster_info = pd.DataFrame({
        'Komoditas': features.index,
        'Cluster': features['Cluster'],
        'Rata-rata Harga': features['mean_price'],
        'Volatilitas Harga': features['std_price']
    })

    # Display cluster information
    st.write("#### Pengelompokan Komoditas Berdasarkan Pola Harga")

    # Create subtabs for different clustering visualizations
    subtab1, subtab2, subtab3 = st.tabs(["Scatter Plot Clustering", "Statistik Cluster", "Daftar Komoditas per Cluster"])

    with subtab1:
        # Scatter plot
        fig_scatter = px.scatter(features, 
                               x='mean_price', 
                               y='std_price',
                               color='Cluster',
                               title='Hasil Clustering: Rata-rata Harga vs Volatilitas Harga',
                               labels={'mean_price': 'Rata-rata Harga (Rp)',
                                      'std_price': 'Volatilitas Harga (Std Dev)',
                                      'Cluster': 'Cluster'},
                               hover_data={'mean_price': ':.2f',
                                         'std_price': ':.2f'},
                               text=features.index)
        fig_scatter.update_traces(textposition='top center')
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with subtab2:
        # Display statistics for each cluster
        for cluster in sorted(features['Cluster'].unique()):
            cluster_stats = features[features['Cluster'] == cluster].agg({
                'mean_price': ['mean', 'min', 'max'],
                'std_price': ['mean', 'min', 'max']
            })
            
            st.write(f"**Cluster {cluster}**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rata-rata Harga", f"Rp {cluster_stats['mean_price']['mean']:,.2f}")
                st.metric("Volatilitas Rata-rata", f"Rp {cluster_stats['std_price']['mean']:,.2f}")
            with col2:
                st.metric("Rentang Harga", 
                         f"Rp {cluster_stats['mean_price']['min']:,.2f} - {cluster_stats['mean_price']['max']:,.2f}")
            st.markdown("---")

    with subtab3:
        # Display commodities in each cluster
        for cluster in sorted(features['Cluster'].unique()):
            st.write(f"**Cluster {cluster}**")
            cluster_commodities = features[features['Cluster'] == cluster].index.tolist()
            for commodity in cluster_commodities:
                st.write(f"- {commodity}")
            st.markdown("---")

    st.markdown("""
    **Penjelasan Clustering:**
    - **Cluster 0**: Komoditas dengan harga stabil dan relatif rendah
    - **Cluster 1**: Komoditas dengan volatilitas harga tinggi
    - **Cluster 2**: Komoditas dengan volatilitas harga menengah

    **Manfaat Analisis:**
    - Membantu mengidentifikasi kelompok komoditas berdasarkan pola harga
    - Memudahkan pemantauan komoditas dengan volatilitas tinggi
    - Mendukung pengambilan keputusan dalam manajemen stok dan harga
    """)

# Menambahkan footer
st.markdown("---")
st.caption("Aplikasi Analisis Harga Komoditas - 2024")
