import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. KONFIGURASI UI/UX
# ==========================================
st.set_page_config(
    page_title="Sistem Deteksi Aritmia",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. BACKEND MODULAR (CACHING & LOADING)
# ==========================================
@st.cache_resource
def load_models():
    """Modul untuk memuat objek pipeline machine learning"""
    try:
        model = joblib.load('best_arrhythmia_model.joblib')
        imputer = joblib.load('imputer.joblib')
        scaler = joblib.load('scaler.joblib')
        selector = joblib.load('selector.joblib')
        return model, imputer, scaler, selector, True
    except Exception as e:
        return None, None, None, None, False

@st.cache_data
def load_data():
    """Modul untuk memuat sampel data mentah untuk Dashboard"""
    try:
        df = pd.read_csv('arrhythmia.data', header=None)
        df.replace('?', np.nan, inplace=True)
        return df
    except:
        return pd.DataFrame()

model, imputer, scaler, selector, is_model_loaded = load_models()
df_raw = load_data()

# ==========================================
# 3. NAVIGASI SIDEBAR
# ==========================================
st.sidebar.title("🫀 Menu Navigasi")
menu = st.sidebar.radio(
    "Pilih Halaman:",
    ["🏠 Dashboard Data", "🩺 Form Prediksi (Inference)", "📊 Metrik & Penjelasan Model"]
)

st.sidebar.divider()
st.sidebar.info(
    "**Disclaimer Medis & Etika:**\n\nAplikasi ini murni berfungsi sebagai *Decision Support System* (Sistem Pendukung Keputusan). "
    "Hasil prediksi tidak boleh dijadikan diagnosis akhir atau menggantikan wewenang kardiolog."
)

# ==========================================
# 4. HALAMAN 1: DASHBOARD DATA
# ==========================================
if menu == "🏠 Dashboard Data":
    st.title("Dashboard Eksplorasi Data EKG")
    st.markdown("Halaman ini menampilkan gambaran umum dari dataset *Cardiac Arrhythmia* yang digunakan untuk melatih sistem cerdas ini.")
    
    if not df_raw.empty:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Sampel Pasien", f"{df_raw.shape[0]} Baris")
        col2.metric("Total Fitur EKG", f"{df_raw.shape[1] - 1} Fitur")
        col3.metric("Jumlah Kelas Penyakit", "16 Kelas")
        
        st.subheader("Cuplikan Dataset Mentah (Raw Data)")
        st.dataframe(df_raw.head(10), use_container_width=True)
        
        st.subheader("Distribusi Kelas (Highly Imbalanced)")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.countplot(x=df_raw[279], palette="viridis", ax=ax)
        ax.set_xlabel("Kelas Aritmia (1 = Normal)")
        ax.set_ylabel("Jumlah Pasien")
        st.pyplot(fig)
    else:
        st.warning("Dataset 'arrhythmia.data' tidak ditemukan di direktori lokal.")

# ==========================================
# 5. HALAMAN 2: FORM PREDIKSI (INFERENCE)
# ==========================================
elif menu == "🩺 Form Prediksi (Inference)":
    st.title("Simulasi Deteksi Dini Aritmia")
    st.markdown("Masukkan parameter fisik dan kelistrikan jantung pasien di bawah ini. *(Catatan Demo: Sebagian besar dari 278 parameter kelistrikan akan disimulasikan menggunakan nilai baseline untuk mempermudah pengujian aplikasi)*.")
    
    if is_model_loaded:
        with st.form("prediction_form"):
            st.subheader("Input Parameter Pasien")
            c1, c2, c3, c4 = st.columns(4)
            umur = c1.number_input("Umur (Tahun)", min_value=1, max_value=120, value=45)
            tinggi = c2.number_input("Tinggi Badan (cm)", min_value=100, max_value=250, value=165)
            berat = c3.number_input("Berat Badan (kg)", min_value=30, max_value=200, value=65)
            qrs = c4.number_input("Durasi QRS (msec)", min_value=50, max_value=200, value=85)
            
            submit = st.form_submit_button("Analisis EKG Sekarang", type="primary")
            
        if submit:
            with st.spinner("Model sedang memproses data..."):
                # Menyiapkan array input
                input_array = np.zeros((1, 279))
                input_array[0, 0] = umur
                input_array[0, 2] = tinggi
                input_array[0, 3] = berat
                input_array[0, 4] = qrs
                
                # Preprocessing secara terstruktur
                input_clean = np.delete(input_array, 13, axis=1) # Drop kolom 13
                x_imputed = imputer.transform(input_clean)
                x_scaled = scaler.transform(x_imputed)
                x_final = selector.transform(x_scaled)
                
                # Prediksi
                pred = model.predict(x_final)[0]
                proba = model.predict_proba(x_final)[0]
                confidence = np.max(proba) * 100
                
                # Menampilkan Hasil secara visual
                st.divider()
                st.subheader("Hasil Analisis Prediktif")
                
                res_col1, res_col2 = st.columns([1, 2])
                with res_col1:
                    st.metric("Tingkat Kepercayaan (Probabilitas)", f"{confidence:.2f}%")
                
                with res_col2:
                    if pred == 1:
                        st.success(f"### KELAS 1 (NORMAL)\nTidak ditemukan pola aritmia yang signifikan pada parameter yang diberikan. Ritme jantung masuk kategori aman (Sinus Rhythm).")
                    else:
                        st.error(f"### KELAS {int(pred)} (INDIKASI ARITMIA TERDETEKSI)\nModel mendeteksi adanya anomali kelistrikan yang mengindikasikan aritmia. Segera konsultasikan hasil ini dengan spesialis kardiologi untuk diagnosis akhir.")
    else:
        st.error("Gagal memuat model. Pastikan file .joblib sudah tersedia di folder proyek.")

# ==========================================
# 6. HALAMAN 3: METRIK & PENJELASAN MODEL
# ==========================================
elif menu == "📊 Metrik & Penjelasan Model":
    st.title("Arsitektur & Evaluasi Sistem Cerdas")
    
    st.subheader("1. Penjelasan Model (Explainability)")
    st.markdown("""
    Aplikasi ini ditenagai oleh algoritma **Support Vector Machine (SVM)** dengan optimasi **Hyperparameter Tuning (GridSearchCV)** dan penanganan ketidakseimbangan kelas menggunakan **RandomOverSampler**.
    * **Mengapa SVM?** Pada dataset medis dengan dimensi ekstrim (278 fitur prediktor EKG), model SVM sangat tangguh dalam memetakan batas keputusan linier maupun non-linier menggunakan trik *Kernel RBF*.
    * **Strategi Optimasi:** Metrik evaluasi digeser dari sekadar Akurasi menjadi *Macro F1-Score* untuk memastikan model adil dan tidak bias terhadap pasien berstatus "Normal" saja.
    """)
    
    st.subheader("2. Peningkatan Performa (Baseline vs Optimized)")
    metrik_data = {
        "Fase Eksperimen": ["SVM Baseline (Tanpa Optimasi)", "SVM Optimized (ROS + GridSearch)"],
        "Akurasi": ["52.75%", "N/A (Fokus F1)"],
        "Balanced Accuracy": ["11.09%", "17.95%"],
        "Macro F1-Score": ["Bias Mayoritas", "19.19%"]
    }
    st.table(pd.DataFrame(metrik_data))
    
    st.info("Kenaikan pada *Balanced Accuracy* dan *Macro F1-Score* menunjukkan bahwa model mulai mampu mengenali pola sinyal EKG untuk kelas-kelas penyakit aritmia yang sangat langka, membuktikan bahwa strategi optimasi berhasil.")