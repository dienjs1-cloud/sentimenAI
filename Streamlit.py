import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
from collections import Counter
import string

# Import model dan metrik
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer

# --- KONFIGURASI HALAMAN ---x
st.set_page_config(
    layout="wide",
    page_title="Analisis Sentimen Dampak AI",
    page_icon="🎨"
)

# --- STYLING KUSTOM DENGAN CSS ---
st.markdown("""
<style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 16px; 
    }
    .main > div {
        background-color: #F0F2F6;
    }
    /* Style untuk menu tab di tengah dan lebih besar */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        display: flex;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px 8px 0px 0px;
        padding: 10px 15px;
        color: #64748B;
        border-bottom: 3px solid transparent;
        font-size: 1.2em; /* Ukuran font diperbesar */
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF;
        color: #1E90FF;
        font-weight: bold;
        border-bottom: 3px solid #1E90FF;
    }
    h1 {
        color: #2C3E50;
        font-weight: 600;
        font-size: 2.5rem; 
        margin-bottom: 0.2rem;
    }
    h3 {
        color: #1E90FF;
        border-bottom: 2px solid #F0F2F6;
        padding-bottom: 8px;
        font-size: 1.75rem;
    }
</style>
""", unsafe_allow_html=True)


# --- FUNGSI-FUNGSI HELPER ---
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"❌ Gagal memuat file: {e}")
        return None

@st.cache_resource
def train_model(_X_train, _y_train):
    model = GaussianNB()
    model.fit(_X_train, _y_train)
    return model

def create_distribution_chart(df_series, title):
    fig = px.bar(df_series, x=df_series.index, y=df_series.values, title=title,
                 color=df_series.index, color_discrete_map={'Positif': '#20C997', 'Negatif': '#FA8072'}, text_auto=True)
    fig.update_layout(showlegend=False, title_x=0.5, plot_bgcolor='rgba(0,0,0,0)')
    return fig

def create_performance_chart(report_df):
    df_plot = report_df.T.reset_index().melt(
        id_vars='index', var_name='Tipe Data', value_name='Skor'
    ).rename(columns={'index': 'Metrik'})
    
    fig = px.bar(
        df_plot, 
        x='Metrik', 
        y='Skor', 
        color='Tipe Data', 
        barmode='group', 
        text_auto='.3f',
        color_discrete_map={'Data Latih (Train)': '#87CEEB', 'Data Uji (Test)': '#4682B4'},
        title='Perbandingan Performa Model'
    )
    
    fig.update_layout(
        title_x=0.30,  # Baris ini yang memposisikan judul di tengah
        title_font_size=24, 
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(gridcolor='#E0E0E0'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_confusion_matrix_chart(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(cm, text_auto=True, labels=dict(x="Prediksi Model", y="Label Sebenarnya", color="Jumlah"),
                    x=labels, y=labels, color_continuous_scale='Blues', title='Confusion Matrix')
    fig.update_layout(title_x=0.30, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def generate_word_visuals(texts, color, title, custom_stop_words, wc_colormap):
    st.markdown(f"<h3 style='text-align: center;'>{title}</h3>", unsafe_allow_html=True)
    corpus = " ".join(texts.astype(str).dropna().tolist()).lower()
    if not corpus.strip():
        st.write("Tidak ada teks untuk dianalisis.")
        return
    all_words = [word.strip(string.punctuation) for word in corpus.split()]
    filtered_words = [w for w in all_words if w and w not in custom_stop_words]
    common_words = Counter(filtered_words).most_common(10)
    if not common_words:
        st.write("Tidak ada kata yang tersisa setelah difilter.")
        return
    df_words = pd.DataFrame(common_words, columns=['Kata', 'Frekuensi'])
    fig_words = px.bar(df_words, x='Frekuensi', y='Kata', orientation='h', title="10 Kata Teratas", color_discrete_sequence=[color])
    fig_words.update_layout(yaxis=dict(autorange="reversed"), title_x=0.5, plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_words, use_container_width=True)
    filtered_corpus = " ".join(filtered_words)
    if filtered_corpus:
        wc = WordCloud(width=800, height=400, background_color='white', colormap=wc_colormap).generate(filtered_corpus)
        st.image(wc.to_array(), use_container_width=True, caption="Wordcloud")

# --- APLIKASI UTAMA ---
def main():
    with st.sidebar:
        st.header("⚙️ Konfigurasi Visualisasi")
        input_file = st.file_uploader("Upload file data final (.csv)", type=["csv"], key="main_uploader")
        st.divider()
        st.header("🚫 Filter Kata")
        custom_stop_words_input = st.text_area(
            "Masukkan stopwords (pisahkan koma)",
            "amp, dan, yg, di, dengan, untuk, pada, dari, itu, ini",
            height=200 
        )

    st.title("🎨 Dashboard Analisis Sentimen Dampak AI di Dunia Kerja")
    st.markdown('<p style="font-size: 20px; color: #606770;">Klasifikasi Dampak Kecerdasan Buatan di Dunia Kerja Menggunakan Naive Bayes</p>', unsafe_allow_html=True)
    st.divider()

    tab_scrape, tab_prep, tab_dist, tab_words, tab_perf, tab_eval = st.tabs([
        "📄 **Data Scraping**", "🧹 **Data Preprocessing**", "📊 **Distribusi Data**", 
        "📝 **Analisis Kata**", "📈 **Performa Model**", "🔍 **Evaluasi Rinci**"
    ])

    with tab_scrape:
        st.subheader("Tampilan Data Mentah (Hasil Scraping)")
        uploaded_file_raw = st.file_uploader("Upload file scraping (.csv)", type=["csv"], key="raw_data")
        if uploaded_file_raw:
            df_raw = load_data(uploaded_file_raw)
            if df_raw is not None:
                st.dataframe(df_raw)

    with tab_prep:
        st.subheader("Tampilan Data Bersih (Hasil Preprocessing)")
        uploaded_file_cleaned = st.file_uploader("Upload file preprocessing (.csv)", type=["csv"], key="cleaned_data")
        if uploaded_file_cleaned:
            df_cleaned = load_data(uploaded_file_cleaned)
            if df_cleaned is not None:
                st.dataframe(df_cleaned)

    if input_file is not None:
        df = load_data(input_file)
        if df is not None:
            required_cols = ['label', 'teks']
            if not all(col in df.columns for col in required_cols):
                st.error(f"Error: Untuk visualisasi, pastikan file CSV memiliki kolom {required_cols}.")
                st.stop()
            
            emb_cols = [col for col in df.columns if col.startswith('embedding_')]
            if not emb_cols:
                st.error("Error: Kolom embeddings (dengan prefix 'embedding_') tidak ditemukan.")
                st.stop()

            X = df[emb_cols].values
            y = df['label'].values
            label_names = {0: "Negatif", 1: "Positif"}
            indices = np.arange(len(df))
            
            X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
                X, y, indices, test_size=0.2, stratify=y, random_state=42
            )
            
            model = train_model(X_train, y_train)
            y_pred_test = model.predict(X_test)

            with tab_dist:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.subheader("Ringkasan Dataset") 
                    st.metric("Total Data", len(df))
                    st.metric("Data Latih (80%)", len(X_train))
                    st.metric("Data Uji (20%)", len(X_test))
                with col2:
                    st.subheader("Distribusi Sentimen")
                    vc = pd.Series(y).value_counts().rename(index=label_names)
                    fig_dist = create_distribution_chart(vc, "Distribusi Label")
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                # --- KODE YANG DITAMBAHKAN ---
                st.divider()
                st.subheader("Tampilan Data Hasil Pembagian")

                # Membuat dataframe untuk train dan test menggunakan indeks yang sudah ada dari train_test_split
                train_df = df.iloc[idx_train]
                test_df = df.iloc[idx_test]

                # Membuat sub-tab untuk menampilkan dataframe
                tab_train_view, tab_test_view = st.tabs(["🛠️ Data Latih (Train)", "🔎 Data Uji (Test)"])
                
                with tab_train_view:
                    # Menampilkan data latih dengan index yang direset agar rapi
                    st.dataframe(train_df.reset_index(drop=True), use_container_width=True)
                
                with tab_test_view:
                    # Menampilkan data uji dengan index yang direset agar rapi
                    st.dataframe(test_df.reset_index(drop=True), use_container_width=True)
                # --- AKHIR KODE YANG DITAMBAHKAN ---
            
            with tab_words:
                stop_words_list = [word.strip().lower() for word in custom_stop_words_input.split(',')]
                col_pos, col_neg = st.columns(2)
                with col_pos:
                    generate_word_visuals(df[df['label']==1]['teks'], '#20C997', "👍 Sentimen Positif", stop_words_list, 'Greens')
                with col_neg:
                    generate_word_visuals(df[df['label']==0]['teks'], '#FA8072', "👎 Sentimen Negatif", stop_words_list, 'Reds')

            with tab_perf:
                st.subheader("Ringkasan Metrik Model")
                def get_metrics(y_true, y_pred):
                    return {'Akurasi': accuracy_score(y_true, y_pred), 'Presisi': precision_score(y_true, y_pred, average='weighted', zero_division=0), 'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0), 'F1-Score': f1_score(y_true, y_pred, average='weighted', zero_division=0)}
                
                y_pred_train = model.predict(X_train)
                train_metrics = get_metrics(y_train, y_pred_train)
                test_metrics = get_metrics(y_test, y_pred_test)
                
                st.info("Metrik ini menunjukkan seberapa baik model dapat menggeneralisasi pada data baru.", icon="💡")
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("Akurasi", f"{test_metrics['Akurasi']:.2%}")
                m_col2.metric("Presisi", f"{test_metrics['Presisi']:.2%}")
                m_col3.metric("Recall", f"{test_metrics['Recall']:.2%}")
                m_col4.metric("F1-Score", f"{test_metrics['F1-Score']:.2%}")
                
                st.divider()
                report_df = pd.DataFrame({'Data Latih (Train)': train_metrics, 'Data Uji (Test)': test_metrics})
                fig_perf = create_performance_chart(report_df.round(4))
                st.plotly_chart(fig_perf, use_container_width=True)

            with tab_eval:
                st.subheader("Evaluasi Rinci Model")
                col1, col2 = st.columns(2)
                with col1:
                    fig_cm = create_confusion_matrix_chart(y_test, y_pred_test, list(label_names.values()))
                    st.plotly_chart(fig_cm, use_container_width=True)
                
                with col2:
                    report = classification_report(y_test, y_pred_test, target_names=label_names.values(), output_dict=True, zero_division=0)
                    report_df = pd.DataFrame(report).transpose().round(4)
                    st.dataframe(report_df)
                    
                st.divider()
                st.subheader("Analisis Kesalahan Klasifikasi")
                test_df_reset = df.iloc[idx_test].copy().reset_index(drop=True)
                eval_table = test_df_reset[['teks']].copy()
                eval_table['Label Asli'] = pd.Series(y_test).map(label_names)
                eval_table['Prediksi Model'] = pd.Series(y_pred_test).map(label_names)
                eval_table['Status'] = np.where(eval_table['Label Asli'] == eval_table['Prediksi Model'], '✅ Benar', '❌ Salah')
                
                mislassified_df = eval_table[eval_table['Status'] == '❌ Salah']
                st.warning(f"Ditemukan **{len(mislassified_df)} kesalahan prediksi** dari total {len(y_test)} data uji.", icon="⚠️")
                
                with st.expander(f"Tampilkan {len(mislassified_df)} Data yang Salah Prediksi"):
                    st.dataframe(mislassified_df, use_container_width=True)
                    csv_export = mislassified_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Data Salah Prediksi (.csv)", csv_export, "hasil_prediksi_salah_nb.csv", "text/csv")
    else:
        with tab_dist:
            st.info("Silakan upload file data final di sidebar untuk melihat visualisasi model.")

if __name__ == "__main__":
    main()