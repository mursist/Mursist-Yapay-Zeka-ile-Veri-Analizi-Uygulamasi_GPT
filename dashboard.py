import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# import openai

# openai.api_key = "YOUR_API_KEY"

import veri_analizi as va

def add_dashboard():
    st.subheader("Genel Dashboard")

    df = va.create_sample_sales_data()
    st.write("Veri Kümesi Özeti")
    st.dataframe(df.head())

    col1, col2, col3 = st.columns(3)

    with col1:
        toplam_satis = df['sales'].sum()
        st.metric("Toplam Satış", f"{toplam_satis:,.0f} ₺")

    with col2:
        ortalama_satis = df['sales'].mean()
        st.metric("Ortalama Satış", f"{ortalama_satis:,.2f} ₺")

    with col3:
        en_yuksek = df['sales'].max()
        st.metric("En Yüksek Günlük Satış", f"{en_yuksek:,.0f} ₺")

    fig, ax = plt.subplots(figsize=(10, 5))
    df_grouped = df.groupby('month')['sales'].mean()
    ax.plot(df_grouped.index, df_grouped.values, marker='o', linestyle='-')
    ax.set_title("Aylık Ortalama Satışlar")
    ax.set_xlabel("Ay")
    ax.set_ylabel("Satış")
    st.pyplot(fig)

    if st.checkbox("Yapay zekadan genel değerlendirme al"):
        summary_stats = df.describe().to_string()

        prompt = f"Aşağıdaki satış verisi özetine göre satış performansı hakkında genel bir değerlendirme yap:\n\n{summary_stats}"

        # GPT çağrısı (opsiyonel)
        # response = openai.ChatCompletion.create(
        #     model="gpt-4",
        #     messages=[{"role": "user", "content": prompt}]
        # )
        # gpt_response = response['choices'][0]['message']['content']

        # Örnek çıktı
        gpt_response = (
            "Satış verilerine göre genel olarak istikrarlı bir performans gözlemleniyor. Ortalama satış tutarı belirli bir seviyede sabitlenmiş durumda. "
            "En yüksek satışlar bazı aylarda belirgin olarak artmakta; bu dönemler özel kampanya, sezonluk etki ya da tatil dönemlerine denk gelmiş olabilir."
        )

        st.markdown("**Yapay Zeka Değerlendirmesi**")
        st.success(gpt_response)
