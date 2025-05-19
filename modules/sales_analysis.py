import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
# import openai

# openai.api_key = "YOUR_API_KEY"

# Ana dizini Python yolu ekle - veri_analizi.py dosyasını import etmek için gerekli
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import veri_analizi as va
except:
    st.error("veri_analizi.py dosyası bulunamadı! Ana dizinde bulunduğundan emin olun.")

def seasonal_analysis(sales_df=None):
    if sales_df is None:
        try:
            sales_df = va.create_sample_sales_data()
        except:
            st.error("Örnek veri oluşturulamadı.")
            return

    st.subheader("Mevsimsel ve Dönemsel Analiz")

    tab_daily, tab_weekly, tab_monthly, tab_yearly = st.tabs(["Günlük", "Haftalık", "Aylık", "Yıllık"])

    with tab_weekly:
        st.write("Haftanın günlerine göre satış dağılımı")
        if 'weekday' in sales_df.columns:
            weekday_counts = sales_df.groupby('weekday')['sales'].mean()
            fig, ax = plt.subplots(figsize=(10, 6))
            weekdays = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
            ax.bar(weekdays[:len(weekday_counts)], weekday_counts, color='royalblue')
            ax.set_title("Haftanın Günlerine Göre Ortalama Satışlar")
            st.pyplot(fig)

    with tab_monthly:
        st.write("Aylara göre satış dağılımı")
        if 'month' in sales_df.columns and 'date' in sales_df.columns:
            sales_df['date'] = pd.to_datetime(sales_df['date'])
            monthly_sales = sales_df.groupby(sales_df['date'].dt.month)['sales'].mean()
            month_names = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", 
                           "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(range(1, 13), [monthly_sales.get(i, 0) for i in range(1, 13)], color='seagreen')
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(month_names)
            ax.set_title("Aylara Göre Ortalama Satışlar")
            plt.xticks(rotation=45)
            st.pyplot(fig)

    with tab_yearly:
        st.write("Yıllara göre satış trendi")
        if 'year' in sales_df.columns and 'date' in sales_df.columns:
            sales_df['date'] = pd.to_datetime(sales_df['date'])
            yearly_sales = sales_df.groupby(sales_df['date'].dt.year)['sales'].sum()
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(yearly_sales.index, yearly_sales.values, marker='o', linewidth=2, color='purple')
            ax.set_title("Yıllık Toplam Satışlar")
            st.pyplot(fig)

    with tab_daily:
        st.write("Günlük satış dağılımı")
        if 'date' in sales_df.columns:
            sales_df_ts = sales_df.copy()
            sales_df_ts['date'] = pd.to_datetime(sales_df_ts['date'])
            sales_df_ts.set_index('date', inplace=True)
            rolling_mean = sales_df_ts['sales'].rolling(window=30).mean()
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(sales_df_ts.index, sales_df_ts['sales'], alpha=0.5, color='gray', label='Günlük Satışlar')
            ax.plot(rolling_mean.index, rolling_mean, color='red', linewidth=2, label='30 Günlük Ortalama')
            ax.set_title("Günlük Satışlar ve Hareketli Ortalama")
            ax.legend()
            st.pyplot(fig)

    if st.checkbox("GPT'den bu mevsimsel verilerle ilgili analiz al"):
        recent_sales = sales_df.tail(60)[['date', 'sales']].to_string(index=False)

        prompt = f"Aşağıdaki satış verilerine göre mevsimsel desenleri ve olası iş stratejilerini yorumla:\n\n{recent_sales}"

        # GPT API çağrısı (opsiyonel):
        # response = openai.ChatCompletion.create(
        #     model="gpt-4",
        #     messages=[{"role": "user", "content": prompt}]
        # )
        # gpt_output = response['choices'][0]['message']['content']

        # Örnek çıktı:
        gpt_output = (
            "Verilen verilere göre yaz aylarında satışlarda artış eğilimi gözlemlenmektedir. Tatil sezonu, promosyonlar ve kampanyalar etkili olabilir. "
            "Bu dönemlerde stok ve lojistik planlaması yapılmalı, reklam bütçeleri bu aylara kaydırılabilir."
        )

        st.markdown("**Yapay Zeka Değerlendirmesi**")
        st.success(gpt_output)
