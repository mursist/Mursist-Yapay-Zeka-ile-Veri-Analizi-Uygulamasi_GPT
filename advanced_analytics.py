import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import openai

# openai.api_key = "YOUR_API_KEY"

def profitability_analysis(data=None):
    st.subheader("Karlılık Analizi")

    if data is None:
        data = pd.DataFrame({
            'ürün': ['A', 'B', 'C', 'D', 'E'],
            'maliyet': [50, 40, 60, 30, 45],
            'fiyat': [80, 60, 90, 50, 70],
            'satış_adedi': [100, 120, 60, 200, 150]
        })

    data['toplam_gelir'] = data['fiyat'] * data['satış_adedi']
    data['toplam_maliyet'] = data['maliyet'] * data['satış_adedi']
    data['kar'] = data['toplam_gelir'] - data['toplam_maliyet']

    st.dataframe(data)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='ürün', y='kar', data=data, palette='crest', ax=ax)
    ax.set_title("Ürün Bazlı Kar Dağılımı")
    st.pyplot(fig)

    if st.checkbox("Bu kar dağılımına göre GPT'den yorum al"):
        description = data.to_string(index=False)

        prompt = f"Aşağıdaki ürün kar verilerine göre hangi ürünler en karlıdır ve neden olabilir? Ayrıca, stratejik önerilerde bulun.\n\n{description}"

        # response = openai.ChatCompletion.create(
        #     model="gpt-4",
        #     messages=[{"role": "user", "content": prompt}]
        # )
        # gpt_comment = response['choices'][0]['message']['content']

        gpt_comment = (
            "En yüksek kâr, C ürünü tarafından sağlanmıştır. Bunun nedeni yüksek fiyatına rağmen satış sayısının düşük olmamasıdır. "
            "Stratejik olarak C ürününün pazarlama faaliyetleri artırılabilir. Ayrıca, D ürününün düşük kârına rağmen yüksek satış adedi dikkat çekicidir. "
            "Bu ürün için fiyat optimizasyonu yapılabilir."
        )

        st.markdown("**Yapay Zeka Yorumu**")
        st.info(gpt_comment)


def trend_analysis(data=None):
    st.subheader("Trend Analizi")

    if data is None:
        data = pd.DataFrame({
            'tarih': pd.date_range(start='2023-01-01', periods=12, freq='M'),
            'satış': [1200, 1300, 1250, 1400, 1600, 1700, 1650, 1800, 1750, 1900, 1950, 2000]
        })

    st.line_chart(data.set_index('tarih'))

    if st.checkbox("Bu satış trendine göre GPT'den analiz al"):
        sales_list = ", ".join(map(str, data['satış'].tolist()))
        prompt = f"Bu aylık satış değerlerine bakarak genel trendi yorumla ve olası nedenleri değerlendir: {sales_list}"

        # response = openai.ChatCompletion.create(
        #     model="gpt-4",
        #     messages=[{"role": "user", "content": prompt}]
        # )
        # trend_output = response['choices'][0]['message']['content']

        trend_output = (
            "Verilere bakıldığında yıl boyunca artan bir satış trendi gözlenmektedir. Bu artış muhtemelen sezonsal etkiler, başarılı kampanyalar "
            "ve müşteri sadakati ile ilgilidir. Özellikle yaz ve yıl sonuna doğru satışlardaki artış belirgindir. Son çeyrek stratejileri bu başarıyı desteklemiş olabilir."
        )

        st.markdown("**Yapay Zeka Analizi**")
        st.success(trend_output)
