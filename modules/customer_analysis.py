import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import openai

# openai.api_key = "YOUR_API_KEY"

import veri_analizi as va

def rfm_analysis():
    st.subheader("RFM (Recency, Frequency, Monetary) Analizi")

    df = va.create_customer_data()
    df['Monetary'] = df['avg_purchase_value'] * df['purchase_frequency']
    df['Recency'] = 30 - (df['purchase_frequency'] % 30)
    df['Frequency'] = df['purchase_frequency']

    rfm = df[['Recency', 'Frequency', 'Monetary']]

    st.dataframe(rfm.head())

    fig, ax = plt.subplots()
    sns.scatterplot(data=rfm, x='Recency', y='Monetary', size='Frequency', ax=ax, legend=False)
    ax.set_title("RFM Müşteri Segmentasyonu")
    st.pyplot(fig)

    if st.checkbox("GPT'den müşteri segmentleri için öneri al"):
        sample_data = rfm.describe().to_string()

        prompt = f"RFM analizi özet bilgileri aşağıdaki gibidir. Buna göre hangi müşteri segmentleri dikkat çekmektedir ve nasıl bir strateji izlenmelidir?\n\n{sample_data}"

        # GPT API çağrısı (opsiyonel)
        # response = openai.ChatCompletion.create(
        #     model="gpt-4",
        #     messages=[{"role": "user", "content": prompt}]
        # )
        # gpt_response = response['choices'][0]['message']['content']

        gpt_response = (
            "RFM analizi sonuçlarına göre yüksek Monetary ve düşük Recency değerine sahip müşteriler sadık ve değerli müşterilerdir. "
            "Bu gruba özel kampanyalar sunmak, sadakati pekiştirir. Düşük Frequency ve düşük Monetary grubundaki müşteriler ise yeniden kazanım stratejileriyle hedeflenebilir."
        )

        st.markdown("**Yapay Zeka Analizi**")
        st.info(gpt_response)


def sentiment_analysis():
    st.subheader("Müşteri Yorumlarında Duygu Analizi")

    comments = [
        "Ürün çok kaliteli ve hızlı teslim edildi, teşekkür ederim.",
        "Beklediğimden kötü çıktı, memnun kalmadım.",
        "Fiyatına göre gayet iyi bir ürün.",
        "Kargo geç geldi ama ürün güzel.",
        "İade süreci çok zordu, müşteri hizmetleri yetersizdi."
    ]

    df = pd.DataFrame({'Yorum': comments})
    st.dataframe(df)

    if st.checkbox("Yorumlara GPT ile duygu analizi yap"):
        yorumlar = "\\n".join(comments)

        prompt = f"Aşağıdaki müşteri yorumlarının her birini inceleyerek olumlu, olumsuz veya nötr şeklinde duygu etiketlemesi yap:\n\n{yorumlar}"

        # GPT API çağrısı (opsiyonel)
        # response = openai.ChatCompletion.create(
        #     model="gpt-4",
        #     messages=[{"role": "user", "content": prompt}]
        # )
        # gpt_sentiment = response['choices'][0]['message']['content']

        gpt_sentiment = (
            "1. Olumlu\n"
            "2. Olumsuz\n"
            "3. Olumlu\n"
            "4. Nötr\n"
            "5. Olumsuz"
        )

        st.markdown("**Yapay Zeka Duygu Etiketi**")
        st.code(gpt_sentiment)
