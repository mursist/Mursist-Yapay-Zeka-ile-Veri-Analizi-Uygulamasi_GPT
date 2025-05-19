import streamlit as st
# import openai

# openai.api_key = "YOUR_API_KEY"

from dashboard import add_dashboard
from sales_analysis import seasonal_analysis
from customer_analysis import rfm_analysis, sentiment_analysis
from advanced_analytics import profitability_analysis, trend_analysis
from product_recommendation import product_recommendation

st.set_page_config(page_title="Veri Analitiği ve Yapay Zeka Paneli", layout="wide")

st.title("📊 Veri Analitiği ve Yapay Zeka Destekli Karar Paneli")

menu = st.sidebar.selectbox("Modül Seçin", [
    "Genel Dashboard",
    "Satış Analizi",
    "Müşteri Analizi",
    "Gelişmiş Analitik",
    "Ürün Öneri Motoru"
])

if menu == "Genel Dashboard":
    add_dashboard()

elif menu == "Satış Analizi":
    seasonal_analysis()

elif menu == "Müşteri Analizi":
    st.subheader("📌 Müşteri Analizi Seçenekleri")
    tab1, tab2 = st.tabs(["RFM Segmentasyonu", "Yorum Duygu Analizi"])
    with tab1:
        rfm_analysis()
    with tab2:
        sentiment_analysis()

elif menu == "Gelişmiş Analitik":
    st.subheader("📌 Karlılık ve Trend Analizi")
    tab1, tab2 = st.tabs(["Karlılık Analizi", "Trend Analizi"])
    with tab1:
        profitability_analysis()
    with tab2:
        trend_analysis()

elif menu == "Ürün Öneri Motoru":
    product_recommendation()

# Not: Gerçek GPT kullanımı için openai importlarını aktif edin ve API anahtarınızı girin.
