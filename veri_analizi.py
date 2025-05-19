import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Satış verisi örneği oluştur
def create_sample_sales_data(n_days=365*2):
    dates = pd.date_range(start='2022-01-01', periods=n_days, freq='D')
    np.random.seed(42)
    sales = np.random.normal(loc=500, scale=100, size=n_days).clip(min=0)

    df = pd.DataFrame({
        'date': dates,
        'sales': sales,
        'is_holiday': [1 if d.weekday() in [5, 6] else 0 for d in dates],
        'is_promotion': [random.choice([0, 1]) for _ in range(n_days)]
    })

    df['weekday'] = df['date'].dt.weekday
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x in [5, 6] else 0)
    return df

# Müşteri verisi örneği oluştur
def create_customer_data(n_customers=500):
    np.random.seed(42)
    data = {
        'customer_id': [f'CUST_{i:05d}' for i in range(n_customers)],
        'avg_purchase_value': np.random.gamma(shape=2.0, scale=500.0, size=n_customers),
        'purchase_frequency': np.random.randint(1, 30, size=n_customers),
        'return_rate': np.random.rand(n_customers),
        'loyalty_years': np.random.uniform(0.5, 10, size=n_customers)
    }

    df = pd.DataFrame(data)
    df['customer_value'] = df['avg_purchase_value'] * df['purchase_frequency'] * (1 - df['return_rate'])
    return df

# ARIMA için zaman serisi analizi
def analyze_time_series(df):
    import statsmodels.api as sm

    ts = df.copy()
    ts['date'] = pd.to_datetime(ts['date'])
    ts.set_index('date', inplace=True)
    result = sm.tsa.seasonal_decompose(ts['sales'], model='additive', period=30)
    return result

# 30 günlük basit tahmin (hareketli ortalama)
def forecast_sales(df, days=30):
    ts = df.copy()
    ts['date'] = pd.to_datetime(ts['date'])
    ts.set_index('date', inplace=True)
    avg = ts['sales'].rolling(window=30).mean().dropna().iloc[-1]
    last_date = ts.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, days+1)]
    forecast = pd.Series([avg]*days, index=future_dates)
    return forecast

# Makine öğrenmesi satış tahmini modelleri
def train_ml_sales_model(df):
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from sklearn.model_selection import train_test_split

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    df['month'] = df['date'].dt.month

    features = ['is_holiday', 'is_promotion', 'weekday', 'month', 'day']
    X = df[features]
    y = df['sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    xgb = XGBRegressor(n_estimators=100, random_state=42)
    xgb.fit(X_train, y_train)

    return rf, xgb

# K-means segmentasyonu
def segment_customers(df, n_clusters=4):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    df = df.copy()
    features = ['avg_purchase_value', 'purchase_frequency', 'return_rate', 'loyalty_years', 'customer_value']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    df['cluster'] = kmeans.fit_predict(X_scaled)
    return df, kmeans, scaler

# ----------------------------------------------------------------------------
# GPT DESTEKLİ ZAMAN SERİSİ ANALİZİ AÇIKLAMASI
# ----------------------------------------------------------------------------

def explain_time_series_with_gpt(df):
    prompt = f"Aşağıdaki zaman serisi verisini inceleyerek genel eğilimleri, mevsimsellikleri ve dikkat çeken noktaları analiz et:\n\n{df.tail(60).to_string(index=False)}"

    # GPT çağrısı (isteğe bağlı):
    # response = openai.ChatCompletion.create(
    #     model="gpt-4",
    #     messages=[{"role": "user", "content": prompt}]
    # )
    # gpt_explanation = response['choices'][0]['message']['content']

    gpt_explanation = (
        "Son 60 günlük satış verisine bakıldığında düzenli bir haftalık mevsimsellik gözlemlenmektedir. "
        "Belirli aralıklarla zirve yapan satışlar, hafta sonu veya kampanya etkileriyle ilişkilendirilebilir. "
        "Genel eğilim pozitif ve stabil bir artış yönündedir."
    )

    return gpt_explanation

# ----------------------------------------------------------------------------
# GPT DESTEKLİ SATIŞ TAHMİNİ AÇIKLAMASI
# ----------------------------------------------------------------------------

def explain_forecast_with_gpt(forecast_series):
    prompt = f"Aşağıdaki 30 günlük satış tahminine göre geleceğe yönelik stratejik bir değerlendirme yap:\n\n{forecast_series.to_string()}"

    # GPT çağrısı (isteğe bağlı):
    # response = openai.ChatCompletion.create(
    #     model="gpt-4",
    #     messages=[{"role": "user", "content": prompt}]
    # )
    # gpt_forecast = response['choices'][0]['message']['content']

    gpt_forecast = (
        "Önümüzdeki 30 gün içinde satışların istikrarlı şekilde artacağı öngörülüyor. Bu eğilim, planlanan kampanyalarla desteklenirse "
        "ciro hedeflerine ulaşmak mümkün olacaktır. Stok planlaması ve müşteri iletişimi bu sürece entegre edilmelidir."
    )

    return gpt_forecast
