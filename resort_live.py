import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import pandas as pd
import joblib

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


df = pd.read_csv("hotel_bookings_preprocessed.csv")
X_classification = pd.read_csv("X_classification.csv")
X_regression = pd.read_csv("X_regression.csv")
df.head()
# Convert 'is_canceled' to boolean
df['is_canceled'] = df['is_canceled'].astype(bool)
# convert the reservation_status_date into date
df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])
#convert agent column from int to object
df['agent'] = df['agent'].astype(object)

rf_model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

st.sidebar.title("🏨 Booking ML Project")
page = st.sidebar.radio("Go to", [
    "📈 Business Problem" ,
    "📊 EDA",
    "🛠️ Feature Engineering",
    "❌ Cancellation Prediction",
    "💰 ADR Prediction",
    "💡 Insights and Recommendations",
    "📑 Appendix"
])

def plotly_time_series(df, date_col, value_col=None, agg='count', title='', ylabel='', color='blue'):
    """
    Plots a time series line chart using Plotly for a given value aggregated by month/year.
    """
    df[date_col] = pd.to_datetime(df[date_col])
    df['year_month'] = df[date_col].dt.to_period('M').astype(str)

    if agg == 'count':
        data = df.groupby('year_month').size()
    elif agg == 'sum':
        data = df.groupby('year_month')[value_col].sum()
    elif agg == 'mean':
        data = df.groupby('year_month')[value_col].mean()
    else:
        raise ValueError("agg must be 'count', 'sum', or 'mean'")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data.values,
        mode='lines+markers',
        line=dict(color=color),
        marker=dict(size=8),
        name=title
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Month",
        yaxis_title=ylabel,
        xaxis_tickangle=-45,
        template="plotly_white"
    )
    st.plotly_chart(fig)
    
#bar chart function
def plotly_bar_chart(df, x_col, y_col=None, agg='count', title='', xlabel='', ylabel='', color='blue'):
    """
    Plots a bar chart using Plotly for categorical comparisons.

    Parameters:
        df: pandas.DataFrame
        x_col: str - categorical column for x-axis
        y_col: str or None - column to aggregate (for 'sum' or 'mean')
        agg: str - aggregation type: 'count', 'sum', 'mean'
        title: str - chart title
        xlabel: str - x-axis label
        ylabel: str - y-axis label
        color: str - bar color
    """
    if agg == 'count':
        data = df[x_col].value_counts(ascending=False)
        x = data.index
        y = data.values
    elif agg == 'sum':
        data = df.groupby(x_col)[y_col].sum().sort_values(ascending=False)
        x = data.index
        y = data.values
    elif agg == 'mean':
        data = df.groupby(x_col)[y_col].mean().sort_values(ascending=False)
        x = data.index
        y = data.values
    else:
        raise ValueError("agg must be 'count', 'sum', or 'mean'")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x,
        y=y,
        marker_color=color,
        name=title
    ))
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template="plotly_white"
    )
    st.plotly_chart(fig)

# Stacked Bar Chart for Cancellations
def plotly_stacked_canceled_chart(df, category_col, title, xlabel):
    """
    Plots a stacked bar chart for canceled vs. non-canceled bookings across a categorical variable.

    Parameters:
        df: pandas.DataFrame
        category_col: str - categorical column for x-axis
        title: str - chart title
        xlabel: str - x-axis label
    """
    # For top 10 countries, filter first
    if category_col == "country":
        top_countries = df['country'].value_counts().head(10).index
        data = df[df['country'].isin(top_countries)]
    else:
        data = df

    canceled_data = data.groupby([category_col, 'is_canceled']).size().unstack(fill_value=0)
    # Sort by total bookings (descending)
    canceled_data = canceled_data.sort_values(by=[False, True], ascending=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=canceled_data.index,
        y=canceled_data[False],
        name='Not Canceled',
        marker_color='royalblue'
    ))
    fig.add_trace(go.Bar(
        x=canceled_data.index,
        y=canceled_data[True],
        name='Canceled',
        marker_color='tomato'
    ))
    fig.update_layout(
        barmode='stack',
        title=title,
        xaxis_title=xlabel,
        yaxis_title='Number of Bookings',
        template="plotly_white"
    )
    st.plotly_chart(fig)

# Box Plot for Numerical Column by Cancellation Status
def plotly_box_by_cancel(df, column, title, ylabel):
    """
    Plots a box plot for a numerical column grouped by is_canceled status.

    Parameters:
        df: pandas.DataFrame
        column: str - numerical column to plot
        title: str - chart title
        ylabel: str - y-axis label
    """
    fig = go.Figure()
    for canceled, color in zip([False, True], ['royalblue', 'tomato']):
        fig.add_trace(go.Box(
            y=df[df['is_canceled'] == canceled][column],
            name='Not Canceled' if not canceled else 'Canceled',
            marker_color=color
        ))
    fig.update_layout(
        title=title,
        yaxis_title=ylabel,
        template="plotly_white"
    )
    st.plotly_chart(fig)

# Feature Importance Chart
def plotly_feature_importance_streamlit(model, features, title ="Feature Importance", num=15):
    feature_imp = pd.DataFrame({
        'Feature': features.columns,
        'Importance': model.feature_importances_
    })
    feature_imp = feature_imp.sort_values(by="Importance", ascending=False).head(num)
    fig = px.bar(
        feature_imp,
        x="Importance",
        y="Feature",
        orientation="h",
        title=title,
        color="Importance",
        color_continuous_scale="Blues",
        width=900,
        height=600
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig
#İş Problemi
if page == "📈 Business Problem":

    st.image("hotel.png", use_container_width=True)

    st.markdown(
        """
        <div style="text-align:center;">
            <h3 style="color:#2980b9;">Veri Seti Hikayesi</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("""
    Veri setimiz, ABD'de yer alan bir otel zincirine bağlı iki farklı otel türüne (şehir ve resort otel) ait yaklaşık 120.000 rezervasyon kaydını içermektedir. Veri setinde rezervasyonun yapıldığı tarihten, konaklama süresine, misafir sayısından, oda tipine ve rezervasyonun iptal edilip edilmediğine dair zengin bilgiler bulunmaktadır.
    """)

    st.markdown(
        """
        <div style="text-align:center;">
            <h6 style="color:#2471a3;">Veri Setinden Örnek Kayıtlar</h4>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.dataframe(df.head(5))

    st.markdown(
        """
        <div style="text-align:center;">
            <h3 style="color:#b9770e;">İş Problemi</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("""
    Bir otel zinciri, iki temel operasyonel zorlukla karşı karşıyadır: gelir optimizasyonu ve rezervasyon iptalleri. Yüksek iptal oranları, doluluk oranlarında belirsizliğe ve önemli gelir kayıplarına yol açarken, yanlış fiyatlandırma (ADR - Ortalama Günlük Fiyat) stratejileri potansiyel geliri maksimize etmede başarısız olmaktadır. Bu durum, hem operasyonel verimsizliklere hem de finansal kayıplara neden olmaktadır.
    """)

    st.markdown("""
    Bu projenin temel amacı, otel yöneticilerinin bu zorlukların üstesinden gelmesine yardımcı olacak veri odaklı bir karar destek sistemi oluşturmaktır. Sistem, aşağıdaki temel iş problemlerine çözüm sunmayı hedefler:

    1. **İptal Riskini Önceden Tahmin Etme**: Hangi rezervasyonların iptal edilme olasılığının yüksek olduğunu önceden tespit etmek. Bu sayede otel yönetimi, riskli rezervasyonlar için proaktif önlemler alabilir (örneğin, teyit için özel indirimler sunma, stratejik olarak fazla rezervasyon yapma vb.).

    2. **Gelir ve Fiyatlandırmayı Optimize Etme**: Rezervasyonun özelliklerine (örneğin, talep yoğunluğu, müşteri segmenti, konaklama süresi) bağlı olarak Ortalama Günlük Fiyat'ı (ADR) tahmin ederek dinamik bir fiyatlandırma stratejisi geliştirmek.
    """)

    st.markdown(
        """
        <div style="text-align:center; margin-top:40px;">
            <h3 style="color:#2980b9;">Veri Bilimi Takımı</h3>
            <div style="display:flex; justify-content:center; gap:40px; font-size:18px; margin-top:10px;">
                <span>Esin Saygın</span>
                <span>Özge Erdemli</span>
                <span>Onur Çelebi</span>
                <span>Samet Girgin</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
#Data Görselleştirme
elif page == "📊 EDA":
    st.header("📊 Keşifçi Veri Analizi")

    # First chart option dropdown
    st.markdown("#### Otel Performans Genel Görünümü")
    chart_option = st.selectbox(
        "Select Bar/Line Chart",
        [
            "Bookings by Hotel Type",
            "Bookings per Month",
            "Bookings by Arrival Month",
            "Average ADR per Month",
            "Total Revenue by Market Segment",
            "Cancellations per Month",
            "Total Guests per Month",
            "Total Revenue per Month"
        ]
    )

    if chart_option == "Bookings per Month":
        plotly_time_series(df, 'reservation_status_date', None, agg='count', title='Bookings per Month', ylabel='Number of Bookings', color='blue')
    elif chart_option == "Total Guests per Month":
        # Prepare both series
        # All bookings (including canceled)
        guests_all = df.groupby(df['reservation_status_date'].dt.to_period('M'))['total_guests'].sum()
        # Only non-canceled bookings (canceled set to zero)
        df_guests = df.copy()
        df_guests.loc[df_guests['is_canceled'] == True, 'total_guests'] = 0
        guests_non_canceled = df_guests.groupby(df_guests['reservation_status_date'].dt.to_period('M'))['total_guests'].sum()

        # Plot both lines in one chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=guests_all.index.astype(str),
            y=guests_all.values,
            mode='lines+markers',
            name='All Bookings',
            line=dict(color='purple')
        ))
        fig.add_trace(go.Scatter(
            x=guests_non_canceled.index.astype(str),
            y=guests_non_canceled.values,
            mode='lines+markers',
            name='Excluding Canceled',
            line=dict(color='orange')
        ))
        fig.update_layout(
            title='Total Guests per Month',
            xaxis_title="Month",
            yaxis_title="Total Guests",
            xaxis_tickangle=-45,
            template="plotly_white"
        )
        st.plotly_chart(fig)
    elif chart_option == "Cancellations per Month":
        plotly_time_series(df[df['is_canceled'] == True], 'reservation_status_date', None, agg='count', title='Cancellations per Month', ylabel='Number of Cancellations', color='red')
    elif chart_option == "Average ADR per Month":
        plotly_time_series(df, 'reservation_status_date', 'adr', agg='mean', title='Average ADR per Month', ylabel='Average ADR', color='green')
    elif chart_option == "Total Revenue per Month":
        # Prepare both series
        df_1 = pd.read_csv("hotel_bookings_visualization.csv")
        df_1['reservation_status_date'] = pd.to_datetime(df_1['reservation_status_date'])
        df_1['year_month'] = df_1['reservation_status_date'].dt.to_period('M').astype(str)
        df_1 = df_1[(df_1['year_month'] >= '2015-06') & (df_1['year_month'] <= '2017-08')]
        revenue_all = df_1.groupby(df_1['reservation_status_date'].dt.to_period('M'))['total_revenue'].sum()
        df_revenue = df_1.copy()
        df_revenue.loc[df_revenue['is_canceled'] == True, 'total_revenue'] = 0
        revenue_non_canceled = df_revenue.groupby(df_revenue['reservation_status_date'].dt.to_period('M'))['total_revenue'].sum()

        print("All Bookings Revenue:", revenue_all)
        print("Excluding Canceled Revenue:", revenue_non_canceled)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=revenue_all.index.astype(str),
            y=revenue_all.values,
            mode='lines+markers',
            name='All Bookings',
            line=dict(color='red')
        ))
        fig.add_trace(go.Scatter(
            x=revenue_non_canceled.index.astype(str),
            y=revenue_non_canceled.values,
            mode='lines+markers',
            name='Excluding Canceled',
            line=dict(color='green')
        ))
        fig.update_layout(
            title='Total Revenue per Month',
            xaxis_title="Month",
            yaxis_title="Total Revenue",
            xaxis_tickangle=-45,
            template="plotly_white"
        )
        st.plotly_chart(fig)
    elif chart_option == "Bookings by Hotel Type":
        plotly_bar_chart(df, 'hotel', agg='count', title='Bookings by Hotel Type', xlabel='Hotel Type', ylabel='Number of Bookings', color='royalblue')
    elif chart_option == "Total Revenue by Market Segment":
        # Only sum total_revenue for non-canceled bookings
        df_non_canceled = df[df['is_canceled'] == False]
        plotly_bar_chart(
            df_non_canceled,
            'market_segment',
            y_col='total_revenue',
            agg='sum',
            title='Total Revenue by Market Segment (Excluding Canceled)',
            xlabel='Market Segment',
            ylabel='Total Revenue',
            color='orange'
        )
    elif chart_option == "Bookings by Arrival Month":
        plotly_bar_chart(df, 'arrival_date_month', agg='count', title='Bookings by Arrival Month', xlabel='Arrival Month', ylabel='Number of Bookings', color='lightseagreen')

    # Second chart option dropdown for canceled vs. non-canceled charts
    st.markdown("#### Rezervasyon İptalleri Analizi")
    stacked_chart_option = st.selectbox(
        "Select Category for Stacked Bar Chart / Distribution Chart",
        [
            "Hotel Type",
            "Arrival Month",
            "Arrival Year",
            "Market Segment",
            "Country (Top 10)",
            "Lead Time Distribution",
            "ADR Distribution",
            "Total Stay Nights Distribution",
            "Total Guests Distribution"
        ]
    )

    if stacked_chart_option == "Hotel Type":
        plotly_stacked_canceled_chart(df, "hotel", "Canceled vs. Non-Canceled Bookings by Hotel Type", "Hotel Type")
    elif stacked_chart_option == "Arrival Month":
        plotly_stacked_canceled_chart(df, "arrival_date_month", "Canceled vs. Non-Canceled Bookings by Arrival Month", "Arrival Month")
    elif stacked_chart_option == "Arrival Year":
        plotly_stacked_canceled_chart(df, "arrival_date_year", "Canceled vs. Non-Canceled Bookings by Arrival Year", "Arrival Year")
    elif stacked_chart_option == "Market Segment":
        plotly_stacked_canceled_chart(df, "market_segment", "Canceled vs. Non-Canceled Bookings by Market Segment", "Market Segment")
    elif stacked_chart_option == "Country (Top 10)":
        plotly_stacked_canceled_chart(df, "country", "Canceled vs. Non-Canceled Bookings by Country (Top 10)", "Country")
    elif stacked_chart_option == "Lead Time Distribution":
        plotly_box_by_cancel(df, "lead_time", "Lead Time Distribution by Cancellation Status", "Lead Time")
    elif stacked_chart_option == "ADR Distribution":
        plotly_box_by_cancel(df, "adr", "ADR Distribution by Cancellation Status", "ADR")
    elif stacked_chart_option == "Total Stay Nights Distribution":
        plotly_box_by_cancel(df, "total_stay_nights", "Total Stay Nights Distribution by Cancellation Status", "Total Stay Nights")
    elif stacked_chart_option == "Total Guests Distribution":
        plotly_box_by_cancel(df, "total_guests", "Total Guests Distribution by Cancellation Status", "Total Guests")
    st.markdown("#### Ziyaretçi Profili Analizi")
    guest_chart_option = st.selectbox(
        "Select Guest Profile Chart",
        [
            "Guest Profiles by Month",
            "Bookings by Country",
            "Cancellations by Country"
        ]
    )

    if guest_chart_option == "Guest Profiles by Month":
        #st.subheader("Guest Profile by Month (Has Children or Babies)")
        df['year_month'] = df['reservation_status_date'].dt.to_period('M').astype(str)
        guest_profile = df.groupby(['year_month', 'has_children']).size().unstack(fill_value=0)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=guest_profile.index,
            y=guest_profile[0],
            name='No Children',
            marker_color='royalblue'
        ))
        fig.add_trace(go.Bar(
            x=guest_profile.index,
            y=guest_profile[1],
            name='Has Children',
            marker_color='orange'
        ))
        fig.update_layout(
            barmode='stack',
            title='Guest Profile by Month',
            xaxis_title='Month',
            yaxis_title='Number of Bookings',
            template="plotly_white"
        )
        st.plotly_chart(fig)
    elif guest_chart_option == "Bookings by Country":
        country_counts = df['country'].value_counts().reset_index()
        country_counts.columns = ['country', 'bookings']
        fig = px.choropleth(
            country_counts,
            locations="country",
            color="bookings",
            locationmode="ISO-3",
            color_continuous_scale="Oranges",
            title="Number of Bookings by Country"
        )
        st.plotly_chart(fig)

    elif guest_chart_option == "Cancellations by Country":
        canceled_df = df[df['is_canceled'] == True]
        country_canceled = canceled_df['country'].value_counts().reset_index()
        country_canceled.columns = ['country', 'cancellations']
        total_bookings = df['country'].value_counts().reset_index()
        total_bookings.columns = ['country', 'total_bookings']
        # Merge to calculate cancellation rate
        merged = pd.merge(country_canceled, total_bookings, on='country', how='left')
        merged['Cancellation Rate %'] = ((merged['cancellations'] / merged['total_bookings']) * 100).round(0)  # percentage

        fig = px.choropleth(
            merged,
            locations="country",
            color="Cancellation Rate %",
            locationmode="ISO-3",
            color_continuous_scale="Reds",
            title="Cancellation Rate by Country"
        )
        st.plotly_chart(fig)
#ADR Prediction      
elif page == "💰 ADR Prediction":
    st.header("💰 Ortalama Günlük Ücret Tahmini")

    from xgboost import XGBRegressor
    final_xgb_model = XGBRegressor()
    final_xgb_model.load_model("final_xgb_model.json")
 
    st.markdown("Aşağıdaki formu doldurarak ADR tahmini alabilirsiniz.")

    #final_xgb_model = joblib.load("final_xgb_model.pkl")

    # Sol sütun
    col1, col2 = st.columns(2)

    with col1:
        hotel_type = st.selectbox("Otel Tipi", options=["Resort", "City"])
        hotel_1 = 1 if hotel_type == "Resort" else 0
        lead_time = st.number_input("Lead Time (gün)", min_value=0, value=10)
        stays_in_weekend_nights = st.number_input("Hafta Sonu Gece Sayısı", min_value=0, value=1)
        stays_in_week_nights = st.number_input("Hafta İçi Gece Sayısı", min_value=0, value=2)
        New_total_guests = st.number_input("Toplam Misafir Sayısı", min_value=1, value=2)
        INFLATION = st.number_input("Enflasyon (%)", value=2.5)
        GDP = st.number_input("Kişi Başı GSYİH ($)", value=18000.0)
        New_has_children = st.selectbox("Çocuk veya Bebek Var mı?", options=["Hayır", "Evet"])
        New_has_children_val = 1 if New_has_children == "Evet" else 0

    with col2:
        season = st.selectbox("Mevsim", options=["İlkbahar", "Yaz", "Kış", "Sonbahar"])
        New_season_Spring = 1 if season == "İlkbahar" else 0
        New_season_Summer = 1 if season == "Yaz" else 0
        New_season_Autumn = 1 if season == "Sonbahar" else 0
        New_season_Winter = 1 if season == "Kış" else 0

        meal = st.selectbox("Yemek Tipi", options=["A (Yarım Pansiyon)", "B (Kahvaltı)", "C (Tam Pansiyon)"])
        New_meal_class_A = 1 if meal == "A (Yarım Pansiyon)" else 0
        New_meal_class_B = 1 if meal == "B (Kahvaltı)" else 0
        New_meal_class_C = 1 if meal == "C (Tam Pansiyon)" else 0

        market_segment = st.selectbox("Pazar Segmenti", options=["A", "B", "C", "D"])
        New_market_segment_grouped_A = 1 if market_segment == "A" else 0
        New_market_segment_grouped_B = 1 if market_segment == "B" else 0
        New_market_segment_grouped_C = 1 if market_segment == "C" else 0
        New_market_segment_grouped_D = 1 if market_segment == "D" else 0

        room_class = st.selectbox("Oda Sınıfı", options=["A", "B", "C", "D"])
        New_room_class_A = 1 if room_class == "A" else 0
        New_room_class_B = 1 if room_class == "B" else 0
        New_room_class_C = 1 if room_class == "C" else 0
        New_room_class_D = 1 if room_class == "D" else 0

        deposit_type = st.selectbox("Depozito Tipi", options=["İade Yok", "İadeli"])
        deposit_type_Non_Refund = 1 if deposit_type == "İade Yok" else 0
        deposit_type_Refundable = 1 if deposit_type == "İadeli" else 0

        agent_class = st.selectbox("Acente Sınıfı", options=["E", "D", "C", "B", "A"])
        New_agent_class_E = 1 if agent_class == "E" else 0
        New_agent_class_D = 1 if agent_class == "D" else 0
        New_agent_class_C = 1 if agent_class == "C" else 0
        New_agent_class_B = 1 if agent_class == "B" else 0
        New_agent_class_A = 1 if agent_class == "A" else 0

        New_customer_type_1 = st.selectbox("Müşteri Tipi (Grup/Sözleşmeli mi?)", options=["Hayır", "Evet"])
        New_customer_type_1_val = 1 if New_customer_type_1 == "Evet" else 0

    # Toplam Konaklama Gecesi otomatik hesaplanıyor
    New_total_stay = stays_in_weekend_nights + stays_in_week_nights

    input_dict = {
        "lead_time": lead_time,
        "stays_in_weekend_nights": stays_in_weekend_nights,
        "stays_in_week_nights": stays_in_week_nights,
        "New_total_guests": New_total_guests,
        "New_total_stay": New_total_stay,
        "INFLATION": INFLATION,
        "GDP": GDP,
        "New_has_children": New_has_children_val,
        "hotel_1": hotel_1,
        "New_season_Spring": New_season_Spring,
        "New_season_Summer": New_season_Summer,
        "New_season_Winter": New_season_Winter,
        "New_meal_class_B": New_meal_class_B,
        "New_meal_class_C": New_meal_class_C,
        "New_market_segment_grouped_B": New_market_segment_grouped_B,
        "New_market_segment_grouped_C": New_market_segment_grouped_C,
        "New_market_segment_grouped_D": New_market_segment_grouped_D,
        "New_room_class_B": New_room_class_B,
        "New_room_class_C": New_room_class_C,
        "New_room_class_D": New_room_class_D,
        "deposit_type_Non Refund": deposit_type_Non_Refund,
        "deposit_type_Refundable": deposit_type_Refundable,
        "New_agent_class_D": New_agent_class_D,
        "New_agent_class_C": New_agent_class_C,
        "New_agent_class_B": New_agent_class_B,
        "New_agent_class_A": New_agent_class_A,
        "New_customer_type_1": New_customer_type_1_val
    }

    if st.button("ADR Tahmini Yap"):
        input_df = pd.DataFrame([input_dict])

        # Modelin fit edildiği X tablosunun feature isimlerini yükle
        X_columns = joblib.load("adr_feature_names.pkl")  # Bu dosya X.columns.tolist() ile kaydedilmiş olmalı

        # Eksik olan feature'ları 0 ile doldur, sıralamayı garanti et
        for col in X_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[X_columns]  # Sıralamayı garanti et

        scaler_adr = joblib.load("scaler_adr.pkl")
        input_scaled = scaler_adr.transform(input_df)
        prediction = final_xgb_model.predict(input_scaled)
        st.success(f"Tahmin edilen ADR: {prediction[0]:.2f}")   
    st.markdown("""
    <b>Kullanılan Model:</b> <span style="color:#27ae60;">XGBoost Regressor</span><br>
    <b>Model, rezervasyonun ortalama günlük fiyatını (ADR) tahmin etmek için eğitilmiştir.</b><br><br>
    """, unsafe_allow_html=True)
    
    # Özellik önem grafiğini göster
    #st.markdown("Model Özellik Önem Grafiği")
    fig = plotly_feature_importance_streamlit(final_xgb_model, X_regression, title="XGBoost Regressor Feature Importance")
    st.plotly_chart(fig, use_container_width=True)

     
#İptal tahmini
elif page == "❌ Cancellation Prediction":
    st.header("❌ Rezervasyon İptal Tahmini")
    
    st.markdown("""
    ### 💻 Model Oluşturma: Veriden Tahmine
    Veriler titizlikle hazırlandıktan ve özellik mühendisliği yapıldıktan sonra, bir sonraki adım tahmin modelini oluşturmaktı. Buradaki amaç, bir rezervasyonun iptal edilme olasılığını doğru bir şekilde tahmin edebilecek bir sınıflandırıcıyı eğitmektir.

    ### 🚀 Temel Modeller ve Performans Değerlendirmesi
    Nihai bir modele karar vermeden önce, en iyi performansı gösteren algoritmayı bulmak için çeşitli yaygın makine öğrenimi sınıflandırma algoritmaları değerlendirildi. Her modelin performansı, bu tür dengesiz sınıflandırma problemlerinde anahtar bir ölçüt olan ROC AUC kullanılarak ölçüldü.

    **Test Edilen Temel Modeller:** Lojistik Regresyon, KNN, SVM, Karar Ağacı, Rastgele Orman, AdaBoost, Gradyan Artırma (Gradient Boosting), XGBoost ve LightGBM.

    **Model Seçimi:** Çapraz doğrulama (cross-validation) sonuçlarına göre, Gradyan Artırma Makinesi (GBM) üstün performans gösterdi.

    ---
    ### ⚙️ Optimal Performans için Hiperparametre Ayarı
    GBM modelinin başlangıç versiyonu, daha da iyi sonuçlar elde etmek için ince ayar yapıldı. Hiperparametre ayarı adı verilen bu süreç, modelin performansını en üst düzeye çıkaran kombinasyonu bulmak için parametrelerinin (learning_rate, max_depth, n_estimators, subsample gibi) farklı konfigürasyonlarını sistematik olarak test etmeyi içerir.

    **Ayar Tekniği:** Belirlenen bir parametre ızgarasını (grid) kapsamlı bir şekilde arayan bir Grid Search kullanıldı.

    **Nihai Model:** Optimize edilmiş GBM modeli, performans metriklerinde (doğruluk, F1-skoru ve ROC AUC) önemli bir artış göstererek etkinliğini doğruladı.

    ---
    ### 📊 Özellik Önem Derecesi: En Çok Ne Önemli?
    Modelin hangi özelliklere güvendiğini anlamak, yorumlanabilirlik açısından çok önemlidir. İptalleri tahmin etmede en etkili faktörleri belirlemek için modelin özellik önem derecesi analizi yapıldı.

    Bu görselleştirme, total_guests, lead_time, adr gibi özelliklerin ve yeni oluşturulan TrustedAgent ve Country_Risk değişkenlerinin, otel rezervasyon iptallerini tahmin etmek için en güçlü göstergeler arasında yer aldığını ortaya koymaktadır.

    ---
    ### 💾 Dağıtım ve Ölçeklenebilirlik
    Pratik uygulama için, nihai optimize edilmiş GBM modeli ve veri ölçekleyici (gbm_model.pkl ve scaler.pkl) bir dosyaya kaydedildi. Bu, modelin yeniden eğitilmesine gerek kalmadan yeni, gerçek zamanlı rezervasyon verileri üzerinde tahmin yapmak için kolayca yüklenip kullanılabilmesini sağlar.
    """, unsafe_allow_html=True)

    
    X_columns = [
        "lead_time", "is_repeated_guest", "adr", "INFLATION_CHG", "CSMR_SENT", "TrustedAgent", "PartnerAgent",
        "total_guests", "has_children", "total_stay_nights", "staying_on_weekends", "Country_Risk",
        "is_board", "is_resort", "has_special_requests",
        "season_Spring", "season_Summer", "season_Winter",
        "reserved_room_D", "reserved_room_Other"
    ]

    display_names = {
        "lead_time": "Lead Time (Days)",
        "is_repeated_guest": "Repeated Guest",
        "adr": "Average Daily Rate (ADR)",
        "INFLATION_CHG": "Inflation Change",
        "CSMR_SENT": "Consumer Sentiment",
        "TrustedAgent": "Trusted Agent",
        "PartnerAgent": "Partner Agent",
        "total_guests": "Total Guests",
        "has_children": "Has Children",
        "total_stay_nights": "Total Stay Nights",
        "staying_on_weekends": "Staying on Weekends",
        "Country_Risk": "Country Risk",
        "is_board": "Board Included",
        "is_resort": "Resort Booking",
        "has_special_requests": "Has Special Requests",
        "season_Spring": "Spring Season",
        "season_Summer": "Summer Season",
        "season_Winter": "Winter Season",
        "reserved_room_D": "Reserved Room Type D",
        "reserved_room_Other": "Reserved Room Type Other"
    }

    yes_no_columns = [
        "is_repeated_guest", "has_children", "staying_on_weekends",
        "is_board", "is_resort", "has_special_requests"
    ]
    dropdown_columns = {
        "TrustedAgent": [0, 1, 2, 3],
        "PartnerAgent": [1, 2, 3, 4],
        "Country_Risk": [1, 2, 3, 4]
    }
    month_to_season = {
        "January": "Winter", "February": "Winter", "March": "Spring", "April": "Spring",
        "May": "Spring", "June": "Summer", "July": "Summer", "August": "Summer",
        "September": "Autumn", "October": "Autumn", "November": "Autumn", "December": "Winter"
    }
    months = list(month_to_season.keys())
    reserved_room_types = ["A", "D", "Other"]

    default_values = {
        "lead_time": 100,
        "adr": 45,
        "total_guests": 1,
        "total_stay_nights": 1,
        "CSMR_SENT": 90
    }

    # --- Input UI ---
    col1, col2 = st.columns(2)
    inputs = {}

    with col1:
        inputs["is_repeated_guest"] = int(st.selectbox(display_names["is_repeated_guest"], ["No", "Yes"]) == "Yes")
        inputs["lead_time"] = st.number_input(display_names["lead_time"], value=default_values["lead_time"])
        inputs["adr"] = st.number_input(display_names["adr"], value=default_values["adr"])
        inputs["total_guests"] = st.number_input(display_names["total_guests"], value=default_values["total_guests"])
        inputs["has_children"] = int(st.selectbox(display_names["has_children"], ["No", "Yes"]) == "Yes")
        inputs["TrustedAgent"] = st.selectbox(display_names["TrustedAgent"], dropdown_columns["TrustedAgent"])
        inputs["PartnerAgent"] = st.selectbox(display_names["PartnerAgent"], dropdown_columns["PartnerAgent"])
        inputs["total_stay_nights"] = st.number_input(display_names["total_stay_nights"], value=default_values["total_stay_nights"])
        inputs["staying_on_weekends"] = int(st.selectbox(display_names["staying_on_weekends"], ["No", "Yes"]) == "Yes")

    with col2:
        selected_month = st.selectbox("Arrival Month", months)
        inputs["is_resort"] = int(st.selectbox(display_names["is_resort"], ["No", "Yes"]) == "Yes")
        inputs["is_board"] = int(st.selectbox(display_names["is_board"], ["No", "Yes"]) == "Yes")
        inputs["has_special_requests"] = int(st.selectbox(display_names["has_special_requests"], ["No", "Yes"]) == "Yes")
        selected_room = st.selectbox("Reserved Room Type", reserved_room_types)
        
        # Replace Country_Risk dropdown with country dropdown and map to risk value
        @st.cache_data
        def get_country_risk_map(df):
            return dict(zip(df['country'], df['Country_Risk']))
        #df.head()
        country_risk_map = get_country_risk_map(df)
        #country_risk_map = dict(zip(df['country'], df['Country_Risk']))
        countries = sorted(df['country'].unique())

        selected_country = st.selectbox("Country", countries)
        inputs["Country_Risk"] = country_risk_map.get(selected_country)  # Default risk value if not found
        inputs["INFLATION_CHG"] = st.number_input(display_names["INFLATION_CHG"], value=0.0)
        inputs["CSMR_SENT"] = st.number_input(display_names["CSMR_SENT"], value=default_values["CSMR_SENT"])

    # --- Feature Engineering for one-hot columns ---
    season = month_to_season[selected_month]
    inputs["season_Winter"] = int(season == "Winter")
    inputs["season_Spring"] = int(season == "Spring")
    inputs["season_Summer"] = int(season == "Summer")
    inputs["reserved_room_D"] = int(selected_room == "D")
    inputs["reserved_room_Other"] = int(selected_room == "Other")

    # --- Prediction ---
    if st.button("Predict Cancellation"):
        # Prepare input for model
        user_input = {col: inputs.get(col, 0) for col in X_columns}
        user_df = pd.DataFrame([user_input])
        user_scaled = scaler.transform(user_df)
        pred = rf_model.predict(user_scaled)[0]
        prob = rf_model.predict_proba(user_scaled)[0]
        cancel_risk = round(prob[1] * 100, 1)
        if pred == 0:
            st.success(f"Ziyaretçinin iptal etme olasılığı %{cancel_risk}")
        else:
            st.error(f"Ziyaretçinin iptal etme olasılığı %{cancel_risk}") 
    
    st.markdown("""
    <b>Kullanılan Model:</b> <span style="color:#2980b9;">Random Forest Classifier</span><br>
    <b>Model, rezervasyonun iptal edilip edilmeyeceğini tahmin etmek için eğitilmiştir.</b>
    """, unsafe_allow_html=True)

    # Özellik önem grafiğini göster
    #st.markdown("Model Özellik Önem Grafiği")
    fig = plotly_feature_importance_streamlit(rf_model, X_classification, title="Random Forest Classifier Feature Importance")
    st.plotly_chart(fig, use_container_width=True)

#Özellik Müh.
elif page == "🛠️ Feature Engineering":
    st.header("🛠️ Veri Ön İşleme ve Özellik Mühendisliği")

    st.subheader("1️⃣ Aykırı Değer (Outliers) İşlemleri")
    st.markdown("""
    - **Outlier Thresholds** fonksiyonu ile uç değer sınırları belirlendi.
    - Uç değerler tespit edilerek, eşik değerlerle değiştirildi.
    - Tüm **sayısal değişkenler** için ayrı kontrol yapıldı.
    - Aykırı değerlerin belirlenmesinde veri dağılımının en uç %1 ve %99’luk dilimleri temel alındı. Bu dilimler arasındaki fark (IQR) kullanılarak 1.5 × IQR kuralı ile alt ve üst eşikler hesaplandı. Böylece yalnızca analiz sonuçlarını bozabilecek ekstrem değerler tespit edildi ve bu değerler eşik değerlerle değiştirildi.
    """)

    st.header("2️⃣ Eksik Değer Analizi ve Doldurma")
    st.markdown("""
    - Eksik değerler hem **sayısal** hem **kategorik** sütunlar için tespit edildi.
    - Eksik değerlerin **bağımlı değişken (is_canceled)** ile ilişkisi analiz edildi.
    - Eksik değerlerin o sütunlardaki dağılımı incelendi.
    - **Doldurma Stratejileri:**
        - `country`: Mod ile dolduruldu.
        - `children`: 0 ile dolduruldu.
        - Makroekonomik değişkenler: Aynı tarihlerdeki değerlerle dolduruldu.
        - `agent`: 'Unknown' ile dolduruldu.
    """)
    
    st.header("3️⃣ Yeni Özelliklerin Oluşturulması")
    st.markdown("""
    ##### 🤝 Acente ve Ülkelerin Gruplandırılması
    - Risk ve ortaklık seviyelerini yakalamak için yeni kategorik özellikler oluşturuldu:
        - **Trusted Agent:** Acenteleri, geçmiş iptal oranlarına göre 4 seviyeye ayırdık (örneğin, düşük iptal oranı daha güvenilir bir acenteye işaret eder).
        - **Partner Agent:** Acenteleri, rezervasyon hacimlerine göre 4 seviyeye ayırdık (örneğin, yüksek hacim güçlü bir ortaklığa işaret eder).
        - **Ülke Riski:** Her bir ülkenin ortalama iptal oranına göre bir risk puanı atandı.
    - ADR tahmininde ülke ve acenteler günlük ortalama ücret dağılımlarına göre gruplandırıldı.
    ##### 👨‍👩‍👧‍👦 Misafir ve Konaklama Özellikleri
    - Her bir rezervasyon hakkında daha bütünsel bir bakış açısı sunmak için ek sezgisel özellikler oluşturuldu:
        - **total_guest:** Bir rezervasyondaki yetişkin, çocuk ve bebek sayılarının toplamı.
        - **has_children:** Bir rezervasyonun çocuk içerip içermediğini gösteren ikili bir bayrak (1 veya 0).
        - **total_nights:** Hafta sonu ve hafta içi gece sayılarının toplamı.
        - **has_weekend_stay:** Hafta sonu konaklama içeren rezervasyonlar için ikili bir bayrak (1 veya 0).
        - **is_resort**, **has_special_requests**: Konaklama tipi ve özel istek bilgileri.
        - **reserved_room**, **is_board**: Oda tipi ve yemek planı bilgilerini birleştiren değişkenler.

    ##### 🌍 Mevsimsel ve Rezervasyonla İlgili Özellikler
    - `arrival_date_month` (varış tarihi ayı) özelliği, rezervasyon davranışının güçlü bir göstergesi olabilecek kategorik bir mevsim özelliğine (Kış, İlkbahar, Yaz, Sonbahar) dönüştürüldü.
    - Ayrıca, özel isteklerin genellikle iptal olasılığıyla ilişkili olması nedeniyle, bir misafirin özel isteği olup olmadığını gösteren bir **özel_istek_var_mı** özelliği de oluşturuldu.

    """)

    st.subheader("4️⃣ Makroekonomik Değişken Analizi")
    st.markdown("""
    - 11 adet makro değişken arasındaki korelasyon matrisi incelendi.
    - En düşük korelasyona sahip çiftler incelendi: **INFLATION_CHG** & **CSMR_SENT**.
    """)
    st.image("macro_correlation.png", caption="Makroekonomik Değişken Korelasyon Matrisi")

    st.subheader("5️⃣ Modelleme İçin Son Hazırlıklar")
    st.markdown("""
    **1. Veriyi Bağımlı ve Bağımsız Değişkenlere Ayırma**
    - Modelimizin tahmin etmeye çalışacağı hedef değişken (bağımlı değişken), iptal edilip edilmediğini gösteren `is_canceled` sütunudur.
    - Geriye kalan, bu tahmini yapmak için kullanacağımız tüm diğer sütunlar ise bağımsız değişkenleri oluşturur (X).
    - Bu adımda, gereksiz veya model için sorun yaratabilecek sütunları da X'in içinden çıkardık.

    **2. Kategorik Özellikleri Kodlama**
    - Bazı özellikler (season, reserved_room gibi) sayısal olmayan, yani kategorik değerler içerir.
    - Makine öğrenimi algoritmalarının bu verileri kullanabilmesi için bunları sayısal formata dönüştürmemiz gerekir.
    - Bu işlem için One-Hot Encoding yöntemi kullanıldı. Bu yöntem, her bir kategorik değer için ayrı bir ikili (binary) sütun oluşturarak veriyi makine için anlamlı hale getirir.

    **3. Sayısal Özellikleri Ölçekleme**
    - adr veya total_stay_nights gibi sayısal veriler, farklı değer aralıklarına sahip olabilir.
    - Bu, bazı algoritmaların büyük değer aralığına sahip özelliklere daha fazla ağırlık vermesine neden olabilir.
    - Bu durumu önlemek için, StandardScaler kullanarak tüm sayısal özellikleri standart bir dağılıma (ortalama 0, standart sapma 1) sahip olacak şekilde ölçeklendirdik.
    """)

    st.subheader("6️⃣ Sonuç")
    st.markdown("""
    - Temizlenmiş, eksik değerleri doldurulmuş, anlamlı yeni değişkenler eklenmiş bir veri seti elde edildi.
    - Bu veri seti, hem makine öğrenmesi modelleri hem de iş zekâsı raporlaması için **hazır hale getirildi**.
    - **Son Kayıt Sayısı:** `{:,.0f}` satır.
    """.format(115344))  # buraya df.shape[0] yazabilirsin

    st.success("✅ Feature Engineering süreci tamamlandı!")

#İçgörüler ve Öneriler
elif page == "💡 Insights and Recommendations":
    st.header("💡 Bulgular ve İş Önerileri")

    st.markdown("""
    ### Key Insights from Analysis

    - **High Cancellation Risk:** Certain market segments and countries show significantly higher cancellation rates. Proactive communication and flexible policies may reduce risk.
    - **Seasonal Trends:** Bookings and cancellations vary by season. Summer months have higher booking volumes, but also increased cancellation risk.
    - **ADR Optimization:** Dynamic pricing strategies based on lead time, guest profile, and season can help maximize revenue.
    - **Guest Profile:** Families (with children) tend to book longer stays and are less likely to cancel compared to solo travelers.
    - **Special Requests:** Bookings with special requests have a lower cancellation rate, indicating higher commitment.

    ### Recommendations

    1. **Targeted Offers:** Provide special incentives for guests from high-risk countries or segments to reduce cancellations.
    2. **Flexible Policies:** Consider more flexible cancellation policies during low-demand periods to attract bookings.
    3. **Dynamic Pricing:** Use ADR prediction model outputs to adjust prices based on demand, season, and guest characteristics.
    4. **Monitor Lead Time:** Closely monitor bookings with short lead times, as these are more likely to be canceled.
    5. **Leverage Feature Importance:** Focus marketing and operational efforts on the most influential features identified by the models.

    ---
    <span style="color:#2980b9;">For further details, see the EDA and Feature Engineering tabs.</span>
    """, unsafe_allow_html=True)

elif page == "📑 Appendix":
    st.header("📑 Appendix")

    st.markdown("### Country & Country Risk Table")

    # Tabloyu yükle
    df = pd.read_csv("hotel_bookings_preprocessed.csv")
    country_table = df[["country", "Country_Risk"]].drop_duplicates().sort_values("Country_Risk").reset_index(drop=True)

    st.dataframe(country_table, use_container_width=True)



st.sidebar.markdown("---")

