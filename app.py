################################################
# MIULL Project
################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit import dataframe
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import joblib

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


df = pd.read_csv("hotel_bookings_raw.csv")

df.head()

df[df["babies"]!=0]
df["babies"].value_counts()

#df = pd.read_csv("hotel_bookings_preprocessed.csv")

df["adults"].value_counts()

# Prepapre data
# Convert 'is_canceled' to boolean
df['is_canceled'] = df['is_canceled'].astype(bool)
# convert the reservation_status_date into date
df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])
#convert agent column from int to object
df['agent'] = df['agent'].astype(object)

df.head()

# Add a new column that sums adults, children, and babies
#df['total_guests'] = df['adults'] + df['children'] + df['babies']
################################################
# 1. Exploratory Data Analysis
################################################
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    #print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    numeric_df = dataframe.select_dtypes(include='number')
    print(numeric_df.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

# convert the reservation_status_date into date
df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])
#convert agent column from int to object
df['agent'] = df['agent'].astype(object)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

check_df(df)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)

def grab_col_names(dataframe, cat_th=3, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    # Only select columns that do NOT include 'date' in their name for num_but_cat
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                    dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    #cat_cols = [col for col in cat_cols if col not in cat_but_car]
    #cat_cols = [col for col in cat_cols]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ['int64', 'float64']]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f"  -> İsimleri: {cat_cols}")
    print(f'num_cols: {len(num_cols)}')
    print(f"  -> İsimleri: {num_cols}")
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f"  -> İsimleri: {cat_but_car}")
    print(f'num_but_cat: {len(num_but_cat)}')
    print(f"  -> İsimleri: {num_but_cat}")
    return cat_cols, num_cols, cat_but_car

# Değişken türlerinin ayrıştırılması
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)
df.head()
# Kategorik değişkenlerin incelenmesi
for col in cat_cols:
    cat_summary(df, col)

# Sayısal değişkenlerin incelenmesi
quantiles = [q/100 for q in range(0, 101, 5)]  # 0%, 5%, 10%, ..., 100%
quantiles   = [0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95,0.99, 1]
df[num_cols].describe(percentiles=quantiles).T

df[df['adr'] == 5400]# ADR'si 0 olan kayıtların sayısı
#df[num_cols].describe().T

for col in num_cols:
    num_summary(df, col, plot=True)

# Sayısal değişkenkerin birbirleri ile korelasyonu
correlation_matrix(df, num_cols)

# Target ile sayısal değişkenlerin incelemesi
for col in num_cols:
    target_summary_with_num(df, "is_canceled", col)
# Target ile kategorik değişkenlerin incelemesi
for col in cat_cols:
    target_summary_with_cat(df, "is_canceled", col)


# Return the count of rows where adr is equal to 0
adr_zero_count = df[df['adr'] == 0].shape[0]
print(f"Number of rows where adr is 0: {adr_zero_count}")

#print(df[df['agent'].isnull()]['is_canceled'].value_counts())


################################################
# 2. Data Preprocessing & Feature Engineering
################################################

def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


#def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    #   low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    #  if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
    #     return True
    #else:
        #   return False


def check_outlier(dataframe, col_name, q1=0.01, q3=0.99):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    col = dataframe[col_name]
    if pd.api.types.is_datetime64_any_dtype(col):
        # For datetime columns, use a different check
        mask = (col > up_limit) | (col < low_limit)
        return mask.sum() > 0
    else:
        return ((col > up_limit) | (col < low_limit)).any()

for col in num_cols:
    print(col, check_outlier(df, col, 0.01, 0.99))

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

check_df(df)

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    # Eksik değer sayısı ve oranını tek bir DataFrame'de birleştirir.
    # 'keys' parametresi sütun isimlerini belirler ('n_miss', 'ratio').
    # 'np.round(ratio, 2)' oranları 2 ondalık basamağa yuvarlar.
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    # Eğer 'na_name' True ise, eksik değer içeren sütunların listesini döndürür.
    # Bu, fonksiyonun hem tabloyu yazdırmasını hem de sütun listesini geri vermesini sağlar.
    if na_name:
        return na_columns

# Fonksiyonu çağırarak eksik değer tablosunu gösterir.
missing_values_table(df)

###################
# Eksik Değerlerin Bağımlı Değişken ile İlişkisinin İncelenmesi
###################
# Önceki kod bloğundan gelen 'missing_values_table' fonksiyonu kullanılarak eksik sütunların listesi alınır.
na_cols = missing_values_table(df, True)

# Eksiklik bayrağı ile hedef değişken arasındaki ilişkiyi inceleyen fonksiyon.
def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    # Her bir eksik sütun için 'NA_FLAG' (Eksiklik Bayrağı) adında yeni bir ikili (binary)
    # değişken oluşturur. Eğer sütunda değer eksikse 1, değilse 0 atanır.
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    # Yeni oluşturulan tüm 'NA_FLAG' sütunlarını seçer.
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    # Her bir 'NA_FLAG' sütunu için döngü yapar.
    for col in na_flags:
        # İlgili 'NA_FLAG' sütununa göre gruplandırma yapar (yani eksik olanlar ve olmayanlar).
        # Her gruptaki hedef değişkenin (target) ortalamasını ve gözlem sayısını hesaplar.
        # Bu, eksik olmanın hedef değişkenin ortalamasını etkileyip etkilemediğini gösterir.
        # Eğer eksik olan grubun hedef ortalaması, eksik olmayan grubun hedef ortalamasından belirgin şekilde farklıysa,
        # eksiklik rastgele değildir (MAR veya MNAR) ve bu bilgi modellemede kullanılmalıdır.
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


# Fonksiyonu çağırır. Burada 'Survived' (Hayatta Kalma) bağımlı değişken olarak inceleniyor.
# (df, "Survived", na_cols)
missing_vs_target(df, "is_canceled", na_cols)


###############
# Filling missing values
################

# 'country' sütunundaki eksik değerleri mod ile doldurur.
df['country'].fillna(df['country'].mode()[0], inplace=True)

#'children' sütunundaki eksik değerleri 0 ile doldurur.
df['children'] = df['children'].fillna(0)

#"agent" dışındaki sütunlardaki eksik değerleri MO_YR değerlerindeki değerlerin ortalamaları ile değiştir.
def fill_missing_values(dataframe, col_name, method='mean'):
    if method == 'mean':
        dataframe[col_name].fillna(dataframe[col_name].mean(), inplace=True)
    elif method == 'median':
        dataframe[col_name].fillna(dataframe[col_name].median(), inplace=True)
    elif method == 'mode':
        dataframe[col_name].fillna(dataframe[col_name].mode()[0], inplace=True)
    else:
        raise ValueError("Method must be 'mean', 'median', or 'mode'.")

columns_list = [
    "CPI_AVG",
    "INFLATION",
    "INFLATION_CHG",
    "CSMR_SENT",
    "UNRATE",
    "INTRSRT",
    "GDP",
    "FUEL_PRCS",
    "CPI_HOTELS",
    "US_GINI",
    "DIS_INC"
]

# Fill missing values for each column in columns_list
for col in columns_list:
    fill_missing_values(df, col, method='mean')

# 'adr' sütununda 0 olan değelere sahip satırları kaldırır.
df = df[df['adr'] != 0]

df.shape

# Show only market_segment and distribution_channel where agent is null--> Agenta ile gelmeyenleri çıkartır
null_agents = df[df['agent'].isnull()]
result = null_agents.groupby(['market_segment', 'distribution_channel']).size().reset_index(name='count')
print(result)

not_null_agents = df[df['agent'].notnull()]
result_not_null = not_null_agents.groupby(['market_segment', 'distribution_channel']).size().reset_index(name='count')

# Fill missing values in 'agent' column with 'Unknown'
df['agent'] = df['agent'].fillna('Unknown')


# Makro ekeonomik verilerden seçilen değişkenler: DIS_INC, UNRATE, INFLATION, FUEL_PRCS

check_df(df)


###################
# Yeni Özellikler Ekleme
###################
df['total_revenue'] = df.apply(
    lambda row: row['adr'] * (row['stays_in_weekend_nights'] + row['stays_in_week_nights']) if not row['is_canceled'] else 0,
    axis=1
)

# Toplam geliri yıl-ay bazında grupla
df['year_month'] = df['reservation_status_date'].dt.to_period('M').astype(str)
monthly_revenue = df.groupby('year_month')['total_revenue'].sum().sort_index()

# Grafikle görselleştir
plt.figure(figsize=(12, 6))
monthly_revenue.plot(kind='line', marker='o')
plt.title('Total Revenue vs Year-Month')
plt.xlabel('Year-Month')
plt.ylabel('Total Revenue')
plt.xticks(rotation=45)
plt.tight_layout()

# Filter out observations outside the desired year-month range
#df['year_month'] = df['reservation_status_date'].dt.to_period('M').astype(str)
df = df[(df['year_month'] >= '2015-06') & (df['year_month'] <= '2017-08')]

###################
# Agent işlemleri
###################

# Group by 'agent', calculate the average of is_canceled, count, and sort descending
agent_cancel_stats = df.groupby(['agent','market_segment','distribution_channel']).agg(
    cancel_rate=('is_canceled', 'mean'),
    booking_count=('is_canceled', 'count')
).sort_values(by='booking_count', ascending=False)
#Agent Category:-> yeni Kolonlar eklenecek
#cancel rate = 1.0, Trust = 0
#cancel rate = 0.7-1, Trust = 1
#cancel rate = 0.5-0.7, Trust = 2
#cancel rate < 0.5, Trust = 3
#booking_count >= 10000, Partnership = 1
#booking_count = 1000-10000, Partnership = 2
#booking_count = 100- 1000, Partnership = 3
# booking_count < 100, Partnership = 4

# Add TrustedAgent and Partnership columns based on agent_cancel_stats

def trust_level(cancel_rate):
    if cancel_rate >= 0.9:
        return 0
    elif 0.7 <= cancel_rate < 0.9:
        return 1
    elif 0.5 <= cancel_rate < 0.7:
        return 2
    else:
        return 3

def partnership_level(booking_count):
    if booking_count >= 10000:
        return 1
    elif 1000 <= booking_count < 10000:
        return 2
    elif 100 <= booking_count < 1000:
        return 3
    else:
        return 4

# Reset index to merge with df
agent_cancel_stats_reset = agent_cancel_stats.reset_index()

# Merge stats to main df
df = df.merge(agent_cancel_stats_reset[['agent', 'market_segment', 'distribution_channel', 'cancel_rate', 'booking_count']],
            on=['agent', 'market_segment', 'distribution_channel'],
            how='left')

# Apply functions to create new columns
df['TrustedAgent'] = df['cancel_rate'].apply(trust_level)
df['PartnerAgent'] = df['booking_count'].apply(partnership_level)

df.head(20)

###################
# Yeni Özellikler Eklemeye devam
###################


# Add a new column that sums adults, children, and babies
df['total_guests'] = df['adults'] + df['children'] + df['babies']
# Add a new column: has_children_or_babies (1 if babies or children > 0, else 0)
df['has_children'] = ((df['children'] > 0)).astype(int)
# Add a new column for total stay nights: stays_in_weekend_nights + stays_in_week_nights
df['total_stay_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
# Add a new column for boolean values staying on weekends
df['staying_on_weekends'] = ((df['stays_in_weekend_nights'] > 0)).astype(int)
#df['year_month'] = df['reservation_status_date'].dt.to_period('M')

df.head(20)


#Ülkeleri riskli ve değil diye sınflandırma
# Calculate cancellation rate for each country
country_cancel_stats = df.groupby('country')['is_canceled'].mean().reset_index()
country_cancel_stats.columns = ['country', 'cancelation_rate']

# Define risk level function
def country_risk_level(rate):
    if rate > 0.9:
        return 1
    elif 0.7 < rate <= 0.9:
        return 2
    elif 0.5 < rate <= 0.7:
        return 3
    else:
        return 4

# Assign risk level
country_cancel_stats['Country_Risk'] = country_cancel_stats['cancelation_rate'].apply(country_risk_level)

# Merge risk level back to main df
df = df.merge(country_cancel_stats[['country', 'Country_Risk']], on='country', how ='left')
df["Country_Risk"].value_counts()

#arrival_date_month bu değeri sezonlara böl (3-4)
# arrival_date_month değerini sezonlara böl

def get_season(month):
    # Kış: Aralık, Ocak, Şubat
    # İlkbahar: Mart, Nisan, Mayıs
    # Yaz: Haziran, Temmuz, Ağustos
    # Sonbahar: Eylül, Ekim, Kasım
    if month in ['December', 'January', 'February']:
        return 'Winter'
    elif month in ['March', 'April', 'May']:
        return 'Spring'
    elif month in ['June', 'July', 'August']:
        return 'Summer'
    else:
        return 'Autumn'

df['season'] = df['arrival_date_month'].apply(get_season)
df.head(20)
# meal-reserved_room_type-customer type-required_car_parking_spaces-total_of_special_requests (Bunlardan bir adet kolon oluştur)
df["meal"].value_counts()
# Group by reserved_room_type and meal, count and average is_canceled
meal_room_group = df.groupby(['reserved_room_type', 'meal']).agg(
    count=('is_canceled', 'size'),
    avg_is_canceled=('is_canceled', 'mean')
).reset_index()

# Add a new column called reserved_room and group B, C, E, F, G, H, L as 'Other'
df['reserved_room'] = df['reserved_room_type'].apply(lambda x: x if x not in ['B', 'C', 'E', 'F', 'G', 'H', 'L'] else 'Other')

# Add a new column called is_board: 1 if meal is BB, FB, HB; else 0
df['is_board'] = df['meal'].isin(['BB', 'FB', 'HB']).astype(int)

check_df(df)
# is_repeated_guest kalsın...


#Makro ekonomoik arasındaki korrelasyona bak ve bir iki veri seç:
# Makro ekonomik değişkenler arasındaki korelasyona bak ve en yüksek iki korelasyonu seç
# Makro ekonomik değişkenler arasındaki korelasyona bak ve en yüksek iki korelasyonu seç

macro_cols = ["CPI_AVG", "INFLATION", "INFLATION_CHG", "CSMR_SENT", "UNRATE", "INTRSRT", "GDP", "FUEL_PRCS", "CPI_HOTELS", "US_GINI", "DIS_INC"]

# Korelasyon matrisini çiz
corr_matrix = df[macro_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='RdBu', linewidths=0.5)
plt.title("Macro Variables Correlation Matrix")
plt.show()

# En az korelasyona sahip iki değişkeni seç
corr_pairs_min = corr_matrix.abs().unstack().sort_values(ascending=True)
corr_pairs_min = corr_pairs_min[corr_pairs_min > 0]  # 0 olanlar diagonal, onları çıkar
min2 = corr_pairs_min.drop_duplicates().head(2)
print("Birbirleriyle korelasyonu en az olan iki değişken çifti:")
print(min2)  # --> INFLATION_CHG  CSMR_SENT    0.095152


# Add a new column called is_resort: 1 if hotel is 'Resort Hotel', else 0
df['is_resort'] = (df['hotel'] == 'Resort Hotel').astype(int)

# Add a new boolean column: 1 if total_of_special_requests > 0, else 0
df['has_special_requests'] = (df['total_of_special_requests'] > 0).astype(int)

df.shape
# Total Guest değeri 0 olanları sil
df = df[df['total_guests'] > 0]

df.shape

##previous_bookings_not_canceled kolonu da al-->


# En Son oluşturulan dataframe'i csv olarak kaydet
df.to_csv("hotel_bookings_preprocessed.csv", index=False)

df["total_guests"].value_counts()

df.head(20)
################################################
# 3. Splitting X and y and Encoding categorical features and scaling X
################################################
X_classification = df.drop(columns=['is_canceled', 'hotel','arrival_date_year', 'arrival_date_month', 'arrival_date_week_number', 'arrival_date_day_of_month','stays_in_weekend_nights', 'stays_in_week_nights','adults','children','babies','meal','country','market_segment','distribution_channel','previous_cancellations','previous_bookings_not_canceled','reserved_room_type','assigned_room_type',	'booking_changes',	'deposit_type',	'agent','days_in_waiting_list','customer_type','required_car_parking_spaces','total_of_special_requests','reservation_status','reservation_status_date','MO_YR', 'CPI_AVG',	'INFLATION','UNRATE','INTRSRT','GDP','FUEL_PRCS','CPI_HOTELS','US_GINI','DIS_INC','total_revenue','year_month','cancel_rate','booking_count'])
y = df['is_canceled']
#df["INFLATION"].value_counts()
X_classification.shape
X_classification.head()
# One-hot encoding for categorical features
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    # Verilen DataFrame ve kategorik sütun listesini One-Hot Encoding'e tabi tutar.
    # 'drop_first' parametresi dışarıdan kontrol edilebilir.
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first,dtype='int64')
    return dataframe

cat_cols_x = ['season', 'reserved_room']
X_classification = one_hot_encoder(X_classification, cat_cols_x, drop_first=True)

X_classification.describe().T

X_classification.head()

X_classification['total_guests'].value_counts()


# Scaling numerical features
X_scaled = StandardScaler().fit_transform(X_classification)
X_scaled= pd.DataFrame(X_scaled, columns=X_classification.columns)
X_scaled["total_guests"].value_counts()
X_scaled.head()

#X_scaled['total_guests'].value_counts()



# Fit scaler on training data --> Bunu da .pkl formatında kaydet
scaler = StandardScaler()
scaler.fit(X_classification)

# Save scaler to disk
joblib.dump(scaler, "scaler.pkl")

##################################################
# 4. Base Modeling
##################################################
# Base model:
def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [#('LR', LogisticRegression()),
                #('KNN', KNeighborsClassifier()),
                #("SVC", SVC()),
                #("CART", DecisionTreeClassifier()),
                #("RF", RandomForestClassifier()),
                #('Adaboost', AdaBoostClassifier()),
                ('GBM', GradientBoostingClassifier()),
                #('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                #('LightGBM', LGBMClassifier()),
                ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=5, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")

base_models(X_scaled, y, scoring="roc_auc")

# 1. Fit the model on all data
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_scaled, y)
X_scaled.to_csv("X_classification.csv", index=False)


#Feature Importance
def plot_importance(model, features, num=len(X_scaled), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X_scaled)

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_scaled)

# 2. Predict on a new entry
random_user = X_scaled.sample(1, random_state=42)
prediction = rf_model.predict(random_user)
probability = rf_model.predict_proba(random_user)

# 3. Predict
print("Prediction (is_canceled):", prediction[0])
print("Probability [Not Canceled, Canceled]:", probability[0])

#Bu modeli kaydetmek için
joblib.dump(rf_model, "rf_model.pkl")

##################################################
# 5. GBM Model--> Hyperparameter Tuning 
##################################################
gbm_model = GradientBoostingClassifier(random_state=17)

gbm_model.get_params()

cv_results = cross_validate(gbm_model, X_scaled, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
#np.float64(0.6762814255494236)
cv_results['test_f1'].mean()
#np.float64(0.5598377008401133)
cv_results['test_roc_auc'].mean()
# np.float64(0.7417073154319364)
gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8, 10],
              "n_estimators": [100, 500, 1000],
              "subsample": [1, 0.5, 0.7]}

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X_scaled, y)

gbm_best_grid.best_params_

gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17, ).fit(X_scaled, y)


cv_results = cross_validate(gbm_final, X_scaled, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean() 
cv_results['test_roc_auc'].mean() 



# Load data
df = pd.read_csv("hotel_bookings_raw.csv")

# Prepapre data
# Add a new column that sums adults, children, and babies
df['total_guests'] = df['adults'] + df['children'] + df['babies']
# Add a new column for total revenue: adr * (stays_in_weekend_nights + stays_in_week_nights)
df['total_revenue'] = df['adr'] * (df['stays_in_weekend_nights'] + df['stays_in_week_nights'])
# Add a new column for total stay nights: stays_in_weekend_nights + stays_in_week_nights
df['total_stay_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
# Add a new column: has_children_or_babies (1 if babies or children > 0, else 0)
df['has_children_or_babies'] = ((df['children'] > 0) | (df['babies'] > 0)).astype(int)

df.to_csv("hotel_bookings_visualization.csv", index=False)


