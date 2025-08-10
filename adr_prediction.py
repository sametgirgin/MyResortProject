### İMPORT İŞLEMLERİ #####
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly as plt
import warnings

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px

import math

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb
import joblib

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


from scipy import stats
from scipy.stats import norm, skew #for some statistics


df = pd.read_csv("hotel_bookings_raw.csv")

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

check_df(df)

#### Bazı dataların daha okunur olması için düzenleme yapıldı .
# Prepapre data
# Convert 'is_canceled' to boolean
df['is_canceled'] = df['is_canceled'].astype(bool)
# convert the reservation_status_date into date
df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])
#convert agent column from int to object
df['agent'] = df['agent'].astype(object)

### MO_YR SIKINTI ÇIKARTIĞI İÇİN ONU KALDIRIM YERİNE BAŞKA BİR KOLON EKLİYORUM

df.drop('MO_YR', axis=1, inplace=True)
df['rezerve_ay_yil'] = df['reservation_status_date'].dt.to_period('M')

##### Tarih verilerini incelediğimde sağlık gelen zaman verilerini dışarıda bırakıyorum .

rezervasyon_sayilari = df.groupby('rezerve_ay_yil').size()
# Grafiğin boyutunu ayarlayalım
plt.figure(figsize=(15, 7))
plt.plot(rezervasyon_sayilari.index.astype(str), rezervasyon_sayilari.values, marker='o', linestyle='-')
plt.title('Rezervasyon Durumlarının Zamana Göre Dağılımı', fontsize=16)
plt.xlabel('Tarih (Ay-Yıl)', fontsize=12)
plt.ylabel('Toplam Rezervasyon Sayısı', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

df = df[(df['rezerve_ay_yil'] >= '2015-06') & (df['rezerve_ay_yil'] <= '2017-08')]

check_df(df)

#### Kategorik - Numarik incelemesi


def grab_col_names(dataframe, cat_th=5, car_th=20):
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
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    # cat_cols = [col for col in cat_cols if col not in cat_but_car]
    # cat_cols = [col for col in cat_cols]

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

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)

num_cols = [col for col in df.columns if df[col].dtypes in ['int64', 'float64']]
cat_cols = [col for col in df.columns if df[col].dtypes in ['object', 'category']]
check_df(df)

### Kategorik ve numarik Kolonlarımı incelmeye devam ediyorum .

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)

###

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

    print("#####################################")


for col in num_cols:
    num_summary(df, col, plot=False)

##### Değişkenlerimin Target ile arasındaki ilişkilere bakıp , Özellik Bölüme hazırlık yapıyorum

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df,"adr",col)

for col in num_cols:
    target_summary_with_num(df,"adr",col)

### İnceleme sonucu  .
    ### Zaman değişkenlerinde Ayları sayısal değere ve Mevsellik etkisine bakmaya karar veriyorum .
    ### Ülke ve Agenta değişknelerimin sayıları fazla olduğu için bunlarıda Adr ortalamalarına göre gruplandıracağım
    ### Market Segment - Oda tipleri - Meal - Customer_Type - Disribution_chanelde tekrardan gruplandıralacak

df["adr"].hist(bins=100)
plt.show()

######################################
# 5. Korelasyon Analizi (Analysis of Correlation)
######################################

corr = df[num_cols].corr()
corr

# Korelasyonların gösterilmesi
sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()



def high_correlated_cols(dataframe, plot=False, corr_th=0.70):
    corr = dataframe[num_cols].corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

yuksek_korelasyonlu_sutunlar = high_correlated_cols(df, plot=True)

adr_zero_count = df[df['adr'] == 0].shape[0]
print(f"Number of rows where adr is 0: {adr_zero_count}")

check_df(df)

### Aralarında yüksek korelasyon Olan ['CPI_AVG', 'UNRATE', 'INTRSRT', 'GDP', 'CPI_HOTELS', 'DIS_INC'] değerlerden sadece biri denkleme alınacak

# Silinecek kolonları bir liste halinde tanımlayalım.
silinecek_kolonlar = ['CPI_AVG', 'UNRATE', 'INTRSRT', 'CPI_HOTELS', 'DIS_INC']
df.drop(silinecek_kolonlar, axis=1, inplace=True)
print(df.columns)

### kolonlarımı tekrar kontrol ediyorum.

num_cols = [col for col in df.columns if df[col].dtypes in ['int64', 'float64']]
cat_cols = [col for col in df.columns if df[col].dtypes in ['object', 'category']]
check_df(df)

high_correlated_cols(df, plot=True)

#####################################
# Görev 2 : Feature Engineering
######################################

######################################
# Aykırı Değer Analizi
######################################

# Aykırı değerlerin baskılanması
def outlier_thresholds(dataframe, variable, low_quantile=0.01, up_quantile=0.99):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit

# Aykırı değer kontrolü
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

### Burada güncelleme yapmam gerekti
df['rezerve_ay_yil'] = df['rezerve_ay_yil'].astype('object')
df['is_canceled'] = df['is_canceled'].astype('object')

for col in num_cols:
    print(col, check_outlier(df, col))

######
# Aykırı değerlerin baskılanması
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    if col != "adr":
        replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

check_df(df)

#### Target harici verileri baskıladım.

### Eksik değerlerimi düzenliyorum

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)

    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])

    print(missing_df, end="\n")

    if na_name:
        return na_columns

# Önceki kod bloğundan gelen 'missing_values_table' fonksiyonu kullanılarak eksik sütunların listesi alınır.
na_cols = missing_values_table(df, True)

df['country'] = df['country'].fillna(df['country'].mode()[0])
df['children'] = df['children'].fillna(0)
### Burada Unknow olarak tanımlamak sıkıntı yarattığı için 0 yaptım
df['agent'] = df['agent'].fillna('0')

missing_values_table(df)

df_yedek = df.copy()

### Rare analizi yaparak daha öncede incelediğiz değişkenler üzerinde yeni düzenlemeler yapacağız

num_cols = [col for col in df.columns if df[col].dtypes in ['int64', 'float64']]
cat_cols = [col for col in df.columns if df[col].dtypes in ['object', 'category']]
check_df(df)

# cat_cols listesindeki tüm sütunları bir döngüde string'e çevir. ( bunu yapmadan rare analiz yapmadım
for col in cat_cols:
    df[col] = df[col].astype(str)


def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "adr", cat_cols)

#### Yeni değişkenler oluşturma

### ÜLkeleri Grupladım
ulke_adr_ort = df.groupby('country')['adr'].mean()
ulke_adr_siniflari = pd.qcut(ulke_adr_ort, q=5, labels=['E', 'D', 'C', 'B', 'A'])
df['New_Country_class'] = df['country'].map(ulke_adr_siniflari)

#####
# 'Group', 'Contract' ve 'Transient-Party' gruplarını 'Sözleşmeli/Grup' gibi bir isimle birleştirme
df['New_customer_type'] = df['customer_type'].replace(['Contract', 'Group', 'Transient-Party'], 'Grup/Sözleşmeli')

#### Odalar

### Odalar
# Az temsil edilen kategorileri birleştirin
az_temsil_edilenler = ['B', 'C', 'H', 'L', 'P']
df['reserved_room_type_grouped'] = df['reserved_room_type'].replace(az_temsil_edilenler, 'Other')

# Odaları adr seviyelerine göre gruplandıran bir sözlük (dictionary) oluşturun ( A en yüksek - B orta - C düşük fiyat )
oda_tipleri_map = {
    'A': 'C',
    'D': 'B',
    'E': 'B',
    'F': 'A',
    'G': 'A',
    'Other': 'D'
}

# Yeni sözlüğü kullanarak gruplandırma yapın
df['New_room_class'] = df['reserved_room_type_grouped'].map(oda_tipleri_map)


#### Market Segment - Grupların dağılımı ve adr Ortalamalarına göre  göre gruplandırıyorum
####Market sengment####
# Gruplandırma için bir sözlük oluşturun
market_segment_map = {
    'Online TA': 'A',
    'Direct': 'A',
    'Offline TA/TO': 'B',
    'Groups': 'B',
    'Corporate': 'C',
    'Aviation': 'D',
    'Undefined': 'D',
    'Complementary':'D'
}

# Yeni sözlüğü kullanarak gruplandırma yapın
df['New_market_segment_grouped'] = df['market_segment'].map(market_segment_map)

###Agent ###
# Her agent'ın ortalama adr değerini hesapla
agent_adr_ort = df.groupby('agent')['adr'].mean()
agent_adr_siniflari = pd.qcut(agent_adr_ort, q=5, labels=['E', 'D', 'C', 'B', 'A'])
df['New_agent_class'] = df['agent'].map(agent_adr_siniflari)

#### Meal
# Gruplandırma için bir sözlük oluşturun - Adr ortalamasına ve oranlara göre grupluyorum 3 grup tespit edildi
meal_map = {
    'BB': 'B',
    'SC': 'B',
    'HB': 'A',
    'FB': 'C',
    'Undefined': 'C'
}

# Yeni sözlüğü kullanarak gruplandırma yapın
df['New_meal_class'] = df['meal'].map(meal_map)

#########
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

df['New_season'] = df['arrival_date_month'].apply(get_season)

######

df['New_arrival_month_numeric'] =df['arrival_date_month'].map({
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
})

#########
df['New_total_guests'] = df['adults'] + df['children'] + df['babies']
df['New_total_stay'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
df = df[df['New_total_guests'] > 0]

df['New_has_children'] = (df['children'] > 0) | (df['babies'] > 0)
df['New_has_children'] = df['New_has_children'].astype(int)

# previous_cancellations sütunundaki sıfır (0) dışındaki tüm değerleri True yapar.
df['New_has_previous_cancellations'] = df['previous_cancellations'] != 0
# herhangi birinde sıfır (0) dışında bir değer olup olmadığını kontrol edin
df['New_has_extra_requests'] = (df['required_car_parking_spaces'] > 0) | (df['total_of_special_requests'] > 0)

# Eğer booking_changes değeri 0'dan büyükse True, aksi halde False yapar.
df['New_has_booking_changes'] = df['booking_changes'] > 0

check_df(df)

df_yedek_2 = df.copy()

### Adr için
# Aykırı değerleri belirlemek için fonksiyon
def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    return up_limit

up_limit = outlier_thresholds(df, "adr")
df = df[df["adr"] < up_limit]
df =df[df['adr'] > 0]

check_df(df)

df.head()
#######
# Yeni DataFrame'e almak istediğiniz tüm kolonları bir liste olarak tanımlayın
istenen_kolonlar = [
'hotel','lead_time','arrival_date_year','New_season','stays_in_weekend_nights','stays_in_week_nights','New_total_guests','New_total_stay','New_has_children','New_meal_class','New_market_segment_grouped',
    'is_repeated_guest', 'New_has_previous_cancellations','New_room_class','New_has_booking_changes','deposit_type','New_agent_class','New_customer_type','adr','New_has_extra_requests','reservation_status','INFLATION'
    ,'GDP','FUEL_PRCS','New_Country_class','New_arrival_month_numeric'
]

# Ana DataFrame'den istediğiniz kolonları seçerek yeni bir DataFrame oluşturun
new_df = df[istenen_kolonlar]

check_df(new_df)

num_cols = [col for col in new_df.columns if new_df[col].dtypes in ['int64', 'float64']]
cat_cols = [col for col in new_df.columns if new_df[col].dtypes in ['object', 'category']]

########### Enkoding
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in new_df.columns if new_df[col].dtypes == "O" and len(new_df[col].unique()) == 2]

for col in binary_cols:
    label_encoder(new_df, col)


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first , dtype = 'int64')
    return dataframe

new_df = one_hot_encoder(new_df, cat_cols, drop_first=True)

check_df(new_df)

# 'drop=True' eski indeksi bir sütun olarak saklamaz.
new_df = new_df.reset_index(drop=True)
X_scaled = StandardScaler().fit_transform(new_df[num_cols])
new_df[num_cols] = pd.DataFrame(X_scaled, columns=new_df[num_cols].columns)

check_df(new_df)

### Modeli kaydet
new_df.to_csv("hotel_bookin_İşlenmiş.csv", index=False)

# Özellikler (X) ve hedef (y) değişkenlerini tanımlayalım
X = new_df.drop('adr', axis=1)
y = new_df['adr']



################# Önemlerini inceledik #####
# X ve y'yi önceki adımlarda olduğu gibi hazırlayalım
# df veri çerçevesi hazır olduğunu varsayıyorum
y = new_df['adr']
X = new_df.drop('adr', axis=1)
# Sütun isimlerindeki boşlukları alt çizgi ile değiştir
X.columns = ["_".join(c.split()) for c in X.columns]
# LightGBM modelini tanımla ve eğit
model = LGBMRegressor()
model.fit(X, y)
# Özellik önem düzeylerini DataFrame olarak al
feature_imp = pd.DataFrame({
    "Value": model.feature_importances_,
    "Feature": X.columns
})
# Önem düzeyine göre azalan sırayla sırala
feature_imp = feature_imp.sort_values(by="Value", ascending=False)
# Sonuçları ekrana yazdır
print("Özellik Önem Düzeyleri (Feature Importances):")
print(feature_imp)

#####
####### Eski veriyi çekip kolonları yeniden oluşturuyorum
df.head()

# Yeni DataFrame'e almak istediğiniz tüm kolonları bir liste olarak tanımlayın
istenen_kolonlar = [
'hotel','lead_time','arrival_date_year','New_season','stays_in_weekend_nights','stays_in_week_nights','New_total_guests','New_total_stay','New_has_children','New_meal_class','New_market_segment_grouped',
     'New_room_class','deposit_type','New_agent_class','New_customer_type','adr','reservation_status','INFLATION'
    ,'GDP','FUEL_PRCS','New_arrival_month_numeric'
]

# Ana DataFrame'den istediğiniz kolonları seçerek yeni bir DataFrame oluşturun
new_df = df[istenen_kolonlar]

check_df(new_df)

num_cols = [col for col in new_df.columns if new_df[col].dtypes in ['int64', 'float64']]
cat_cols = [col for col in new_df.columns if new_df[col].dtypes in ['object', 'category']]

########### Enkoding

binary_cols = [col for col in new_df.columns if new_df[col].dtypes == "O" and len(new_df[col].unique()) == 2]

for col in binary_cols:
    label_encoder(new_df, col)


new_df = one_hot_encoder(new_df, cat_cols, drop_first=True)

check_df(new_df)

# Drop unnecessary columns from X
drop_cols = [
    "arrival_date_year",
    "FUEL_PRCS",
    "New_arrival_month_numeric",
    "reservation_status_Check-Out",
    "reservation_status_No-Show"
]

new_df = new_df.drop(columns=drop_cols)


new_df.head()

# 'drop=True' eski indeksi bir sütun olarak saklamaz.
X_new= new_df.drop('adr', axis=1)

#new_df = new_df.reset_index(drop=True)
X_scaled = StandardScaler().fit_transform(X_new)
X = pd.DataFrame(X_scaled, columns=X_new.columns)

X.to_csv("X_regression.csv", index=False)

check_df(X)
X.shape
# Fit scaler on training data --> Bunu da .pkl formatında kaydet
scaler_adr = StandardScaler()
scaler_adr.fit(X_new)
# Save scaler to disk
joblib.dump(scaler_adr, "scaler_adr.pkl")


# Regresyon modellerini içe aktaralım
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
# from catboost import CatBoostRegressor
from sklearn.model_selection import cross_validate

def base_regressors(X, y, scoring="neg_mean_squared_error"):
    print("Base Regressors....")
    regressors = [('LR', LinearRegression()),
                  #('KNN', KNeighborsRegressor()),
                  #("SVR", SVR()),
                  #("CART", DecisionTreeRegressor()),
                  #("RF", RandomForestRegressor()),
                  #('GBM', GradientBoostingRegressor()),
                  ('XGBoost', XGBRegressor()),
                  #('LightGBM', LGBMRegressor())
    ]

    for name, regressor in regressors:
        print(f"[{name}] modeli çalıştırılıyor...")  # <-- Yeni satır
        cv_results = cross_validate(regressor, X, y, cv=3, scoring=scoring, return_train_score=True)
        print(f"########## {name} ##########")
        print(f"Train Score: {cv_results['train_score'].mean()}")
        print(f"Test Score: {cv_results['test_score'].mean()}")
        print("-" * 30)  # <-- Yeni satır


# Fonksiyonu tekrar çalıştırabilirsiniz
base_regressors(X, y)

## EN İYİ SONUCU KNN verdi son model üzerinden Knn optimizasyonu yapacağım .
y = df['adr']
# X = df.drop('adr', axis=1)
#y.head()
# KNN modelini tanımlayalım
knn_model = KNeighborsRegressor()

# Deneyebileceğimiz parametreleri bir sözlük içinde belirleyelim
knn_params = {'n_neighbors': range(2, 20),
              'weights': ['uniform', 'distance'],
              'p': [1, 2]} # 1: Manhattan, 2: Euclidean

# GridSearchCV nesnesini oluşturalım
# scoring parametresini önceki analizimizle tutarlı olması için 'neg_mean_squared_error' seçelim
knn_gs = GridSearchCV(knn_model,
                      knn_params,
                      cv=5, # 5 katlı çapraz doğrulama
                      n_jobs=-1, # Tüm işlemcileri kullan
                      verbose=1,
                      scoring='neg_mean_squared_error')

# Optimizasyon işlemini başlat
knn_gs.fit(X, y)

# En iyi parametreleri ve en iyi skoru yazdır
print("En iyi parametreler: ", knn_gs.best_params_)
print("En iyi skor: ", knn_gs.best_score_)

# Y verisi de scale edildiği için çıktı bu şekilde
###En iyi parametreler:  {'n_neighbors': 19, 'p': 1, 'weights': 'distance'}
##En iyi skor:  -0.4993667151243149

from sklearn.neighbors import KNeighborsRegressor


# GridSearchCV ile bulunan en iyi parametreleri kullanarak nihai modeli oluştur
final_knn_model = KNeighborsRegressor(n_neighbors=19, p=1, weights='distance')
final_knn_model.fit(X, y)
y.head()
print("Optimize edilmiş KNN modeli hazır.")

random_user = X.sample(1, random_state=42)
prediction = final_knn_model.predict(random_user)

# 3. Predict
print("Prediction (is_canceled):", prediction)

# Save the final_knn_model as a pickle file
joblib.dump(final_knn_model, "final_knn_model.pkl")
print("final_knn_model.pkl olarak kaydedildi.")




# XGBoost parametreleri
xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [5, 7],
    'learning_rate': [0.01, 0.2],
    'subsample': [0.8, 1.0]
}

xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)

xgb_gs = GridSearchCV(
    xgb_model,
    xgb_params,
    cv=5,
    n_jobs=-1,
    verbose=1,
    scoring='neg_mean_squared_error'
)

xgb_gs.fit(X, y)

print("En iyi XGBoost parametreleri: ", xgb_gs.best_params_)
print("En iyi XGBoost skor: ", xgb_gs.best_score_)

final_xgb_model = XGBRegressor(**xgb_gs.best_params_, objective='reg:squarederror', random_state=42)
final_xgb_model.fit(X, y)
y.head()
print("Optimize edilmiş XGBoost modeli hazır.")

random_user = X.sample(1, random_state=42)
prediction = final_xgb_model.predict(random_user)

# 3. Predict
print("Prediction (is_canceled):", prediction)

# Save the final_xgb_model as a pickle file
joblib.dump(final_xgb_model, "final_xgb_model.pkl")
print("final_xgb_model.pkl olarak kaydedildi.")

X.columns

X.tail(20)

new_df["GDP"].value_counts()

import joblib
joblib.dump(list(X.columns), "adr_feature_names.pkl")





def plotly_feature_importance(model, features, num=15):
    import plotly.express as px
    import pandas as pd

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
        title="XGBoost Feature Importance",
        color="Importance",
        color_continuous_scale="Blues",
        width=900,  # Daha geniş grafik
        height=600
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    fig.show()

# Kullanım:
plotly_feature_importance(final_xgb_model, X)