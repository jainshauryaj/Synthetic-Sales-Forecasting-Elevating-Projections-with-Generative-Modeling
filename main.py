import pandas as pd
import numpy as np
from datetime import date
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn import preprocessing
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.models import Model
from keras import losses
from keras.layers import Layer, PReLU, BatchNormalization, Dropout, ReLU, LSTM, Input, Dense, Lambda
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from tqdm import tqdm
import os
import itertools
import statsmodels.api as sm
from statsmodels.tsa.api import SimpleExpSmoothing, Holt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from warnings import filterwarnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
filterwarnings("ignore")

df_train = pd.read_csv('train.csv')
df_store = pd.read_csv('stores.csv')
df_test = pd.read_csv('test.csv')

#train data preprocessing
df_train['Date'] = pd.to_datetime(df_train['Date'])
df_train['Year'] = df_train['Date'].dt.year
df_train['Month'] = df_train['Date'].dt.month
df_train['Day'] = df_train['Date'].dt.day
df_train['WeekOfYear'] = df_train['Date'].dt.weekofyear
df_train['weekday'] = df_train['Date'].dt.weekday
df_train['weekday'] = df_train['weekday'].astype(str)
df_train['weekday'] = df_train['weekday'].replace(['0','1','2','3','4','5','6'],['1','1','1','1','0','0','0'])

df_train = df_train.merge(df_store[['store_nbr','cluster']], on=['store_nbr'], how = 'left')
df_train = df_train[df_train['family'] == 'BEVERAGES'].reset_index(drop=True)
df_train = df_train.drop(columns=['id','store_nbr','onpromotion'])
df_train = df_train.groupby(['date','cluster','family']).agg({'sales':'sum'}).reset_index()
df_train = df_train[df_train['cluster'] == 1].reset_index(drop=True)

#test data preprocessing
df_test['Date'] = pd.to_datetime(df_test['Date'])
df_test['Year'] = df_test['Date'].dt.year
df_test['Month'] = df_test['Date'].dt.month
df_test['Day'] = df_test['Date'].dt.day
df_test['WeekOfYear'] = df_test['Date'].dt.weekofyear
df_test['weekday'] = df_test['Date'].dt.weekday
df_test['weekday'] = df_test['weekday'].astype(str)
df_test['weekday'] = df_test['weekday'].replace(['0','1','2','3','4','5','6'],['1','1','1','1','0','0','0'])

df_test = df_test.merge(df_store[['store_nbr','cluster']], on=['store_nbr'], how = 'left')
df_test = df_test[df_test['family'] == 'BEVERAGES'].reset_index(drop=True)
df_test = df_test.drop(columns=['id','store_nbr','onpromotion'])
df_test = df_test.groupby(['date','cluster','family']).agg({'sales':'sum'}).reset_index()
df_test = df_test[df_test['cluster'] == 1].reset_index(drop=True)

#feature engneering
sqrt_x_train = df_train.copy()
sqrt_x_train['sales'] = np.sqrt(sqrt_x_train['sales'])
box_x_train = sqrt_x_train.copy()

#anamoly detection (K-Means)
data = box_x_train[['sales', 'day']]
min_max_scaler = preprocessing.StandardScaler()
np_scaled = min_max_scaler.fit_transform(data)
data = pd.DataFrame(np_scaled)
n_cluster = range(1, 6)
kmeans = [KMeans(n_clusters=i, init='k-means++',n_init=20, random_state=42, algorithm='auto', max_iter=500).fit(data) for i in n_cluster]
scores = [kmeans[i].score(data) for i in range(len(kmeans))]
box_x_train['cluster_kmeans'] = kmeans[2].predict(data)

pair = {}
pair_list = []
pair[0] = box_x_train[box_x_train['cluster_kmeans'] == 0]['sales'].mean()
pair_list.append(box_x_train[box_x_train['cluster_kmeans'] == 0]['sales'].mean())
pair[1] = box_x_train[box_x_train['cluster_kmeans'] == 1]['sales'].mean()
pair_list.append(box_x_train[box_x_train['cluster_kmeans'] == 1]['sales'].mean())
pair[2] = box_x_train[box_x_train['cluster_kmeans'] == 2]['sales'].mean()
pair_list.append(box_x_train[box_x_train['cluster_kmeans'] == 2]['sales'].mean())
minimum = min(pair.values())
pair_list.remove(minimum)
minimum = [key for key, value in pair.items() if value == minimum][0]
maximum = max(pair.values())
pair_list.remove(maximum)
maximum = [key for key, value in pair.items() if value == maximum][0]
no_outleir_key = pair_list[0]
no_outleir_key = [key for key, value in pair.items() if value == no_outleir_key][0]

box_x_test = box_x_train[box_x_train['year'] > 2016]
box_x_train = box_x_train[box_x_train['year'] <= 2016]
box_x_train = box_x_train.drop(columns=['family','date','year','cluster','cluster_kmeans'])

gradient_check_callback = GradientCheckCallback()

# Create a MinMaxScaler
scaler = MinMaxScaler()

# Fit the scaler to your training data and transform it
box_x_train_normalized = scaler.fit_transform(box_x_train)

# Number of folds for cross-validation
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

input_dim = 4
latent_dim = 2

# Iterate over folds
for fold, (train_index, val_index) in enumerate(kf.split(box_x_train_normalized)):
    print(f"\nTraining Fold {fold + 1}/{num_folds}")

    # Split data into training and validation sets for this fold
    x_train_fold, x_val_fold = box_x_train_normalized[train_index], box_x_train_normalized[val_index]

    # Build and compile the model
    vae = build_vae_model()
    optimizer = Adam(learning_rate=0.001, clipvalue=0.5)
    vae.compile(optimizer=optimizer, loss=lambda x, y: y)

    # Train the model
    vae.fit(x_train_fold, x_train_fold, epochs=10, batch_size=32, validation_data=(x_val_fold, x_val_fold), callbacks=[gradient_check_callback])

# Generate synthetic data
num_samples = 200
latent_samples = np.abs(np.random.normal(size=(num_samples, input_dim)))
# Predict with the model
synthetic_data_normalized = vae.predict(latent_samples)
# Denormalize the predicted data
synthetic_data = scaler.inverse_transform(synthetic_data_normalized)
columns = ['sales', 'day', 'week', 'month']
synthetic_df = pd.DataFrame(synthetic_data, columns=columns)

# Your data preparation code here...
tmp = box_x_train.copy()
seasonal_period = get_seasonal_periods(tmp)

# Tune Holt Winter Model to get the right parm
holtiwinter_param, holtiwinter_mape = tune_HW(tmp,seasonal_period)

# Fit the model
model = ExponentialSmoothing(tmp, trend=holtiwinter_param['trend'], seasonal=holtiwinter_param['seasonal'], seasonal_periods=holtiwinter_param['seasonal_periods'])
model_fit = model.fit()

# Predict
forecast_df = model_fit.predict(start=len(tmp), end=len(tmp)+Num_f_count-1)
forecast_df = forecast_df - 1
forecast_df['reference'] = sku[1]
forecast_df['platform'] = sku[0]
forecast_df['err'] = np.round(holtiwinter_mape,2)
forecast_df['config'] = str(holtiwinter_param)
forecast_df['Model'] = 'Holt Winter'
output.append(forecast_df)

# Your post-processing code here...

final_result = pd.concat(output).drop_duplicates().reset_index(drop=True)
final_result.to_csv(raw_forecasted_file,index=False)  ## storing Raw Forecasted

# Build Final Forecast File
final_gb = final_result.groupby(['reference','platform'])
refined_forecast = []
for i in tqdm(final_gb.count().index):
    tmp = final_gb.get_group(i)
    if TimesSeries_Expansion != 'M':
        tmp = tmp.set_index('time')['fcst'].resample('M').sum().reset_index()    
    tmp['platform'] = i[1]
    tmp['reference'] = i[0]
    refined_forecast.append(tmp)
refined_forecast = pd.concat(refined_forecast).drop_duplicates().reset_index(drop=True)
refined_forecast  = refined_forecast.rename(columns={'time':'order_date','fcst':'quantity'})

# Your data merging code here...

refined_forecast.to_csv(output_forecasted,index=False)

print('Executed Sucessfully')