# data analysis stack
from datetime import datetime, timezone
import pmdarima as pm
import warnings
import datetime
import json
import numpy as np
import pandas as pd
from pipelines.format import format_data_frame
import matplotlib.pyplot as plt
import seaborn as sns

import os
import re


import pickle
import json

import warnings
warnings.filterwarnings("ignore")


# Evaluation metrics
dir_path = os.path.dirname(os.path.realpath(__file__))
dirname, filename = os.path.split(os.path.abspath(__file__))  # type: ignore


# miscellaneous
warnings.filterwarnings("ignore")

main_result_date_time = datetime.datetime.now().now(timezone.utc).strftime("%H_%M_%S_%d_%m_%Y")
model_id = f'AutoARIMA_{main_result_date_time}'


meta_params = {
    'dataset_year_start': 2000,
    'drop_cols': ['DATE', 'month', 'SOUID', 'Q_TG'],
    'AutoARIMA': {
        'm': 24,  # Distancing for the season
        'd': 2,  # distancing for the trend
        'maxiter': 50,  # 1000 # The maximum number of function evaluations. Default is 50
        'seasonal': True,  # Must be TRUE if there is a seasonality_cycle value
        'start_p': 3,  # "Lag amount"
        'max_p': 3,
        'start_q': 12,  # "Moving average window"
        'max_q': 12,
        'stepwise': True,
        'trace': True,
        'test': 'adf',
        'n_jobs': 6,
    }
}


def get_log(array, base):
    return np.log(array) / np.log(base)  # = [3, 4]

# --- Prepare Data ---
df = pd.read_csv(f"{dir_path}/data/TG_STAID002759.txt",
                 header=14, index_col=1, parse_dates=True)

df = format_data_frame(df, set_index_time_step=False)

df = df[df.index.year > meta_params['dataset_year_start']]  # type: ignore

# ASSET NO MISSING DATES
# Make sure no missing values are in the time series
assert len(pd.date_range(df.index.min(), df.index.max()).difference(
    df.index)) == 0, "Missing values in date range"

# train_size = int(len(X) * 0.66)
train_size = int(len(df) * 0.99)
train, test = df[0:train_size], df[train_size:]

model_params = meta_params['AutoARIMA']

# --- Fit model ---
arima = pm.AutoARIMA(**model_params)

y = df['temp_c']

arima.fit(y)

dir_for_model = f'{dir_path}/artifacts/models/{model_id}/'
dir_for_model_images = f'{dir_path}/artifacts/models/{model_id}/images/'
path_for_model = f'{dir_for_model}{model_id}.sav'
path_for_model_meta_data = f'{dir_for_model}/{model_id}_meta_data.json'


logs = arima.summary()


result_meta = {
    **meta_params,
    'model': {
        **arima.get_params(),
        'report': list(
            map(
                lambda t: {'data': t.data, 'output_formats': t.output_formats,
                           'title': t.title}, logs.tables
            )
        )
    }
    # 'best_result': result_to_json(run_data['best_result']),
    # 'results': list(map(result_to_json,run_data['results']))
}

os.makedirs(os.path.dirname(dir_for_model), exist_ok=True)
os.makedirs(os.path.dirname(dir_for_model_images), exist_ok=True)


pickle.dump(arima, open(
    f'{path_for_model}', 'wb'), protocol=None, fix_imports=True)

# Serializing json
json_object = json.dumps(result_meta, indent=4)

# Writing to file
with open(path_for_model_meta_data, "w") as outfile:
    outfile.write(json_object)

# create some date values for the forecast horizon
index_vals = pd.date_range('1961-01-01', '1965-12-01', freq='MS')
index_vals = pd.date_range('2022-09-30', '2023-09-20', freq='MS')

## --- MAKE PREDICTIONS ---
y_forecast, ci = arima.predict(n_periods=12*5, return_conf_int=True)


fig_1 = plt.figure(1, figsize=(12, 6))  # type: ignore

fig1, ax1 = plt.subplots()
ax1.plot(y.index.values, y.values)
ax1.fill_between(index_vals.values,
                 ci[:, 0], ci[:, 1], alpha=0.7, color='lightgrey', label='95% CI')
ax1.plot(index_vals.values, y_forecast,
         label='Temperature Forecast', marker='.')
sns.despine()
ax1.legend()

fig1.savefig(f'{dir_for_model_images}/temp_forecast.png')
plt.close(fig1)  




