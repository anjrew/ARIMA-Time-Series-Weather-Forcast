# data analysis stack
import warnings
import datetime
import json
import numpy as np
import pandas as pd
from format import format_data_frame

import os
import re


import pickle
import json

# machine learning stack
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import VotingRegressor, StackingRegressor, RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV


from statsmodels.tsa.seasonal import seasonal_decompose

# Evaluation metrics
from sklearn.model_selection import TimeSeriesSplit

from datetime import datetime, timezone
import hashlib


dir_path = os.path.dirname(os.path.realpath(__file__))
dirname, filename = os.path.split(os.path.abspath(__file__))  # type: ignore


# miscellaneous
warnings.filterwarnings("ignore")

main_result_date_time = datetime.now(timezone.utc).strftime("%H_%M_%S_%d_%m_%Y")
model_id = f'Regressor_{main_result_date_time}'

meta_params = {
    'ts_split': 5,
    'lags_to_test':  list(range(2,5))
}


df = pd.read_csv(f"{dir_path}/data/TG_STAID002759.txt",
                 header=14, index_col=1, parse_dates=True)

df = format_data_frame(df, set_index_time_step=False)

df = df[df.index.year > 1945]  # type: ignore

# ASSET NO MISSING DATES
# Make sure no missing values are in the time series
assert len(pd.date_range(df.index.min(), df.index.max()).difference(
    df.index)) == 0, "Missing values in date range"


# model, fit, period, two_sided, extrapolate_trend
sd = seasonal_decompose(df['temp_c'], model='additive',
                        period=365)  # Period in days

# Add remainder to the dataframe
df = df.join(sd.resid)
df = df.dropna()
df = df.rename(columns={'resid': 'remainder'})


def get_estimator_details(estimators_in) -> list:
    """Gets details of estimators

    Args:
        estimators_in (_type_): A Grid search with a classification model
    """
    details = []
    for est in estimators_in:
        details.append((est[0], est[1].__class__.__name__))
    return details

def result_to_json(result):
    return {
        **result,
        'predictor':
        {
            'type': result['predictor']['type'],
            # 'details': result['predictor']['details']
        }
    }


results = [
]

seasonal_dummies = pd.get_dummies(df.index.month,  # type: ignore
                                  prefix='month',
                                  drop_first=True).set_index(df.index)
df = df.join(seasonal_dummies)

df['time_step'] = range(len(df))

df = df.reset_index()
df = df.set_index('time_step')


for lag_amount in meta_params['lags_to_test']:
    # maximum interval to consider
    lags = [i+1 for i in range(lag_amount)]

    for lag in lags:
        column_name = 'lag_' + str(lag)
        df[column_name] = df['remainder'].shift(lag)

    df = df.dropna()

    gs_params = {
        # RANDOM FOREST
        'rf__n_estimators': [50, 100],  # range(30, 100)
        'rf__max_depth': [2 , 5 , 10], #
        'rf__max_features': ['auto'] ,#['auto'],  # ['auto', 'sqrt', 'log2']
        'xg__nthread':[4],
        'xg__n_estimators': [500, 1100],
        'xg__max_depth': [2,10],
    }

    ts_split = TimeSeriesSplit(n_splits=meta_params['ts_split'])

    columns = df.columns
    # lag_matcher = re.compile('^((?!lag).)*$')
    col_matcher = re.compile('(lag_\d|month_\d|time_step)')
    cols_to_use = [l for l in columns if col_matcher.match(l)]

    # model for remainder
    X = df[cols_to_use]
    y = df['temp_c']

    splits = ts_split.split(X, y)

    cv = list(splits)

    grid = GridSearchCV(
        estimator=VotingRegressor(
            estimators=[
                ('rf',  RandomForestRegressor(n_jobs=6)),
                ('lg',  LinearRegression(n_jobs=6)),
                ('xg',  XGBRegressor(n_jobs=6)),
            ],
            n_jobs=6
        ),
        param_grid=gs_params,
        cv=cv,
        verbose=3
    )

    grid.fit(X, y)

    date_time = datetime.now(timezone.utc).strftime("%H_%M_%S_%d_%m_%Y")
    best_score = grid.best_score_

    predictor = grid.best_estimator_
    estimator_id = f'l_{lag_amount}_{predictor.__class__.__name__}_{date_time}_{int(round(best_score, 5) * 10000)}'
    estimators = predictor.estimators  # type: ignore
    remainder_test_meta_data = {
        'id': estimator_id,
        'num_lags': lag_amount,
        'cv_score': best_score,
        'timestamp': date_time,
        'estimators': get_estimator_details(estimators),
        'estimators_hash': hashlib.sha256(json.dumps(get_estimator_details(estimators)).encode('utf-8')).hexdigest(),
        'best_params_': grid.best_params_,
        'best_params_hash': hashlib.sha256(json.dumps(grid.best_params_).encode('utf-8')).hexdigest(),
        'predictor': {
            'model': predictor,
            'type': predictor.__class__.__name__,
            'details': predictor.__dict__,
        }
    }
    
    dir_for_model = f'{dir_path}/artifacts/models/{model_id}/estimators/{estimator_id}/'
    path_for_model = f'{dir_for_model}{estimator_id}.sav'
    path_for_model_meta_data = f'{dir_for_model}/{estimator_id}_meta_data.json'

    os.makedirs(os.path.dirname(dir_for_model), exist_ok=True)

    pickle.dump(grid, open(
        f'{path_for_model}', 'wb'), protocol=None, fix_imports=True)

    # Serializing json
    json_object = json.dumps(result_to_json(remainder_test_meta_data), indent=4)

    # Writing to file
    with open(path_for_model_meta_data, "w") as outfile:
        outfile.write(json_object)

    results.append(remainder_test_meta_data)


best_score = 0
best_result = None
for result in results:
    if best_result is None or result['cv_score'] > best_score:
        best_score = result['cv_score']
        best_result = result

assert best_result is not None, 'No best result was found'

run_data = {
    'best_score': best_score,
    'best_result': best_result,
    'results': results
}

dir_for_model = f'{dir_path}/artifacts/models/{model_id}/'
path_for_model = f'{dir_for_model}{model_id}.sav'
path_for_model_meta_data = f'{dir_for_model}/{model_id}_meta_data.json'

os.makedirs(os.path.dirname(dir_for_model), exist_ok=True)

pickle.dump(best_result['predictor']['model'], open(
    f'{path_for_model}', 'wb'), protocol=None, fix_imports=True)

result_meta = {
    **meta_params,
    **run_data, 
    'best_result': result_to_json(run_data['best_result']),
    'results': list(map(result_to_json,run_data['results']))
}

# Serializing json
json_object = json.dumps(result_meta, indent=4)

# Writing to file
with open(path_for_model_meta_data, "w") as outfile:
    outfile.write(json_object)


print(f'\nScore:', best_score)
