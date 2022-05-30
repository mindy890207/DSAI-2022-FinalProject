import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import jpx_tokyo_market_prediction
pd.set_option('display.max_columns', 100)
import warnings, gc
import matplotlib.colors
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error,mean_absolute_error
from lightgbm import LGBMRegressor
from decimal import ROUND_HALF_UP, Decimal
warnings.filterwarnings("ignore")
import plotly.figure_factory as ff

init_notebook_mode(connected=True)
temp = dict(layout=go.Layout(font=dict(family="Franklin Gothic", size=12), width=800))
colors=px.colors.qualitative.Plotly

lgbm_params = {
    'task': 'train',
    'boosting_type': 'gbdt',  # gbdt - traditional Gradient Boosting Decision Tree
    'objective': 'regression',  # L2 loss
    'metric': 'rmse',
    'learning_rate': 0.05,
    'lambda_l1': 0.5,  # L1 regularization
    'lambda_l2': 0.5,  # L2 regularization
    'num_leaves': 10,
    'feature_fraction': 0.5,  # LightGBM will select 50% of features before training each tree
    'bagging_fraction': 0.5,  # LightGBM will select 50% part of data without resampling
    'bagging_freq': 5,  #  perform bagging at every k iteration
    'min_child_samples': 10,
    'seed': 42
}

file_path = '/kaggle/input/jpx-tokyo-stock-exchange-prediction/'
prices = pd.read_csv(Path(file_path, 'train_files/stock_prices.csv'))
stock_list = pd.read_csv(Path(file_path, 'stock_list.csv'))

prices.head()
prices.info(show_counts=True)

prices['Date'] = pd.to_datetime(prices['Date'])
min_date = prices['Date'].min()
prices['date_rank'] = (prices['Date'] - min_date).dt.days

stock_list['SectorName']=[i.rstrip().lower().capitalize() for i in stock_list['17SectorName']]
stock_list['Name']=[i.rstrip().lower().capitalize() for i in stock_list['Name']]
train_df = prices.merge(stock_list[['SecuritiesCode','Name','SectorName']], on='SecuritiesCode', how='left')
train_df['Year'] = train_df['Date'].dt.year
#Since some of the stocks were added in December 2020, use the data filtered after this date so that the data will consist of 231 days of stock prices for all 2,000 stocks.
train_df=train_df[train_df.Date>'2020-12-23']

def adjust_price(price):
    """
    Args:
        price (pd.DataFrame)  : pd.DataFrame include stock_price
    Returns:
        price DataFrame (pd.DataFrame): stock_price with generated AdjustedClose
    """
    # transform Date column into datetime
    price.loc[: ,"Date"] = pd.to_datetime(price.loc[: ,"Date"], format="%Y-%m-%d")

    def generate_adjusted_close(df):
        """
        Args:
            df (pd.DataFrame)  : stock_price for a single SecuritiesCode
        Returns:
            df (pd.DataFrame): stock_price with AdjustedClose for a single SecuritiesCode
        """
        # sort data to generate CumulativeAdjustmentFactor
        df = df.sort_values("Date", ascending=False)
        # generate CumulativeAdjustmentFactor
        df.loc[:, "CumulativeAdjustmentFactor"] = df["AdjustmentFactor"].cumprod()
        # generate AdjustedClose
        df.loc[:, "AdjustedClose"] = (
            df["CumulativeAdjustmentFactor"] * df["Close"]
        ).map(lambda x: float(
            Decimal(str(x)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
        ))
        # reverse order
        df = df.sort_values("Date")
        # to fill AdjustedClose, replace 0 into np.nan
        df.loc[df["AdjustedClose"] == 0, "AdjustedClose"] = np.nan
        # forward fill AdjustedClose
        df.loc[:, "AdjustedClose"] = df.loc[:, "AdjustedClose"].ffill()
        return df
    
    # generate AdjustedClose
    price = price.sort_values(["SecuritiesCode", "Date"])
    price = price.groupby("SecuritiesCode").apply(generate_adjusted_close).reset_index(drop=True)
    return price

prices=prices.drop('ExpectedDividend',axis=1).fillna(0)
ad_prices=adjust_price(prices)

features = ['Open', 'High', 'Low', 'Close', 'Volume', 'date_rank', 'SecuritiesCode']

ad_prices = ad_prices.dropna(subset=features)

target = ad_prices.pop('Target')
scaler = StandardScaler()
target = scaler.fit_transform(np.array(target).reshape(-1, 1)).ravel()
target = pd.Series(target, index = ad_prices.index)
target_mean = target.mean()

train_f, valid_f = train_test_split(ad_prices[features], test_size=0.2)
train_idx = train_f.index
valid_idx = valid_f.index
lgb_train = lgb.Dataset(train_f, target[train_idx])
lgb_valid = lgb.Dataset(valid_f, target[valid_idx], reference=lgb_train)

train_f.head()

model = lgb.train(
    lgbm_params,
    lgb_train,
    valid_sets=[lgb_train, lgb_valid],
    valid_names=['Train', 'Valid'],
    num_boost_round=2000,
    early_stopping_rounds=100,
    verbose_eval=100,
)

test_prices = pd.read_csv(Path(file_path, 'example_test_files/stock_prices.csv'))
test_prices['date_rank'] = (pd.to_datetime(test_prices['Date']) - min_date).dt.days

preds =  model.predict(test_prices[features], num_iteration=model.best_iteration)
preds

pd.Series(preds).fillna(target_mean).rank(ascending = False,method = 'first').astype(int)

env = jpx_tokyo_market_prediction.make_env()
iter_test = env.iter_test()

for (ad_prices, options, financials, trades, secondary_prices, sample_prediction) in iter_test:
    ad_prices['date_rank'] = (pd.to_datetime(ad_prices['Date']) - min_date).dt.days
    preds = model.predict(ad_prices[features], num_iteration=model.best_iteration)
    preds = np.squeeze(preds)
    print(preds)
    sample_prediction["Prediction"] = preds
    sample_prediction = sample_prediction.sort_values(by = "Prediction", ascending=False)
    sample_prediction.Rank = np.arange(0,2000)
    sample_prediction = sample_prediction.sort_values(by = "SecuritiesCode", ascending=True)
    sample_prediction.drop(["Prediction"],axis=1)
    submission = sample_prediction[["Date","SecuritiesCode","Rank"]]
    env.predict(submission)
    