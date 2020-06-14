import os
import sys
import gc  # garbage collect
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings

from pandas.plotting import register_matplotlib_converters
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_squared_log_error
import lightgbm as lgbm

warnings.filterwarnings("ignore")  # ignore warnings will emerge in the program
# set maximun display columns and raws
pd.set_option("display.max_columns", 500)
pd.set_option("display.max_rows", 500)
register_matplotlib_converters()  # regiser converter by using matplotlib
sns.set()  # using seaborn style for ploting and showing
verbose = True  # set log level


def read_data():
    input_dir = r"/home/panxin/data"
    # This is a f-string from python 3.6
    input_dir = f"{input_dir}/m5_forecasting_accuracy/"
    print("Reading files")
    # [1969,14]. set header=0 means raw[0] is the title
    calendar = pd.read_csv(f"{input_dir}/calendar.csv",
                           header=0).pipe(reduce_mem_usage)
    prices = pd.read_csv(f"{input_dir}/sell_prices.csv").pipe(reduce_mem_usage)
    sales = pd.read_csv(
        f"{input_dir}/sales_train_validation.csv").pipe(reduce_mem_usage)
    submission = pd.read_csv(
        f"{input_dir}/sample_submission.csv").pipe(reduce_mem_usage)

    print("sales shape:", sales.shape)
    print("prices shape:", prices.shape)
    print("calendar shape:", calendar.shape)
    print("submission shape:", submission.shape)
    return calendar, prices, sales, submission


# `df` here means the input Data Frame  from pandas
# This func is used for reduce memory usage
def reduce_mem_usage(df, verbose=False):
    # ** means operation power(1024,2)
    start_mem = df.memory_usage().sum() / 1024 ** 2
    int_columns = df.select_dtypes(
        include="int").columns  # return column  names
    float_columns = df.select_dtypes(include="float").columns
    # downcast dtypes
    for col in int_columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in float_columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print("Memory usage  decreased from {:5.2f}M to {:5.2}M".format(
            start_mem, end_mem))
    return df


def encoder_category(df, cols):  # encode input columns
    for col in cols:
        le = LabelEncoder()
        # extrac this cols which are not null
        not_null = df[col][df[col].notnull()]
        df[col] = pd.Series(le.fit_transform(not_null), index=not_null.index)
    return df


def extract_num(ser):  # extract num from column "d" which stores "d_1, d_2,...." return "1,2,3,....."
    return ser.str.extract(r"(\d+)").astype(np.int16)


def reshape_sales(sales, submission, pred_days, d_thresh=0, verbose=False):
    # melt sales data, get it ready for training
    id_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    product = sales[id_columns]  # columns in sales relates with id_columns
    # new column value named demand and variable column named d
    sales = sales.melt(id_vars=id_columns, var_name="d", value_name="demand")
    sales = reduce_mem_usage(sales)

    # separate test dataframes.
    vals = submission[submission["id"].str.endswith("validation")]
    evals = submission[submission["id"].str.endswith("evaluation")]

    # change column name
    # change vals.columns to [id,d_1914,......d_1941]  which are [id, F1,F2,......,F28] before
    vals.columns = ["id"] + [f"d_{d}" for d in range(1914, 1914 + pred_days)]
    evals.columns = ["id"] + [f"d_{d}" for d in range(1942, 1942 + pred_days)]

    # merge with product table
    evals["id"] = evals["id"].str.replace("_evaluation", "_validation")
    vals = vals.merge(product, how="left", on="id")
    evals = evals.merge(product, how="left", on="id")
    evals["id"] = evals["id"].str.replace("_validation", "_evaluation")

    if verbose:
        print(evals.head(5))
        print(vals.head(5))

    # generate data for training part2
    vals = vals.melt(id_vars=id_columns, var_name="d", value_name="demand")
    evals = evals.melt(id_vars=id_columns, var_name="d", value_name="demand")

    sales["part"] = "train"  # add a new column "part"
    vals["part"] = "validation"
    evals["part"] = "evaluation"

    data = pd.concat([sales, vals, evals], axis=0)
    del sales, vals, evals

    data["d"] = extract_num(data["d"])
    data = data[data["d"] >= d_thresh]

    data = data[data["part"] != "evaluation"]
    gc.collect()

    if verbose:
        print("data")
        # display(data)

    return data


def merge_calendar(data, calendar):
    calendar = calendar.drop(["weekday", "wday", "month", "year"], axis=1)
    return data.merge(calendar, how="left", on="d")


def merge_prices(data, prices):
    return data.merge(prices,
                      how="left",
                      on=["store_id", "item_id", "wm_yr_wk"])


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def add_demand_features(df, pred_days):
    for diff in [0, 1, 2]:
        shift = pred_days + diff
        # This operation  need TO be watched not clear by now
        df[f"shift_t{shift}"] = df.groupby(
            ["id"])["demand"].transform(lambda x: x.shift(shift))

    for window in [7, 30, 60, 90, 180]:
        df[f"rolling_std_t{window}"] = df.groupby(["id"])["demand"].transform(
            lambda x: x.shift(pred_days).rolling(window).std())

    for window in [7, 30, 60, 90, 180]:
        df[f"rolling_mean_t{window}"] = df.groupby(["id"])["demand"].transform(
            lambda x: x.shift(pred_days).rolling(window).mean())

    for window in [7, 30, 60]:
        df[f"rolling_min_t{window}"] = df.groupby(["id"])["demand"].transform(
            lambda x: x.shift(pred_days).rolling(window).min())

    for window in [7, 30, 60]:
        df[f"rolling_max_t{window}"] = df.groupby(["id"])["demand"].transform(
            lambda x: x.shift(pred_days).rolling(window).max())

    df["rolling_skew_t30"] = df.groupby([
        "id"
    ])["demand"].transform(lambda x: x.shift(pred_days).rolling(30).skew())
    df["rolling_kurt_t30"] = df.groupby([
        "id"
    ])["demand"].transform(lambda x: x.shift(pred_days).rolling(30).kurt())
    return df


def add_price_features(df):
    df["shift_price_t1"] = df.groupby(
        ["id"])["sell_price"].transform(lambda x: x.shift(1))
    df["price_change_t1"] = (df["shift_price_t1"] -
                             df["sell_price"]) / (df["shift_price_t1"])
    df["rolling_price_max_t365"] = df.groupby([
        "id"
    ])["sell_price"].transform(lambda x: x.shift(1).rolling(365).max())
    df["price_change_t365"] = (df["rolling_price_max_t365"] - df["sell_price"]
                               ) / (df["rolling_price_max_t365"])

    df["rolling_price_std_t7"] = df.groupby(
        ["id"])["sell_price"].transform(lambda x: x.rolling(7).std())
    df["rolling_price_std_t30"] = df.groupby(
        ["id"])["sell_price"].transform(lambda x: x.rolling(30).std())
    return df.drop(["rolling_price_max_t365", "shift_price_t1"], axis=1)


def add_time_features(df, dt_col):
    df[dt_col] = pd.to_datetime(df[dt_col])
    attrs = [
        "year",
        "quarter",
        "month",
        "week",
        "day",
        "dayofweek",
    ]

    for attr in attrs:
        dtype = np.int16 if attr == "year" else np.int8
        df[attr] = getattr(df[dt_col].dt, attr).astype(dtype)

    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(np.int8)
    return df


def main():
    calendar, prices, sales, submission = read_data()
    num_items = sales.shape[0]
    pred_days = submission.shape[1] - 1  # 28

    # encoder calendar and sales cols into numeric
    calendar = encoder_category(
        calendar, ["event_name_1", "event_type_1", "event_name_2", "event_type_2"
                   ]).pipe(reduce_mem_usage, verbose)
    sales = encoder_category(
        sales,
        ["item_id", "dept_id", "cat_id", "store_id", "state_id"],
    ).pipe(reduce_mem_usage, verbose)
    prices = encoder_category(prices,
                              ["item_id", "store_id"]).pipe(reduce_mem_usage, verbose)

    data = reshape_sales(sales, submission, pred_days, d_thresh=1941 -
                         int(365 * 2), verbose=False)  # d_thresh why is this
    del sales

    calendar["d"] = extract_num(calendar["d"])
    data = merge_calendar(data, calendar)
    data = merge_prices(data, prices)
    del calendar, prices
    gc.collect()
    data = reduce_mem_usage(data)
    data = add_demand_features(data, pred_days).pipe(reduce_mem_usage)
    data = add_price_features(data).pipe(reduce_mem_usage)
    dt_col = "date"
    data = add_time_features(data, dt_col).pipe(reduce_mem_usage)
    data = data.sort_values("date")

    print("start date:", data[dt_col].min())
    print("end date:", data[dt_col].max())
    print("data shape:", data.shape)


if __name__ == "__main__":
    main()
