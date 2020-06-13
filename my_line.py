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
import lightgbm as lgbm

warnings.filterwarnings("ignore")  # ignore warnings will emerge in the program
# set maximun display columns and raws
pd.set_option("display.max_columns", 500)
pd.set_option("display.max_rows", 500)
register_matplotlib_converters()  # regiser converter by using matplotlib
sns.set()  # using seaborn style for ploting and showing


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
        not_null = df[col][df[col].notnull()]
        df[col] = pd.Series(le.transform(not_null), index=not_null.index)
    return df


def main():
    calendar, prices, sales, submission = read_data()
    num_items = sales.shape[0]
    pred_days = submission.shape[1] - 1  # 28

# not sure below TO be watched
    calendar = encoder_category(
        calendar, ["event_name_1", "event_type_1", "event_name_2", "event_type_2"
                   ]).pipe(reduce_mem_usage)
    sales = encoder_category(
        sales,
        ["item_id", "dept_id", "cat_id", "store_id", "state_id"],
    ).pipe(reduce_mem_usage)
    prices = encoder_category(prices,
                              ["item_id", "store_id"]).pipe(reduce_mem_usage)


if __name__ == "__main__":
    main()
