import os
import sys
import gc  # garbage collect
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
from src.TimeSeriesSpliter import CustomTimeSeriesSpliter

from pandas.plotting import register_matplotlib_converters
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_squared_log_error
import lightgbm as lgbm
from mlflow_extend import mlflow, plotting as mplt

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


def encoder_category(df, cols):  # encode input columns from string to int numeric
    for col in cols:
        le = LabelEncoder()
        # extrac this cols which are not null
        not_null = df[col][df[col].notnull()]
        df[col] = pd.Series(le.fit_transform(not_null),
                            index=not_null.index)  # 新的列的索引还是使用旧的列的索引编号
    return df


def extract_num(ser):  # extract num from column "d" which stores "d_1, d_2,...." return "1,2,3,....."
    return ser.str.extract(r"(\d+)").astype(np.int16)


def reshape_sales(sales, submission, pred_days, d_thresh=0, verbose=False):
    # melt sales data, get it ready for training
    id_columns = ["id", "item_id", "dept_id", "cat_id", "store_id",
                  "state_id"]  # all these columns all belongs to sales
    product = sales[id_columns]  # columns in sales relates with id_columns
    # new column value named demand and variable column named d
    sales = sales.melt(id_vars=id_columns, var_name="d", value_name="demand")
    sales = reduce_mem_usage(sales)
    '''
                                                                      id  item_id  dept_id  cat_id  store_id    state_id    demand
0  HOBBIES_1_001_CA_1_validation     1437        3       1         0           0                d_1              0
1  HOBBIES_1_002_CA_1_validation     1438        3       1         0           0                d_1              0
2  HOBBIES_1_003_CA_1_validation     1439        3       1         0           0                d_1              0
    '''

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


def show_cv_days(cv, x, dt_col, day_col):
    for ii, (tr, tt) in enumerate(cv.split(x)):
        print(f"----- Fold: ({ii + 1} / {cv.n_splits}) -----")
        tr_start = x.iloc[tr][dt_col].min()
        tr_end = x.iloc[tr][dt_col].max()
        tr_days = x.iloc[tr][day_col].max() - x.iloc[tr][day_col].min() + 1

        tt_start = x.iloc[tt][dt_col].min()
        tt_end = x.iloc[tt][dt_col].max()
        tt_days = x.iloc[tt][day_col].max() - x.iloc[tt][day_col].min() + 1

        df = pd.DataFrame(
            {
                "start": [tr_start, tt_start],
                "end": [tr_end, tt_end],
                "days": [tr_days, tt_days],
            },
            index=["train", "test"],
        )
        print(df)


def plot_cv_indices(cv, x, dt_col, lw=10):
    n_splits = cv.get_splits()
    fig, ax = plt.subplots(figsize=(20, n_splits))

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(x)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(x))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(
            x[dt_col],
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=plt.cm.coolwarm,
            vmin=-0.2,
            vmax=1.2,
        )

    # Formatting
    MIDDLE = 15
    LARGE = 20
    ax.set_xlabel("Datetime", fontsize=LARGE)
    ax.set_xlim([x[dt_col].min(), x[dt_col].max()])
    ax.set_ylabel("CV iteration", fontsize=LARGE)
    ax.set_yticks(np.arange(n_splits) + 0.5)
    ax.set_yticklabels(list(range(n_splits)))
    ax.invert_yaxis()
    ax.tick_params(axis="both", which="major", labelsize=MIDDLE)
    ax.set_title("{}".format(type(cv).__name__), fontsize=LARGE)
    plt.show()
    return ax


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


def train_lgb(bst_params, fit_params, x, y, cv, drop_when_train=None):
    models = []
    if drop_when_train is None:
        drop_when_train = []

    for idx_fold, (idx_train, idx_val) in enumerate(cv.split(x, y)):
        print(f"\n----- Fold: ({idx_fold + 1} / {cv.get_splits()}) -----\n")

        x_train, x_val = x.iloc[idx_train], x.iloc[idx_val]
        y_train, y_val = y.iloc[idx_train], y.iloc[idx_val]
        train_set = lgbm.Dataset(x_train.drop(
            drop_when_train, axis=1), y_train, categorical_feature=["item_id"])
        valid_set = lgbm.Dataset(x_val.drop(
            drop_when_train, axis=1), y_val, categorical_feature=["item_id"])

        model = lgbm.train(bst_params, train_set, valid_sets=[
                           train_set, valid_set], valid_names=["train", "valid"], **fit_params)
        models.append(model)

    # release resource
    del idx_train, idx_val, x_train, x_val, y_train, y_val
    gc.collect()
    return models


def make_submission(test, submission, pred_days):
    preds = test[["id", "date", "demand"]]
    preds = preds.pivot(index="id", columns="date",
                        values="demand").reset_index()
    preds.columns = ["id"] + ["F" + str(d + 1) for d in range(pred_days)]

    vals = submission[["id"]].merge(preds, how="inner", on="id")
    evals = submission[submission["id"].str.endswith("evaluation")]
    final = pd.concat([vals, evals])

    assert final.drop("id", axis=1).isnull().sum().sum() == 0
    assert final["id"].equals(submission["id"])

    final.to_csv("submission.csv", index=False)


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

    # stage 2
    day_col = "d"
    cv_params = {"n_splits": 3, "train_days": int(
        365 * 1.5), "test_days": pred_days, "day_col": day_col, "pred_days": pred_days}
    cv = CustomTimeSeriesSpliter(**cv_params)
    sample = data.iloc[::1000][[day_col, dt_col]].reset_index(drop=True)
    show_cv_days(cv, sample, dt_col, day_col)
    plot_cv_indices(cv, sample, dt_col)
    del sample
    gc.collect()

    features = [
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "snap_CA",
        "snap_TX",
        "snap_WI",
        "sell_price",
        # demand features
        "shift_t28",
        "shift_t29",
        "shift_t30",
        # std
        "rolling_std_t7",
        "rolling_std_t30",
        "rolling_std_t60",
        "rolling_std_t90",
        "rolling_std_t180",
        # mean
        "rolling_mean_t7",
        "rolling_mean_t30",
        "rolling_mean_t60",
        "rolling_mean_t90",
        "rolling_mean_t180",
        # min
        "rolling_min_t7",
        "rolling_min_t30",
        "rolling_min_t60",
        # max
        "rolling_max_t7",
        "rolling_max_t30",
        "rolling_max_t60",
        # others
        "rolling_skew_t30",
        "rolling_kurt_t30",
        # price features
        "price_change_t1",
        "price_change_t365",
        "rolling_price_std_t7",
        "rolling_price_std_t30",
        # time features
        "year",
        "quarter",
        "month",
        "week",
        "day",
        "dayofweek",
        "is_weekend",
    ]
    is_train = data["d"] < 1914

    # Attach "d" to X_train for cross validation.
    x_train = data[is_train][[day_col] + features].reset_index(drop=True)
    y_train = data[is_train]["demand"].reset_index(drop=True)
    x_test = data[~is_train][features].reset_index(drop=True)
    # keep these two columns to use later.
    id_date = data[~is_train][["id", "date"]].reset_index(drop=True)
    del data
    gc.collect()

    print("x_train shape:", x_train.shape)
    print("x_test shape:", x_test.shape)
    bst_params = {"boosting_type": "gbdt", "metric": "rmse",
                  "objective": "regression", "n_jobs": -1, "seed": 42, "learning_rate": 0.1,
                  "bagging_fraction": 0.75, "bagging_freq": 10, "colsample_bytree": 0.75}
    fit_params = {"num_boost_round": 100_000,
                  "early_stopping_rounds": 50, "verbose_eval": 100}
    models = train_lgb(bst_params, fit_params, x_train,
                       y_train, cv, drop_when_train=[day_col])
    del x_train, y_train
    gc.collect()

    imp_type = "gain"
    importances = np.zeros(x_test.shape[1])
    preds = np.zeros(x_test.shape[0])
    for model in models:
        preds += model.predict(x_test)
        importances += model.feature_importance(imp_type)
    preds = preds / cv.get_splits()
    importances = importances / cv.get_splits()

    with mlflow.start_run():
        mlflow.log_params_flatten(
            {"bst": bst_params, "fit": fit_params, "cv": cv_params})

    features = models[0].feature_name()
    fig = mplt.feature_importance(features, importances, imp_type, limit=30)
    plt.show()

    make_submission(id_date.assign(demand=preds), submission, pred_days)


if __name__ == "__main__":
    main()
