import os
import sys
import gc  # garbage collect
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import warnings

from pandas.plotting import register_matplotlib_converters
import lightgbm as lgbm

warnings.filterwarnings("ignore")  # ignore warnings will emerge in the program
# set maximun display columns and raws
pd.set_option("display.max_columns", 500)
pd.set_option("display.max_raws", 500)
register_matplotlib_converters()  # regiser converter by using matplotlib
sns.set()  # using seaborn style for ploting and showing
