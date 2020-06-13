# Kaggle: M5_Forecasting_Accuracy学习到的东西

Author: Xin Pan

Date: 2020.06.12

----

## m5 baseline代码分析

1. import gc

python 的垃圾回收模块garbage collector。使用起来很简单，import gc .........gc.collect()就可以了。

[参考](https://www.jianshu.com/p/b6a20c812ce4)



2. import warnings

 warnings.filterwarnings("ignore")可以忽略当前程序中出现的warnings。这些东西可能当前看起来很碍眼。

[参考](https://blog.csdn.net/u013544265/article/details/28617527)



3. from pandas.plotting import register_matplotlib_converters

也就是说这个命令把pandas的格式转换到matplotlib进行实现。Register Pandas Formatters and Converters with matplotlib.

[pandas doc](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.plotting.register_matplotlib_converters.html)



4. import seaborn

seaborn 和matplotlib并成为世界两大图像现实库。seborn基于matplotlib，比matplotlib封装更高级，做出的图统计统计效果更好。

[什么是lightgbm](https://zhuanlan.zhihu.com/p/52583923)



5. import lightgbm as lgbm

一个python上边使用的gbm库

[参考](https://zhuanlan.zhihu.com/p/52583923)



6. from sklearn.metrics import mean_squared_error

计算均方误差回归损失
格式：
sklearn.metrics.mean_squared_error(y_true, y_pred, sample_weight=None, multioutput=’uniform_average’)
参数：
y_true：真实值。
y_pred：预测值。
sample_weight：样本权值。
multioutput：多维输入输出，默认为’uniform_average’，计算所有元素的均方误差，返回为一个标量；也可选‘raw_values’，计算对应列的均方误差，返回一个与列数相等的一维数组。

[原文链接](https://blog.csdn.net/Dear_D/java/article/details/86136779)



7. from sklearn.preprocessing import LabelEncoder

[参考](https://blog.csdn.net/kancy110/article/details/75043202)