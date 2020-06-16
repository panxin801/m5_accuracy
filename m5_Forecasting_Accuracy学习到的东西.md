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
y_true：真实值。[]
y_pred：预测值。
sample_weight：样本权值。
multioutput：多维输入输出，默认为’uniform_average’，计算所有元素的均方误差，返回为一个标量；也可选‘raw_values’，计算对应列的均方误差，返回一个与列数相等的一维数组。

[原文链接](https://blog.csdn.net/Dear_D/java/article/details/86136779)



7. from sklearn.preprocessing import LabelEncoder

[参考](https://blog.csdn.net/kancy110/article/details/75043202)



8. pd.set_option("display.max_columns", 500)

[参考](https://blog.csdn.net/xiongzaiabc/article/details/103023256)



9. sns.set()

[参考](https://blog.csdn.net/unixtch/article/details/78820654)



10. input_dir=f"{this_will_be_replaced}/data/dir"

python 3.6 新加入的f-string。

[f-string](https://www.cnblogs.com/insane-Mr-Li/p/12973941.html)

[定义字符串其他前缀](https://www.cnblogs.com/walo/p/10608436.html)



11. pd.read_csv()

返回的`dataframe[a][b]`假设read_csv读取的文件是[113,4]就是113行4列的数据。那么dataframe.shape=[113,4]但是dataframe["abc"]是找key="abc"的那个列

[read_csv](https://blog.csdn.net/weixin_37841694/article/details/80139479)



12. pd.DataFrame.pipe()

类似linux 中的管道|

[参考](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pipe.html)

[不同维度用不同函数](https://www.e-learn.cn/topic/3546511)



13. df.select_dtypes(include=["int"])

通过类型选择子数据框

```python
>>> df.select_dtypes(include="int")
   a
0  1
1  2
2  1
3  2
4  1
5  2
>>> df.select_dtypes(include="int").columns
Index(['a'], dtype='object')
>>> df
   a      b    c
0  1   True  1.0
1  2  False  2.0
2  1   True  1.0
3  2  False  2.0
4  1   True  1.0
5  2  False  2.0
>>> 

```

[参考](https://blog.csdn.net/xiezhen_zheng/article/details/83716267)



14. sklearn.LabelEncoder

[参考](https://blog.csdn.net/lw_power/article/details/82981122)

[官方解释](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)



15. pd.Series)()

pandas使用两种方法组织数据一种是df 另一种是Series类似numpy

[参考](https://blog.csdn.net/weixin_43868107/article/details/102631717)



16. df.melt

pandas做数据df的转换melt。与之相反的操作是pivot

![image-20200614162428365](m5_Forecasting_Accuracy学习到的东西.assets/image-20200614162428365.png)

[参考](https://blog.csdn.net/mingkoukou/article/details/82867218)

[写的更好的一个](https://blog.csdn.net/maymay_/article/details/80039677)



17. pandas df.merge

merge: 合并数据集， 通过left， right确定连接字段，默认是两个数据集相同的字段 

[参考](https://www.cnblogs.com/lijinze-tsinghua/p/9878649.html)



18. df.groupby

分组计算

![image-20200614210014220](m5_Forecasting_Accuracy学习到的东西.assets/image-20200614210014220.png)

[参考](https://www.cnblogs.com/keye/p/11153427.html)



19. df.transform

transform处理完了以后的输入输出的维度是一样的。groupby的结果transform之后就是和整个数据表的维度。

[参考]https://www.jianshu.com/p/509d7b97088c



20. rolling

就是一个滑动窗的概念其中的参数`window`表示这个窗里边的元素有多少个，`min_periods`表示这个窗口里最少需要多少个元素。pandas里边的`offset`是一个很有意思的东西，我还需要去确定以下这个东西怎么用。

[官网解释](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html)

[参考](https://blog.csdn.net/qifeidemumu/article/details/88748248)

[高阶参考](https://blog.csdn.net/wj1066/article/details/78853717)



21. shift

在某个方向平移数据，可以设置空数据的填充值

[参考](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.shift.html)



22. skew

计算数据的偏度`skewness`。实现是通过三阶标准化矩来实现的。

[参考](https://blog.csdn.net/u010665216/article/details/78591288)



23. kurt

数据的尖锐程度，和上边的skew都是常用的统计量

[参考](https://www.cnblogs.com/wyy1480/p/10474046.html)



24. pd.to_datetime()

解析时间格式

[参考](https://blog.csdn.net/Kwoky/article/details/91480035)



25. df.drop

用法：DataFrame.drop(labels=None,axis=0, index=None, columns=None, inplace=False)

参数说明：
labels 就是要删除的行列的名字，用列表给定
axis 默认为0，指删除行，因此删除columns时要指定axis=1；
index 直接指定要删除的行
columns 直接指定要删除的列
inplace=False，默认该删除操作不改变原数据，而是返回一个执行删除操作后的新dataframe；
inplace=True，则会直接在原数据上进行删除操作，删除后无法返回。

因此，删除行列有两种方式：
1）labels=None,axis=0 的组合
2）index或columns直接指定要删除的行或列

[原文链接](https://blog.csdn.net/songyunli1111/article/details/79306639)



26. python getattr()

python中用于获得数据的attr的函数

[参考](https://www.runoob.com/python/python-func-getattr.html)



27. python **args
28. df.iloc()

提取整行或者整列的数据，iloc[[1,2],[0，1]]就是提取[1,2]行和[0,1]列的数据。

[参考](https://blog.csdn.net/w_weiying/article/details/81411257)