import pandas as pd
import numpy as np 
import datetime,time
from sklearn.preprocessing import PolynomialFeatures



def func_num_agg_by_id(df:pd.DataFrame,id=[],num_feature=[]):
    """
  
    根据指定的 [id,] 比如（客户编号、卡号）列对对应的 数值变量 [num_feature] 进行聚合 比如进行聚集特征统计：

    参数:
        df : 原始数据框 DataFrame  
        id :就是 主键列表 
        num_feature: 分类变量名列表
    输出:
        输出新的DataFrame,并且自动命名
        df 
    """

    def q1(x):
        return x.quantile(0.25)

    def q3(x):
        return x.quantile(0.75)

    def kurt(x):
        return x.kurt()

    def cv(x):
        return x.std() / (x.mean() + 10 ** -8)  # 变异系数
    # 根据id 进行删除
    funcs = ['count', 'min', 'mean', 'median', 'max', 'sum', 'std', 'var', 'sem', 'skew', kurt, q1, q3]

    # 根据id 进行增删改查
    group_cols = id
    # 根据数值变量进行统计
    agg_cols = num_feature


    df_agg = df.groupby(group_cols)[agg_cols].agg(funcs)
    df_agg.columns = ['_'.join(i) for i in df_agg.columns]

    for col in agg_cols:
        # 极差
        df_agg[col + '_max_min'] = df_agg[col + '_max'] - df_agg[col + '_min']
        # 三份位数-1分位数
        df_agg[col + '_q3_q1'] = df_agg[col + '_q3'] - df_agg[col + '_q1']
        # 变异系数
        df_agg[col + '_cv'] = df_agg[col + '_std'] / (df_agg[col + '_mean'] + 10 ** -8)

        # 变异系数导数
        df_agg[col + '_cv_reciprocal'] = 1 / (df_agg[col + '_cv'] + 10 ** -8)

    return df_agg.reset_index()


# 时间序列特征统计 1  
def func_by_timeline(df:pd.DataFrame,id:str,dt:datetime.datetime,cat_feature=[] ,num_feature=[]):
    """
    对数值变量 进行聚合操作

    参数:
        df : 原始数据框 DataFrame  
        id :就是 主键列表
        dt :日期变量列表
        cat_feature: 分类变量名列表
        num_feature: 数值变量名列表
    输出:
        输出新的DataFrame,并且自动命名
        df 
    """
    return df 


# 根据时间窗进行统计  
def func_by_time_window(df:pd.DataFrame,id:str,dt:datetime.datetime,num_feature=[]):
    """
    对数值变量 进行 时间窗 滑动统计

    参数:
        df : 原始数据框 DataFrame  
        id :就是 主键列表
        dt :日期变量列表
        num_feature: 数值变量名列表
    输出:
        输出新的DataFrame,并且自动命名
        df 
    """
    return df 




def get_feats_row_poly(self, df, feats=None, degree=2, return_df=True):
    """PolynomialFeatures
    :param data: np.array or pd.DataFrame
    :param feats: columns names
    :param degree:
    :return: df
    """
    if feats is None:
        feats = df.columns

    poly = PolynomialFeatures(degree, include_bias=False)
    df = poly.fit_transform(df[feats])
    self.feat_poly_cols = poly.get_feature_names(feats)

    if return_df:
        df = pd.DataFrame(df, columns=self.feat_poly_cols)
    return df





















