#!/usr/bin/env python
# coding: utf-8

# ## 单特征衍生

# ### 1 单变量数值型 聚合函数

# In[1]:


import pandas as pd
import numpy as np 
import datetime,time
from sklearn.preprocessing import PolynomialFeatures


def func_cols_agg_by_id(df:pd.DataFrame,id=[],agg_cols=[]):
    """
    对数值变量 进行聚合操作 并求出聚合数据

    参数:
        df : 原始数据框 DataFrame  
        id :就是 主键
        dt :日期变量
        cat_feature: 分类变量名列表
        num_feature: 数值变量名列表
        verbose: 日志打印详细程度
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
    agg_cols = agg_cols


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


# ### 单变量-num  时序特征       差分，时序，lag

# In[ ]:





# ### 单变量-cat            lag

# In[ ]:





# ### 单变量 cat    target_encode    count_encode          

# In[ ]:





# ### 双变量--num-num                +-*/   多项式                                                                               

# ### 双变量--cat-cat               笛卡尔积、交、并、补

# In[ ]:


def binary_cross_combination(primary_key: id, col_names: list, features: pd.DataFrame, one_hot: bool = True):
    """
    分类特征两两组合交叉衍生函数--笛卡尔积
    primary_key: 主键名
    :param col_names: 参与交叉衍生的列名称
    :param features: 原始数据集
    :param one_hot: 是否进行One-Hot编码，默认会对组合后的特征进行独热编码
    :return:
    """
    # 创建空列表存储器
    col_names_new_l = []
    features_new_l = []

    # 提取需要进行交叉组合的特征
    features = features[col_names]

    # 逐个创造新特征名称、新特征
    for col_index, col_name in enumerate(col_names):
        for col_sub_index in range(col_index + 1, len(col_names)):
            new_names = col_name + '_' + col_names[col_sub_index]
            col_names_new_l.append(new_names)

            new_df = pd.Series(
                data=features[col_name].astype('str') + '_' + features[col_names[col_sub_index]].astype('str'),
                name=col_name)
            features_new_l.append(new_df)

    # 拼接新特征矩阵
    features_new = pd.concat(features_new_l, axis=1)
    features_new.columns = col_names_new_l
    col_names_new = col_names_new_l

    # 对新特征矩阵进行独热编码
    if one_hot:
        enc = OneHotEncoder()
        enc.fit_transform(features_new)
        col_names_new = cate_col_name(enc, col_names_new_l, skip_binary=True)
        features_new = pd.DataFrame(enc.fit_transform(features_new).toarray(), columns=col_names_new)
        df_temp = pd.concat([features, features_new], axis=1)
        
    return df_temp


# ### 双变量--cat-num       

# In[ ]:


def binary_group_statistics(primary_key: cst_Id,
                            features: pd.DataFrame,
                            col_num: list = None,
                            col_cat: list = None,
                            num_stat: list = ['mean', 'var', 'max', 'min', 'skew', 'median'],
                            cat_stat: list = ['mean', 'var', 'max', 'min', 'median', 'count', 'nunique'],
                            quantile: bool = True):
    """
    双特征分组统计特征衍生函数

    :param primary_key: 主键
    :param features: 原始数据集
    :param col_num: 参与衍生的连续型特征
    :param col_cat: 参与衍生的离散型特征
    :param num_stat: 连续特征分组统计指标
    :param cat_stat: 离散特征分组统计指标
    :param quantile: 是否计算分位数

    :return：交叉衍生后的新特征和新特征的名称
    """

    # 当输入的特征有连续型特征时
    if col_num is not None:
        aggs_num = {}

        col_names = col_num
        # 创建agg方法所需字典
        for col in col_num:
            aggs_num[col] = num_stat

        # 创建衍生特征名称列表
        cols_num = [primary_key]
        for key in aggs_num.keys():
            cols_num.extend([key + '_' + primary_key + '_' + stat for stat in aggs_num[key]])

        # 创建衍生特征df
        features_num_new = features[col_num + [primary_key]].groupby(primary_key).agg(aggs_num).reset_index()
        features_num_new.columns = cols_num

        # 当输入的特征有连续型也有离散型特征时
        if col_cat is not None:
            aggs_cat = {}
            col_names = col_num + col_cat

            # 创建agg方法所需字典
            for col in col_cat:
                aggs_cat[col] = cat_stat

            # 创建衍生特征名称列表
            cols_cat = [primary_key]
            for key in aggs_cat.keys():
                cols_cat.extend([key + '_' + primary_key + '_' + stat for stat in aggs_cat[key]])

                # 创建衍生特征df
            features_cat_new = features[col_cat + [primary_key]].groupby(primary_key).agg(aggs_cat).reset_index()
            features_cat_new.columns = cols_cat

            # 合并连续特征衍生结果与离散特征衍生结果
            df_temp = pd.merge(features_num_new, features_cat_new, how='left', on=primary_key)
            features_new = pd.merge(features[primary_key], df_temp, how='left', on=primary_key)
            features_new.loc[:, ~features_new.columns.duplicated()]
            col_names_new = cols_num + cols_cat
            col_names_new.remove(primary_key)
            col_names_new.remove(primary_key)

        # 当只有连续特征时
        else:
            # merge连续特征衍生结果与原始数据，然后删除重复列
            features_new = pd.merge(features[primary_key], features_num_new, how='left', on=primary_key)
            features_new.loc[:, ~features_new.columns.duplicated()]
            col_names_new = cols_num
            col_names_new.remove(primary_key)

    # 当没有输入连续特征时
    else:
        # 但存在分类特征时，即只有分类特征时
        if col_cat is not None:
            aggs_cat = {}
            col_names = col_cat

            for col in col_cat:
                aggs_cat[col] = cat_stat

            cols_cat = [primary_key]
            for key in aggs_cat.keys():
                cols_cat.extend([key + '_' + primary_key + '_' + stat for stat in aggs_cat[key]])

            features_cat_new = features[col_cat + [primary_key]].groupby(primary_key).agg(aggs_cat).reset_index()
            features_cat_new.columns = cols_cat

            features_new = pd.merge(features[primary_key], features_cat_new, how='left', on=primary_key)
            features_new.loc[:, ~features_new.columns.duplicated()]
            col_names_new = cols_cat
            col_names_new.remove(primary_key)

    if quantile:
        # 定义四分位计算函数
        def q1(x):
            """
            下四分位数
            """
            return x.quantile(0.25)

        def q2(x):
            """
            上四分位数
            """
            return x.quantile(0.75)

        agg_name = {}
        for col in col_names:
            agg_name[col] = ['q1', 'q2']

        cols = [primary_key]
        for key in agg_name.keys():
            cols.extend([key + '_' + primary_key + '_' + stat for stat in agg_name[key]])

        aggs = {}
        for col in col_names:
            aggs[col] = [q1, q2]

        features_temp = features[col_names + [primary_key]].groupby(primary_key).agg(aggs).reset_index()
        features_temp.columns = cols

        features_new = pd.merge(features_new, features_temp, how='left', on=primary_key)
        features_new.loc[:, ~features_new.columns.duplicated()]
        col_names_new = col_names_new + cols
        col_names_new.remove(primary_key)

    features_new.drop([primary_key], axis=1, inplace=True)

    return features_new, col_names_new


# In[ ]:





# ## 双特征衍生

# In[ ]:


def binary_polynomial_features(col_names: list, degree: int, features: pd.DataFrame):
    """
    连续特征两特征多项式衍生函数

    :param col_names: 参与交叉衍生的列名称
    :param degree: 多项式最高阶
    :param features: 原始数据集

    :return：交叉衍生后的新特征和新列名称
    """

    # 创建空列表存储器
    col_names_new_l = []
    features_new_l = []

    # 提取需要进行多项式衍生的特征
    features = features[col_names]

    # 逐个进行多项式特征组合
    for col_index, col_name in enumerate(col_names):
        for col_sub_index in range(col_index + 1, len(col_names)):
            col_temp = [col_name] + [col_names[col_sub_index]]
            array_new_temp = PolynomialFeatures(degree=degree, include_bias=False).fit_transform(features[col_temp])
            features_new_l.append(pd.DataFrame(array_new_temp[:, 2:]))

            # 逐个创建衍生多项式特征的名称
            for deg in range(2, degree + 1):
                for i in range(deg + 1):
                    col_name_temp = col_temp[0] + '**' + str(deg - i) + '*' + col_temp[1] + '**' + str(i)
                    col_names_new_l.append(col_name_temp)

    # 拼接新特征矩阵
    features_new = pd.concat(features_new_l, axis=1)
    features_new.columns = col_names_new_l
    col_names_new = col_names_new_l

    return features_new, col_names_new


# In[ ]:


def group_statistics_extension(col_names: list, key_col: str, features: pd.DataFrame):
    """
    双特征分组统计二阶特征衍生函数

    :param col_names: 参与衍生的特征
    :param key_col: 分组参考的关键特征
    :param features: 原始数据集

    :return：交叉衍生后的新特征和新列名称
    """

    # 定义四分位计算函数
    def q1(x):
        """
        下四分位数
        """
        return x.quantile(0.25)

    def q2(x):
        """
        上四分位数
        """
        return x.quantile(0.75)

    # 第一轮特征衍生

    # 先定义用于生成列名称的aggs
    aggs = {}
    for col in col_names:
        aggs[col] = ['mean', 'var', 'median', 'q1', 'q2']
    cols = [key_col]
    for key in aggs.keys():
        cols.extend([key + '_' + key_col + '_' + stat for stat in aggs[key]])

    # 再定义用于进行分组汇总的aggs
    aggs = {}
    for col in col_names:
        aggs[col] = ['mean', 'var', 'median', q1, q2]

    features_new = features[col_names + [key_col]].groupby(key_col).agg(aggs).reset_index()
    features_new.columns = cols

    col_name_temp = [key_col]
    col_name_temp.extend(col_names)
    features_new = pd.merge(features[col_name_temp], features_new, how='left', on=key_col)
    features_new.loc[:, ~features_new.columns.duplicated()]
    col_names_new = cols
    col_names_new.remove(key_col)
    col1 = col_names_new.copy()

    # 第二轮特征衍生

    # 流量平滑特征
    for col_temp in col_names:
        col = col_temp + '_' + key_col + '_' + 'mean'
        features_new[col_temp + '_dive1_' + col] = features_new[col_temp] / (features_new[col] + 1e-5)
        # 另一种的计算方法是
        # features_new[col_temp + '_dive1_' + col] = features_new[key_col] / (features_new[col] + 1e-5)
        col_names_new.append(col_temp + '_dive1_' + col)
        col = col_temp + '_' + key_col + '_' + 'median'
        features_new[col_temp + '_dive2_' + col] = features_new[col_temp] / (features_new[col] + 1e-5)
        # 另一种的计算方法是
        # features_new[col_temp + '_dive2_' + col] = features_new[key_col] / (features_new[col] + 1e-5)
        col_names_new.append(col_temp + '_dive2_' + col)

    # 黄金组合特征
    for col_temp in col_names:
        col = col_temp + '_' + key_col + '_' + 'mean'
        features_new[col_temp + '_minus1_' + col] = features_new[col_temp] - features_new[col]
        # 另一种的计算方法是
        # features_new[col_temp + '_minus1_' + col] = features_new[key_col] - features_new[col]
        col_names_new.append(col_temp + '_minus1_' + col)
        features_new[col_temp + '_minus2_' + col] = features_new[col_temp] - features_new[col]
        # 另一种的计算方法是
        # features_new[col_temp + '_minus2_' + col] = features_new[key_col] - features_new[col]
        col_names_new.append(col_temp + '_minus2_' + col)

    # 组内归一化特征
    for col_temp in col_names:
        col_mean = col_temp + '_' + key_col + '_' + 'mean'
        col_var = col_temp + '_' + key_col + '_' + 'var'
        features_new[col_temp + '_norm_' + key_col] = (features_new[col_temp] - features_new[col_mean]) / (
                    np.sqrt(features_new[col_var]) + 1e-5)
        # 另一种的计算方法是
        # features_new[col_temp + '_norm_' + key_col] = (features_new[key_col] - features_new[col_mean]) / (
        #           np.sqrt(features_new[col_var]) + 1e-5)
        col_names_new.append(col_temp + '_norm_' + key_col)

    # Gap特征
    for col_temp in col_names:
        col_q1 = col_temp + '_' + key_col + '_' + 'q1'
        col_q2 = col_temp + '_' + key_col + '_' + 'q2'
        features_new[col_temp + '_gap_' + key_col] = features_new[col_q2] - features_new[col_q1]
        col_names_new.append(col_temp + '_gap_' + key_col)

    # 数据倾斜特征
    for col_temp in col_names:
        col_mean = col_temp + '_' + key_col + '_' + 'mean'
        col_median = col_temp + '_' + key_col + '_' + 'median'
        features_new[col_temp + '_mag1_' + key_col] = features_new[col_median] - features_new[col_mean]
        col_names_new.append(col_temp + '_mag1_' + key_col)
        features_new[col_temp + '_mag2_' + key_col] = features_new[col_median] / (features_new[col_mean] + 1e-5)
        col_names_new.append(col_temp + '_mag2_' + key_col)

    # 变异系数
    for col_temp in col_names:
        col_mean = col_temp + '_' + key_col + '_' + 'mean'
        col_var = col_temp + '_' + key_col + '_' + 'var'
        features_new[col_temp + '_cv_' + key_col] = np.sqrt(features_new[col_var]) / (features_new[col_mean] + 1e-5)
        col_names_new.append(col_temp + '_cv_' + key_col)

    features_new.drop([key_col], axis=1, inplace=True)
    features_new.drop(col1, axis=1, inplace=True)
    col_names_new = list(features_new.columns)

    return features_new, col_names_new


# ## 多特征衍生

# In[ ]:


def multi_cross_combination(col_names: list, features: pd.DataFrame, one_hot: bool = True):
    """
    多特征组合交叉衍生

    :param col_names: 参与交叉衍生的列名称
    :param features: 原始数据集
    :param one_hot: 是否进行独热编码

    :return：交叉衍生后的新特征和新列名称
    """

    # 创建组合特征
    col_names_new = '_'.join([str(i) for i in col_names])
    features_new = features[col_names[0]].astype('str')

    for col in col_names[1:]:
        features_new = features_new + '_' + features[col].astype('str')

        # 将组合特征转化为DataFrame
    features_new = pd.DataFrame(features_new, columns=[col_names_new])

    # 对新的特征列进行独热编码
    if one_hot:
        enc = OneHotEncoder()
        enc.fit_transform(features_new)
        col_names_new = cate_col_name(enc, [col_names_new], skip_binary=True)
        features_new = pd.DataFrame(enc.fit_transform(features_new).toarray(), columns=col_names_new)

    return features_new, col_names_new


# In[ ]:


def multi_group_statistics(key_col: list,
                           features: pd.DataFrame,
                           col_num: list = None,
                           col_cat: list = None,
                           num_stat: list = ['mean', 'var', 'max', 'min', 'skew', 'median'],
                           cat_stat: list = ['mean', 'var', 'max', 'min', 'median', 'count', 'nunique'],
                           quantile: bool = True):
    """
    多特征分组统计特征衍生函数

    :param key_col: 分组参考的关键特征
    :param features: 原始数据集
    :param col_num: 参与衍生的连续型特征
    :param col_cat: 参与衍生的离散型特征
    :param num_stat: 连续特征分组统计指标
    :param cat_stat: 离散特征分组统计指标
    :param quantile: 是否计算分位数

    :return：交叉衍生后的新特征和新特征的名称
    """
    # 生成原数据合并的主键
    features_key1, col1 = multi_cross_combination(key_col, features, one_hot=False)

    # 当输入的特征有连续型特征时
    if col_num is not None:
        aggs_num = {}
        col_names = col_num

        # 创建agg方法所需字典
        for col in col_num:
            aggs_num[col] = num_stat

        # 创建衍生特征名称列表
        cols_num = key_col.copy()

        for key in aggs_num.keys():
            cols_num.extend([key + '_' + col1 + '_' + stat for stat in aggs_num[key]])

        # 创建衍生特征df
        features_num_new = features[col_num + key_col].groupby(key_col).agg(aggs_num).reset_index()
        features_num_new.columns = cols_num

        # 生成主键
        features_key2, col2 = multi_cross_combination(key_col, features_num_new, one_hot=False)

        # 创建包含合并主键的数据集
        features_num_new = pd.concat([features_key2, features_num_new], axis=1)

        # 当输入的特征有连续型也有离散型特征时
        if col_cat is not None:
            aggs_cat = {}
            col_names = col_num + col_cat

            # 创建agg方法所需字典
            for col in col_cat:
                aggs_cat[col] = cat_stat

            # 创建衍生特征名称列表
            cols_cat = key_col.copy()

            for key in aggs_cat.keys():
                cols_cat.extend([key + '_' + col1 + '_' + stat for stat in aggs_cat[key]])

            # 创建衍生特征df
            features_cat_new = features[col_cat + key_col].groupby(key_col).agg(aggs_cat).reset_index()
            features_cat_new.columns = cols_cat

            # 生成主键
            features_key3, col3 = multi_cross_combination(key_col, features_cat_new, one_hot=False)

            # 创建包含合并主键的数据集
            features_cat_new = pd.concat([features_key3, features_cat_new], axis=1)

            # 合并连续特征衍生结果与离散特征衍生结果
            # 合并新的特征矩阵
            df_temp = pd.concat([features_num_new, features_cat_new], axis=1)
            df_temp = df_temp.loc[:, ~df_temp.columns.duplicated()]
            # 将新的特征矩阵与原始数据集合并
            features_new = pd.merge(features_key1, df_temp, how='left', on=col1)


        # 当只有连续特征时
        else:
            # merge连续特征衍生结果与原始数据，然后删除重复列
            features_new = pd.merge(features_key1, features_num_new, how='left', on=col1)
            features_new = features_new.loc[:, ~features_new.columns.duplicated()]

    # 当没有输入连续特征时
    else:
        # 但存在分类特征时，即只有分类特征时
        if col_cat is not None:
            aggs_cat = {}
            col_names = col_cat

            for col in col_cat:
                aggs_cat[col] = cat_stat

            cols_cat = key_col.copy()

            for key in aggs_cat.keys():
                cols_cat.extend([key + '_' + col1 + '_' + stat for stat in aggs_cat[key]])

            features_cat_new = features[col_cat + key_col].groupby(key_col).agg(aggs_cat).reset_index()
            features_cat_new.columns = cols_cat

            features_new = pd.merge(features_key1, features_cat_new, how='left', on=col1)
            features_new = features_new.loc[:, ~features_new.columns.duplicated()]

    if quantile:
        # 定义四分位计算函数
        def q1(x):
            """
            下四分位数
            """
            return x.quantile(0.25)

        def q2(x):
            """
            上四分位数
            """
            return x.quantile(0.75)

        agg_name = {}
        for col in col_names:
            agg_name[col] = ['q1', 'q2']

        cols = key_col.copy()

        for key in agg_name.keys():
            cols.extend([key + '_' + col1 + '_' + stat for stat in agg_name[key]])

        aggs = {}
        for col in col_names:
            aggs[col] = [q1, q2]

        features_temp = features[col_names + key_col].groupby(key_col).agg(aggs).reset_index()
        features_temp.columns = cols
        features_new.drop(key_col, axis=1, inplace=True)

        # 生成主键
        features_key4, col4 = multi_cross_combination(key_col, features_temp, one_hot=False)

        # 创建包含合并主键的数据集
        features_temp = pd.concat([features_key4, features_temp], axis=1)

        # 合并新特征矩阵
        features_new = pd.merge(features_new, features_temp, how='left', on=col1)
        features_new = features_new.loc[:, ~features_new.columns.duplicated()]

    features_new.drop(key_col + [col1], axis=1, inplace=True)
    col_names_new = list(features_new.columns)

    return features_new, col_names_new


# In[ ]:


def multi_polynomial_features(col_names: list, degree: int, features: pd.DataFrame):
    """
    连续特征多特征多项式衍生函数

    :param col_names: 参与交叉衍生的列名称
    :param degree: 多项式最高阶
    :param features: 原始数据集

    :return：交叉衍生后的新特征和新列名称
    """

    # 创建空列表容器
    col_names_new_l = []

    # 计算带入多项式计算的特征数
    n = len(col_names)

    # 提取需要进行多项式衍生的特征
    features = features[col_names]

    # 进行多项式特征组合
    array_new_temp = PolynomialFeatures(degree=degree, include_bias=False).fit_transform(features)
    # 选取衍生的特征
    array_new_temp = array_new_temp[:, n:]

    # 创建列名称列表
    deg = 2
    while deg <= degree:
        m = 1
        a1 = range(deg, -1, -1)
        a2 = []
        while m < n:
            a1 = list(product(a1, range(deg, -1, -1)))
            if m > 1:
                for i in a1:
                    i_temp = list(i[0])
                    i_temp.append(i[1])
                    a2.append(i_temp)
                a1 = a2.copy()
                a2 = []
            m += 1

        a1 = np.array(a1)
        a3 = a1[a1.sum(1) == deg]

        for i in a3:
            col_names_new_l.append('_'.join(col_names) + '_' + ''.join([str(i) for i in i]))

        deg += 1

    # 拼接新特征矩阵
    features_new = pd.DataFrame(array_new_temp, columns=col_names_new_l)
    col_names_new = col_names_new_l

    return features_new, col_names_new


# ## 时序字段的特征衍生

# In[ ]:


from datetime import datetime
import pandas as pd

def time_series_creation(time_series: pd.Series, time_stamp: dict = None, precision_high: bool = False):
    """
    时序字段的特征衍生

    :param time_series：时序特征，需要是一个Series
    :param time_stamp：手动输入的关键时间节点的时间戳，需要组成字典形式，字典的key、value分别是时间戳的名字与字符串
    :param precision_high：是否精确到时、分、秒
    :return features_new, col_names_new：返回创建的新特征矩阵和特征名称
    """

    # 创建衍生特征df
    features_new = pd.DataFrame()

    # 提取时间字段及时间字段的名称
    time_series = pd.to_datetime(time_series)
    col_names = time_series.name

    # 年月日信息提取
    features_new[col_names + '_year'] = time_series.dt.year
    features_new[col_names + '_month'] = time_series.dt.month
    features_new[col_names + '_day'] = time_series.dt.day

    if precision_high:
        features_new[col_names + '_hour'] = time_series.dt.hour
        features_new[col_names + '_minute'] = time_series.dt.minute
        features_new[col_names + '_second'] = time_series.dt.second

    # 自然周期提取
    features_new[col_names + '_quarter'] = time_series.dt.quarter
    # features_new[col_names + '_weekofyear'] = time_series.dt.weekofyear
    # Series.dt.weekofyear and Series.dt.week have been deprecated.Please use Series.dt.isocalendar().week instead.
    features_new[col_names + '_weekofyear'] = time_series.dt.isocalendar().week
    features_new[col_names + '_dayofweek'] = time_series.dt.dayofweek + 1
    features_new[col_names + '_weekend'] = (features_new[col_names + '_dayofweek'] > 5).astype(int)

    if precision_high:
        features_new['hour_section'] = (features_new[col_names + '_hour'] // 6).astype(int)

    # 关键时间点时间差计算
    # 创建关键时间戳名称的列表和时间戳列表
    time_stamp_name_l = []
    time_stamp_l = []

    if time_stamp is not None:
        time_stamp_name_l = list(time_stamp.keys())
        time_stamp_l = [pd.to_datetime(x) for x in list(time_stamp.values())]

    # 准备通用关键时间点时间戳
    time_max = time_series.max()
    time_min = time_series.min()
    time_now = pd.to_datetime(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    time_stamp_name_l.extend(['time_max', 'time_min', 'time_now'])
    time_stamp_l.extend([time_max, time_min, time_now])

    # 时间差特征衍生
    for time_stamp, time_stampName in zip(time_stamp_l, time_stamp_name_l):
        time_diff = time_series - time_stamp
        features_new['time_diff_days' + '_' + time_stampName] = time_diff.dt.days
        # features_new['time_diff_months'+'_'+time_stampName] = np.round(features_new['time_diff_days'+'_'+time_stampName] / 30).astype('int')
        features_new['time_diff_months' + '_' + time_stampName] = (time_series.dt.year - time_stamp.year) * 12 + (time_series.dt.month - time_stamp.month)

        if precision_high:
            features_new['time_diff_seconds' + '_' + time_stampName] = time_diff.dt.seconds
            features_new['time_diff_h' + '_' + time_stampName] = time_diff.values.astype('timedelta64[h]').astype('int')
            features_new['time_diff_s' + '_' + time_stampName] = time_diff.values.astype('timedelta64[s]').astype('int')

    col_names_new = list(features_new.columns)
    return features_new, col_names_new


# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer

def nlp_group_statistics(features: pd.DataFrame,
                         col_cat: list,
                         key_col: str(list) = None,
                         tfidf: bool = True,
                         count_vec: bool = True):
    """
    多变量分组统计特征衍生函数

    :param features: 原始数据集
    :param col_cat: 参与衍生的离散型变量，只能带入多个列
    :param key_col: 分组参考的关键变量，输入字符串时代表按照单独列分组，输入list代表按照多个列进行分组
    :param tfidf: 是否进行tfidf计算
    :param count_vec: 是否进行count_vectorizer计算

    :return：NLP特征衍生后的新特征和新特征的名称
    """

    # 提取所有需要带入计算的特征名称和特征
    if key_col is not None:
        if type(key_col) == str:
            key_col = [key_col]
        col_name_temp = key_col.copy()
        col_name_temp.extend(col_cat)
        features = features[col_name_temp]
    else:
        features = features[col_cat]

    # 定义count_vectorizer计算和TF-IDF计算过程
    def col_stat(features_col_stat: pd.DataFrame = features,
                 col_cat_col_stat: list = col_cat,
                 key_col_col_stat: str(list) = key_col,
                 count_vec_col_stat: bool = count_vec,
                 tfidf_col_stat: bool = tfidf):
        """
        count_vectorizer计算和TF-IDF计算函数
        返回结果需要注意，此处返回带有key_col的衍生特征矩阵及特征名称
        """
        n = len(key_col_col_stat)
        col_cat_col_stat = [x + '_' + '_'.join(key_col_col_stat) for x in col_cat_col_stat]
        if tfidf_col_stat:
            # 计算count_vectorizer
            features_col_stat_new_cntv = features_col_stat.groupby(key_col_col_stat).sum().reset_index()
            col_names_new_cntv = key_col_col_stat.copy()
            col_names_new_cntv.extend([x + '_cntv' for x in col_cat_col_stat])
            features_col_stat_new_cntv.columns = col_names_new_cntv

            # 计算TF-IDF
            transformer_col_stat = TfidfTransformer()
            tfidf_df = transformer_col_stat.fit_transform(features_col_stat_new_cntv.iloc[:, n:]).toarray()
            col_names_new_tfv = [x + '_tfidf' for x in col_cat_col_stat]
            features_col_stat_new_tfv = pd.DataFrame(tfidf_df, columns=col_names_new_tfv)

            if count_vec_col_stat:
                features_col_stat_new = pd.concat([features_col_stat_new_cntv, features_col_stat_new_tfv], axis=1)
                col_names_new_cntv.extend(col_names_new_tfv)
                col_names_col_stat_new = col_names_new_cntv
            else:
                col_names_col_stat_new = pd.concat([features_col_stat_new_cntv[:, :n], features_col_stat_new_tfv],
                                                   axis=1)
                features_col_stat_new = key_col_col_stat + features_col_stat_new_tfv

        # 如果只计算count_vectorizer时
        elif count_vec_col_stat:
            features_col_stat_new_cntv = features_col_stat.groupby(key_col_col_stat).sum().reset_index()
            col_names_new_cntv = key_col_col_stat.copy()
            col_names_new_cntv.extend([x + '_cntv' for x in col_cat_col_stat])
            features_col_stat_new_cntv.columns = col_names_new_cntv

            col_names_col_stat_new = col_names_new_cntv
            features_col_stat_new = features_col_stat_new_cntv

        return features_col_stat_new, col_names_col_stat_new

    # key_col==None时对原始数据进行NLP特征衍生
    # 此时无需进行count_vectorizer计算
    if key_col is None:
        if tfidf:
            transformer = TfidfTransformer()
            tfidf = transformer.fit_transform(features).toarray()
            col_names_new = [x + '_tfidf' for x in col_cat]
            features_new = pd.DataFrame(tfidf, columns=col_names_new)

    # key_col!=None时对分组汇总后的数据进行NLP特征衍生
    else:
        n = len(key_col)
        # 如果是依据单个特征取值进行分组
        if n == 1:
            features_new, col_names_new = col_stat()
            # 将分组统计结果拼接回原矩阵
            features_new = pd.merge(features[key_col[0]], features_new, how='left', on=key_col[0])
            features_new = features_new.iloc[:, n:]
            col_names_new = features_new.columns

        # 如果是多特征交叉分组
        else:
            features_new, col_names_new = col_stat()
            # 在原数据集中生成合并主键
            features_key1, col1 = multi_cross_combination(key_col, features, one_hot=False)
            # 在衍生特征数据集中创建合并主键
            features_key2, col2 = multi_cross_combination(key_col, features_new, one_hot=False)
            features_key2 = pd.concat([features_key2, features_new], axis=1)
            # 将分组统计结果拼接回原矩阵
            features_new = pd.merge(features_key1, features_key2, how='left', on=col1)
            features_new = features_new.iloc[:, n + 1:]
            col_names_new = features_new.columns

    return features_new, col_names_new

