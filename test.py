import yn_feature
import pandas as pd
import numpy as np 
import datetime,time

def test_01():
    df = pd.DataFrame({
        'cst_id':  [1,2,3,4,5,6]
        ,'dt':  [datetime.datetime(2023,1,1),datetime.datetime(2023,1,1),datetime.datetime(2023,1,1)
               ,datetime.datetime(2023,1,1),datetime.datetime(2023,1,1),datetime.datetime(2023,1,1)
               ]
        ,'x_cat_01': ['a','b','c','d','e','f']
        ,'x_cat_02': ['h','i','j','k','l','m']

        ,'x_num_01': [1,2,3,4,5,6]
        ,'x_num_02': [1,2,3,4,5,6]


    })
    # yn_feature.ft_single_numerical_feature(df,'cst_id','dt',['x_cat_01','x_cat_02'],['x_num_01','x_num_02'])
    df_result = yn_feature.func_cols_agg_by_id(pd.concat([df,df]),'cst_id',['x_num_01','x_num_02'])
    print(df_result)


def main():
    test_01()


if __name__ == "__main__":
    main()

