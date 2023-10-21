import numpy as np
# import re
# import math
# import pandas as pd
# from statsmodels.stats.multitest import multipletests
# from sklearn.metrics import cohen_kappa_score

# This is handles the WHO Life Expectancy data specifically
# It could definitely be generalized see if a dtype is any type of int, etc
def fillna_by_number_dtype(df):
    for col in df:
        data_type = df[col].dtype
        if data_type == np.int64:
            df[col] = df[col].fillna(df[col].median().round(0).astype(np.int64))
        elif data_type == np.float64:
            df[col] = df[col].fillna(df[col].median())