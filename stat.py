import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import os
import types
# import missingno as mn

# sns.set_style("whitegrid")


def chi_square_test(x, y):
    return 0


def cate_int_encoding(data, nan_change_to=np.NaN):

    # Return integer encoding & unique value list
    # Input can be list OR df

    map = {}

    is_list = type(data) is type([]):
    if is_list:
        data = DataFrame({"a": data})

    for col in data.columns:
        is_category = type(data.ix[0, col]) is type('a')
        if is_category:
            data[col], unique = pd.factorize(data[col], na_sentinel=-999)
            data[col] = data[col].map(lambda x: nan_change_to if x == -999 else x)
            map[col] = unique

    if is_list:
        return list(data), map
    else:
        return data, map


class EDA(object):

    def __init__(self):
        self.raw_data = None
        self.vars_info = {}
        self.num_row = None
        self.num_col = None

        self.na_stat = None
        self.na_cols = None

    def read_file(self, csv_path):
        print ('file path:', csv_path)
        self.raw_data = pd.read_csv(csv_path)
        self.vars_info = dict.fromkeys(self.raw_data.columns, {})

        self.num_row, self.num_col = self.raw_data.shape
        print ('shape of df: row %d, col %d .' % (self.num_row, self.num_col))

    def na_detect(self):

        # NA percent & NA cols
        na_stat = self.raw_data.isnull().sum()/self.num_row
        na_cols = na_stat[na_stat > 0].index
        na_stat = na_stat.reindex(na_cols).dropna().sort_values()
        print('num of na columns: ', len(na_cols))

        # save info
        self.na_stat = na_stat
        self.na_cols = na_cols


        # gap = [0, 0.01, 0.2, 0.5, 1]
        # # keys in na: [ cols name ]
        # na = {'na{0}_{1}'.format(gap[i], gap[i+1]): [] 
        #       for i in range(len(gap)-1)}


    def pre_process_data(self):
        cate_int_encoding(self.raw_data)









if __name__ == "__main__":

    folder = 'd:/home_credit_kaggle'
    ap_train_path = folder + '/raw_data/application_train.csv'


    obj = EDA()
    obj.read_file(ap_train_path)
    obj.na_detect()
    obj.pre_process_data()
