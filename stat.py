# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import math
from copy import deepcopy
import types
# import missingno as mn
import lightgbm as lgb
# sns.set_style("whitegrid")


def chi_square_test(x, y):
    return 0

def _is_str(input_):
    return type(input_) == type('str')

def _is_int(input_):
    return type(input_) == type('1')

def _is_nan(input_):
    if _is_str(input_): return False
    else: return math.isnan(input_)

def _is_category(seri):
    if _is_nan(seri[0]):  seri = seri.dropna()
    value = seri.values[0]
    # if exist NaN, Int also be converted to Float64
    if _is_str(value) or _is_int(value) or value % 1 == 0: return True
    else: return False

def _value_counts(seri, norm=True):
    val_count = dict(seri.value_counts(normalize=norm))
    num_cat = len(val_count.keys())
    return val_count, num_cat

def _is_unbalance(seri_or_dict):
    values = dict(seri_or_dict).values()
    return True if max(values) / min(values) < 15 else False

# def _fill_num(seri, mode="", change_to=None):
#     name = "_".join(['NA', seri.name, mode])

def _fillna_special_num(seri, name="", fill_modes=[""], change_to=[np.NaN]):
    df = seri.to_frame(name)
    fill_dict = {}
    for m in fill_modes:
        col_name = 'NA_' + name + '_' + m
        if m == "int":
            fill_dict.update({col_name + '_' + str(num): num for num in change_to })
        if m == "exp":
            if _is_category(seri):
                fill_dict.update({col_name + '_' + str(num): num for num in list(seri.mode())})
            else:
                fill_dict.update({col_name: seri.mean()})
    for new in fill_dict.keys():
        df[new] = df[name]
    del df[name]
    return df.fillna(value=fill_dict)

def _fillna_model(seri):
    return pd.DataFrame({})

def _one_hot(seri, dummy_na=False):
    return pd.get_dummies(seri, prefix='one-hot', dummy_na=dummy_na)

def _int_encoding(seri):
    seri, unique = pd.factorize(seri, na_sentinel=-999)
    seri = pd.Series(seri).map(lambda x: np.NaN if x == -999 else x)
    map_to_int = {unique[pos]: pos for pos in range(len(unique))}
    return seri, map_to_int

class EDA(object):

    VAR_INFO = ['value_count', 'f_type', 'num_cat', 'map2int']

    def __init__(self):
        self.raw_data = None
        self.num_row = None
        self.num_col = None

        self.train = None
        self.valid = None
        self.test = None

        self.cat_var_info = {}
        self.num_var_info = {}
        self.var_high_dim = {}

        self.var_unbalance = []

        self.is_na = None
        self.na_stat = None
        self.var_na = None

    def read_file(self, csv_path):
        self.raw_data = pd.read_csv(csv_path)
        self.num_row, self.num_col = self.raw_data.shape

    def na_detect(self):
        # NA percent & NA cols
        na_stat = self.raw_data.isnull().sum()/self.num_row
        var_na = na_stat[na_stat > 0].index
        na_stat = na_stat.reindex(var_na).dropna().sort_values()
        print('num of na columns: ', len(var_na))
        # save info
        self.is_na = len(var_na) > 0
        self.na_stat = na_stat
        self.var_na = var_na

    def _init_var_info(self):
        return 0

    def assert_f_type(self):
        # Divide into Category / Numerical
        for f in self.raw_data.columns:
            if _is_category(self.raw_data[f]):
                self.cat_var_info[f] = dict.fromkeys(EDA.VAR_INFO, None)
                self.cat_var_info[f]['f_type'] = 'ORIGIN'
            else:
                self.num_var_info[f] = dict.fromkeys(EDA.VAR_INFO, None)
                self.num_var_info[f]['f_type'] = 'ORIGIN'

    def assert_unbalance(self):
        for f in self.cat_var_info.keys():
            val_freq, num_cat = _value_counts(self.raw_data[f])
            self.cat_var_info[f]['value_count'] = val_freq
            self.cat_var_info[f]['num_cat'] = num_cat
            if _is_unbalance(val_freq):
                self.var_unbalance.append(f)

    def _cat_int_encoding(self, train_df):
        if not self.cat_var_info:
            raise ValueError
        for col in self.cat_var_info.keys():
            seri, map2int = _int_encoding(train_df[col])
            train_df[col] = seri
            self.cat_var_info[col]['map2int'] = map2int     
        return train_df

    def _na_fill(self, train_df):
        for col in self.var_na:
            add_vars = []
            # NA - expectation, 999.
            add_df_1 = _fillna_special_num(self.raw_data[col], name=col, fill_modes=['int','exp'], change_to=[999])
            train_df = pd.concat([train_df, add_df_1], axis=1)
            add_vars += list(add_df_1.columns)
            # NA - NN & RF pred
            # add_df_2 = _fillna_model(self.raw_data)
            # train_df = pd.concat([train_df, add_df_2], axis=1)
            # add_vars.append(add_df_2.columns)

            # update cat/num var_info
            is_cat = col in self.cat_var_info.keys()
            if is_cat:
                self.cat_var_info.update(dict.fromkeys(add_vars, deepcopy(self.cat_var_info[col])))
                for var in add_vars:
                    val_freq, _ = _value_counts(train_df[var])
                    self.cat_var_info[var]['value_count'] = val_freq
                    self.cat_var_info[var]['f_type'] = 'NA_DERIVE'
        return train_df

    def _assert_high_dem_cat(self):
        self.var_high_dim = {col: self.cat_var_info[col]['num_cat'] 
                            for col in self.cat_var_info.keys() 
                            if self.cat_var_info[col]['num_cat'] > 1000}

    def _is_high_dim(self, feature_name):
        if not self.var_high_dim: self._assert_high_dem_cat()
        return feature_name in self.var_high_dim.keys()

    def _one_hot_encoding(self, data_df, target_col, del_origin=True):
        for col in target_col:
            is_na_derive = self.cat_var_info[col]['f_type'] == 'NA_DERIVE'
            is_high_dim = self._is_high_dim(col)
            if is_na_derive or is_high_dim:  
                continue
            add_df = _one_hot(data_df[col], dummy_na=True)
            data_df = pd.concat([data_df, add_df], axis=1)
            if del_origin:
                del data_df[col]
        return data_df

    def preprocess(self,ap_train_path, one_hot=False, save_to=""):
        self.read_file(ap_train_path)
        print('file path:', ap_train_path)
        print('shape of df: row %d, col %d .' % (self.num_row, self.num_col))
        print('NA Detect...')
        self.na_detect()
        print('Count feature type...')
        self.assert_f_type()
        print('Assert unbalance feature...')
        self.assert_unbalance()
        print('****** Pre process data *******')
        train_data = self.raw_data
        print('category var int encoding...')
        train_data = self._cat_int_encoding(train_data)  #Rep DF
        print('Fill NA...')
        train_data = self._na_fill(train_data)  # Add DF
        if one_hot:
            print('one hot encoding...')
            train_data = self._one_hot_encoding(train_data, self.cat_var_info.keys())
        print('save preprocess to ' + save_to + '...')
        train_data.to_csv(save_to)


""" 
def lgb_baseline(raw_data, m_saved_folder):

    data,  map_ = _cat_int_encoding(raw_data)
    label = data['TARGET']
    del data['TARGET']
    train_data = lgb.Dataset(
        data, label=label, feature_name=data.columns, categorical_feature=map_.keys())
    val_data = lgb.Dataset('test.svm', reference=train_data)

    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.2,
        # 'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        # Minimum number of data need in a child(min_data_in_leaf)
        'min_child_samples': 20,
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        # Subsample ratio of columns when constructing each tree.
        'colsample_bytree': 0.3,
        # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'min_child_weight': 5,
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 4,
        'verbose': 0
    }
    num_round = 10
    early_stopping_rounds = 10
    num_boost_round = 3000
    bst1 = lgb.train(lgb_params,
                     train_data,
                     valid_sets=[val_data]
                     evals_result=evals_results,
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10,
                     feval=None)

    bst1.save_model(m_saved_folder+'model.txt',
                    num_iteration=bst1.best_iteration)
    return 0
     """


if __name__ == "__main__":

    folder = 'd:/home_credit_kaggle'
    ap_train_path = folder + '/raw_data/application_train.csv'
    test_path = folder + '/raw_data/application_test.csv'

    eda = EDA()
    eda.preprocess(ap_train_path=ap_train_path, one_hot=True, save_to='d:/pre_process_nontree.csv')

    # lgb_baseline(eda.raw_data, floder)
