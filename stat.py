# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import types
# import missingno as mn
import lightgbm as lgb
# sns.set_style("whitegrid")


def chi_square_test(x, y):
    return 0

def _is_category(seri):
    if seri[0] == np.NaN: 
        seri = seri.dropna()
    # if exist NaN, Int will be converted to Float64
    if type(seri[0]) is type('str'): return True
    if seri[0] % 1 == 0: return True
    else: return False

def _value_counts(seri, norm=True):
    # Inputs: Series
    val_count = dict(seri.value_counts(normalize=norm))
    num_cat = len(val_count.keys())
    return val_count, num_cat

def _is_unbalance(data):
    # Inputs: Series / Dict
    values = dict(data).values()
    return True if max(values) / min(values) < 15 else False

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

        self.var_unbalance = []
        self.var_na_add = []
        self.cat_var_info = {}
        self.num_var_info = {}

        self.is_na = None
        self.na_stat = None
        self.var_na = None

    def read_file(self, csv_path):
        self.raw_data = pd.read_csv(csv_path)
        self.num_row, self.num_col = self.raw_data.shape

        print('file path:', csv_path)        
        print('shape of df: row %d, col %d .' % (self.num_row, self.num_col))

    def na_detect(self):
        print('NA Detect...')
        # NA percent & NA cols
        na_stat = self.raw_data.isnull().sum()/self.num_row
        var_na = na_stat[na_stat > 0].index
        na_stat = na_stat.reindex(var_na).dropna().sort_values()
        print('num of na columns: ', len(var_na))
        # save info
        self.is_na = len(var_na) > 0
        self.na_stat = na_stat
        self.var_na = var_na

    def assert_f_type(self):
        print('Count feature type...')
        for f in self.raw_data.columns:
            if _is_category(self.raw_data[f]):
                self.cat_var_info[f] = dict.fromkeys(EDA.VAR_INFO, None)
            else:
                self.num_var_info[f] = dict.fromkeys(EDA.VAR_INFO, None)

    def assert_unbalance(self):
        print('Assert unbalance feature...')
        for f in self.cat_var_info.keys():
            val_freq, num_cat = _value_counts(self.raw_data[f])
            self.cat_var_info[f]['value_count'] = val_freq
            self.cat_var_info[f]['num_cat'] = num_cat
            if _is_unbalance(val_freq):
                self.var_unbalance.append(f)

    def _cat_int_encoding(self, train_df):
        print('category var int encoding...')
        # Int encoding for cat
        # Replaced TrainSet
        if not self.cat_var_info:
            raise ValueError
        for col in self.cat_var_info.keys():
            seri, map2int = _int_encoding(train_df[col])
            train_df[col] = seri
            self.cat_var_info[col]['map2int'] = map2int     
        return train_df

    def _na_fill(self, train_df):
        print('Fill NA...')
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
                print(col)
                self.cat_var_info.update(dict.fromkeys(add_vars, self.cat_var_info[col]))
                for var in add_vars:
                    val_freq, _ = _value_counts(train_df[var])
                    self.cat_var_info[var]['value_count'] = val_freq
                    self.cat_var_info[var]['f_type'] = 'NA_DERIVE'
        return train_df

    def preprocess(self, one_hot=False, save_to=""):
        print('****** Pre process data *******')
        train_data = self.raw_data
        # Int encoding - Rep DF
        train_data = self._cat_int_encoding(train_data)
        # Fill NA - Add DF
        train_data = self._na_fill(train_data)

        if one_hot:
            print('one hot encoding...')
            for col in self.cat_var_info.keys():
                if self.cat_var_info[col]['f_type'] == 'NA_DERIVE': 
                    continue
                if self.cat_var_info[col]['num_cat'] > 1000:
                    print('continue ',self.cat_var_info[col]['num_cat'])
                    continue
                print(self.cat_var_info[col]['num_cat'])
                add_df = _one_hot(train_data[col], dummy_na=True)
                train_data = pd.concat([train_data, add_df], axis=1)
                del train_data[col]
        print('shape of train: ', train_data.shape)
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
    eda.read_file(ap_train_path)
    eda.na_detect()
    eda.assert_f_type()
    eda.assert_unbalance()
    eda.preprocess(one_hot=True, save_to='d:/pre_process_nontree.csv')

    # lgb_baseline(eda.raw_data, floder)
