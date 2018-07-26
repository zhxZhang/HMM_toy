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


def cate_int_encoding(data, nan_change_to=np.NaN):

    # Return integer encoding & unique value list
    # Input can be list OR df

    map_ = {}
    is_list = type(data) is type([])
    if is_list:
        data = pd.DataFrame({"a": data})

    for col in data.columns:
        is_category = type(data.ix[0, col]) is type('a')
        if is_category:
            data[col], unique = pd.factorize(data[col], na_sentinel=-999)
            data[col] = data[col].map(
                lambda x: nan_change_to if x == -999 else x)
            map_[col] = unique

    if is_list:
        return list(data), map_
    else:
        return data, map_


class EDA(object):

    def __init__(self):
        self.raw_data = None
        self.vars_info = {}
        self.num_row = None
        self.num_col = None

        self.na_stat = None
        self.na_cols = None

    def read_file(self, csv_path):
        print('file path:', csv_path)
        self.raw_data = pd.read_csv(csv_path)
        self.vars_info = dict.fromkeys(self.raw_data.columns, {})

        self.num_row, self.num_col = self.raw_data.shape
        print('shape of df: row %d, col %d .' % (self.num_row, self.num_col))

    def na_detect(self):

        # NA percent & NA cols
        na_stat = self.raw_data.isnull().sum()/self.num_row
        na_cols = na_stat[na_stat > 0].index
        na_stat = na_stat.reindex(na_cols).dropna().sort_values()
        print('num of na columns: ', len(na_cols))

        # save info
        self.na_stat = na_stat
        self.na_cols = na_cols


def lgb_baseline(raw_data, m_saved_folder):
    data, map_ = cate_int_encoding(raw_data)
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
    
    


if __name__ == "__main__":

    folder = 'd:/home_credit_kaggle'
    ap_train_path = folder + '/raw_data/application_train.csv'
    test_path = folder + '/raw_data/application_test.csv'

    eda = EDA()
    eda.read_file(ap_train_path)
    eda.na_detect()
    lgb_baseline(eda.raw_data, floder)
