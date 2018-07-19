import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")

folder = 'd:\\home_credit_kaggle\\raw_data'

ap_train_path = folder+'\\application_train.csv'

def read_file(csv_path):
    print ('read_file path:', csv_path)
    df = pd.read_csv(csv_path)

    # print (df.describe())

    # print ('heads :', df.head())

    num_row, num_col = df.shape
    # print ('shape of df: row %d, col %d .' % (num_row, num_col) )

    # count and print NAs
    na = df.isnull().sum()/num_row
    na = na.sort_values()

    indice = na[na>0].index
    num_na_col = len(indice)
    na = na.reindex(indice).dropna()

    # print ('num of NAs cols: ', num_na_col)
    # print ('rate of num_na_col to all_cols : %.2f.' % (num_na_col/float(num_col)))

    
    sns.barplot(data=na.to_frame(name='rate').T)
    plt.show()




read_file(ap_train_path)