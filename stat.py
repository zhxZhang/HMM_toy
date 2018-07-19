import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

# sns.set_style("whitegrid")


class EDA(object):

    def read_file(self, csv_path):

        # print ('file path:', csv_path)
        self.df = pd.read_csv(csv_path)

        num_row, num_col = df.shape
        print 'shape of df: row %d, col %d .' % (num_row, num_col)


    def na_detect(self):
        # count NA and save NA cols to csv files
        na_dict = {}

        na = df.isnull().sum()/num_row
        na = na.sort_values()

        indice = na[na>0].index
        num_na_col = len(indice)
        na = na.reindex(indice).dropna()

        for ind, val in na.iteritems():
            na_dict[ind] = val

        print 'num of na cols: ', num_na_col

        gap = [0, 0.01, 0.2, 0.5, 1]
        na = {'na{0}_{1}'.format(gap[i], gap[i+1]): [] for i in range(len(gap)-1)}

        for k, v in na_dict.iteritems():
            for i in range(len(gap)):
                if v <= gap[i]:
                    tmp = 'na{0}_{1}'.format(gap[i-1], gap[i])
                    na[tmp].append(k)

        # num of NA in different level
        for k, col in na.iteritems():
            print 'num of %s : %d' % (k, len(col))
            self.df[col].to_csv(k+'.csv')


        # sns.barplot(data=na.to_frame(name='rate').T)
        # plt.show()


if __name__ == "__main__":

    folder = 'd:\\credit'
    ap_train_path = folder + '\\application_train.csv'

    obj = EDA()
    obj.read_file(ap_train_path)
    obj.na_detect()

