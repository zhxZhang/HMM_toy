import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import os 

# sns.set_style("whitegrid")


class EDA(object):

    def read_file(self, csv_path):

        print ('file path:', csv_path)
        self.df = pd.read_csv(csv_path)

        self.num_row, self.num_col = self.df.shape
        print ('shape of df: row %d, col %d .' % (self.num_row, self.num_col))


    def na_detect(self, output_folder):

        # check folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)


        # count NA and save NA cols to csv files
        na_dict = {}

        na = self.df.isnull().sum()/self.num_row
        na = na.sort_values()

        indice = na[na>0].index
        num_na_col = len(indice)
        na = na.reindex(indice).dropna()

        for ind, val in na.iteritems():
            na_dict[ind] = val

        print ('num of na cols: ', num_na_col)

        gap = [0, 0.01, 0.2, 0.5, 1]
        na = {'na{0}_{1}'.format(gap[i], gap[i+1]): [] for i in range(len(gap)-1)}

        for k, v in na_dict.items():
            for i in range(len(gap)):
                if v <= gap[i]:
                    tmp = 'na{0}_{1}'.format(gap[i-1], gap[i])
                    na[tmp].append(k)
                    break

        # num of NA in different level
        for k, col in na.items():
            print ('num of %s : %d' % (k, len(col)))
            self.df[col].to_csv(output_folder + '/' + k + '.csv')

    
        # sns.barplot(data=na.to_frame(name='rate').T)
        # plt.show()


if __name__ == "__main__":

    folder = 'd:/home_credit_kaggle'
    ap_train_path = folder + '/raw_data/application_train.csv'

    obj = EDA()
    obj.read_file(ap_train_path)
    # obj.na_detect(folder + "/na_data")

