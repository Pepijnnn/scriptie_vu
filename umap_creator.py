import pandas as pd
import umap
from sklearn.externals import joblib
import time
import re

class Umaps():

    def create_tab_ten(self, short_table, nn=5, min_dist=0.1, metric='yule', n_comp = 2  ):
        print(short_table.info())
        # for col in short_table.columns:
        #     short_table.loc[:, col] = short_table.loc[:, col] / short_table.loc[:, col].max()
        #     print(col)
        # subdf = short_table[['Geslacht','IsOverleden','Postcode']].fillna("0",inplace=True)
        # dummies = pd.get_dummies(subdf) 
        short_table.drop(['Pseudo_id','Geslacht','IsOverleden','Postcode'], axis=1, inplace=True)
        short_table.fillna("0", inplace=True)
        for col in short_table.columns:
            if short_table[col].isnull().sum() > int(0.995*len(short_table)):
                short_table.drop([col], axis=1, inplace=True)
            else:
                # make it from string to numerical
                short_table.loc[:, col] = pd.to_numeric(short_table[col].apply(lambda x: re.sub(',', '.', str(x))))
                # normalize the columns
                short_table.loc[:, col] = short_table.loc[:, col] / short_table.loc[:, col].max() 

        one_hot_table = short_table

        title=f"tab10_metric={metric}, nn={nn}, min_dis={min_dist}, amount= {len(short_table)}"

        print("start umap")
        start = time.time()
        fit = umap.UMAP(n_neighbors=nn, min_dist=min_dist, n_components=n_comp, metric=metric)
        end = time.time()
        print(f"umap completed in {end - start} seconds")        
    
        u = fit.fit_transform(one_hot_table)
        print(f'transform completed in {time.time()-end} seconds')
        joblib_file = str(f"tab10_columns_all = {len(short_table.columns)} tab10_length= {len(one_hot_table)} metric = {kwargs['metric']}.pkl")
        joblib.dump(u, joblib_file)

        with open('tab_file_column_info.txt', 'a') as filehandle:
            print("################## SPLIT START ####################", file=filehandle)
            print(title, file=filehandle)
            print(joblib_file, file=filehandle)
            print(short_table.columns, file=filehandle)
            print("################## SPLIT END ####################", file=filehandle)