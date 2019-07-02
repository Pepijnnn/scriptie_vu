import pandas as pd
import umap
from sklearn import preprocessing

import seaborn as sns
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import matplotlib.style as style
sns.set_context('paper')
style.use('seaborn-darkgrid')

from sklearn.externals import joblib
import time
import re


class Umaps():

    def create_perc_df(self, df):
        # sort database to the pseudoid column
        pd.to_numeric(df["Column 47"], errors='coerce')
        df.rename(columns={"Column 47":"Pseudo_id"}, inplace=True)
        df.sort_values(['Pseudo_id'], inplace=True)

        # sub-dataframe of percentage R/S and the total of R+S instead of just total R and total S
        RS_df = df[[col for col in df.columns if str(col)[-2] == "_"]]
        percentage_total_RS = pd.DataFrame()
        for c in range(0, len(RS_df.columns), 2):
            RS_Percentage = pd.Series(RS_df[RS_df.columns[c]]/(RS_df[RS_df.columns[c+1]]+RS_df[RS_df.columns[c]])).rename(f"{RS_df.columns[c]}/S_Percentage")
            RS_Total = pd.Series(RS_df[RS_df.columns[c+1]]+RS_df[RS_df.columns[c]]).rename(f"{RS_df.columns[c]}+S_Total")
            percentage_total_RS = pd.concat([percentage_total_RS, RS_Percentage, RS_Total], axis=1)
        percentage_total_RS.fillna(0, inplace=True)

        # make percentages of antibiotics and add a total column
        ab_df = df[["Cotrimoxazol", "Meropenem", "Imipenem/cilastatine", "Vancomycine", "Fluoroquinolonen"]]
        ab_tot = pd.Series(ab_df.iloc[:].sum(axis=1)).rename(f"ABs_Total")
        percentage_abs = ab_df.div(ab_df.sum(axis=1), axis=0)
        percentage_total_abs = pd.concat([percentage_abs, ab_tot], axis=1)
        percentage_total_abs.fillna(0, inplace=True)

        # make percentages of departments and add a total column
        dep_df = df[[col for col in df.columns if "VUMC" in str(col)]]
        dep_tot = pd.Series(dep_df.iloc[:].sum(axis=1)).rename(f"Deps_Total")
        percentage_deps = dep_df.div(dep_df.sum(axis=1), axis=0)
        percentage_total_deps = pd.concat([percentage_deps, dep_tot], axis=1)
        percentage_total_deps.fillna(0, inplace=True)

        new_df = pd.concat([percentage_total_RS, percentage_total_abs, percentage_total_deps], axis=1)
        return new_df

    # cosine
    def create_umap_kweken_ab_opname(self, df, nn=20, min_dist=0.2, metric='cosine', n_comp = 2):
        
        new_df = self.create_perc_df(df)

        u = umap.UMAP(n_neighbors=nn, min_dist=min_dist, n_components=n_comp, metric=metric).fit_transform(new_df)
        
        title = str(f"thomasfile_flitered_length_pid = {len(new_df)} metric = {metric}, nn = {nn}, md = {min_dist}.pkl")
        joblib_file = title
        joblib.dump(u, joblib_file)

        ax = sns.scatterplot(data=df, x=u[:, 0], y=u[:, 1], s=10, palette='Spectral') # , hue="Avg(MIC)", legend="full" , linewidth=0
        ax.set_title(title)
        plt.savefig(str(title[:-3] + "png"))
        plt.show()        
        
        with open('tab_file_column_info.txt', 'a') as filehandle:
            print("################## SPLIT START ####################", file=filehandle)
            print("Afdelingen top 10", file=filehandle)
            print(title, file=filehandle)
            print(joblib_file, file=filehandle)
            print(new_df.columns, file=filehandle)
            print("################## SPLIT END ####################", file=filehandle)

    def create_umap_per_department(self, df, nn=30, min_dist=0.05, metric='yule', n_comp = 2):
        print(df.info())
        focus_table = "AfdelingNaamAanvrager"
        n = 11
        departments = df[focus_table].value_counts()[1:n].index.tolist()

        short_df = df[df[focus_table].isin(departments)]

        orig_df = short_df.copy()
        # department_df = short_df[focus_table]
        short_df.drop(focus_table, axis=1, inplace=True)
        short_df.fillna("0", inplace=True)

        oht = pd.get_dummies(short_df)
        # department_labels = preprocessing.LabelEncoder().fit_transform(department_df)

        u = umap.UMAP(n_neighbors=nn, min_dist=min_dist, n_components=n_comp, metric=metric).fit_transform(oht)

        title = str(f"Afdelingen top 10 = {len(oht.columns)} tab10_length= {len(oht)} metric = {metric}, nn = {nn}.pkl")
        joblib_file = title
        joblib.dump(u, joblib_file)
        

        sns.scatterplot(data=orig_df, x=u[:, 0], y=u[:, 1], s=10, hue=focus_table, legend="full", palette='Spectral', linewidth=0) # , hue="Avg(MIC)", legend="full"
        plt.show()        
        
        with open('tab_file_column_info.txt', 'a') as filehandle:
            print("################## SPLIT START ####################", file=filehandle)
            print("Afdelingen top 10", file=filehandle)
            print(title, file=filehandle)
            print(joblib_file, file=filehandle)
            print(oht.columns, file=filehandle)
            print("################## SPLIT END ####################", file=filehandle)

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
        joblib_file = str(f"tab10_columns_all = {len(short_table.columns)} tab10_length= {len(one_hot_table)} metric = {metric}.pkl")
        joblib.dump(u, joblib_file)

        with open('tab_file_column_info.txt', 'a') as filehandle:
            print("################## SPLIT START ####################", file=filehandle)
            print(title, file=filehandle)
            print(joblib_file, file=filehandle)
            print(short_table.columns, file=filehandle)
            print("################## SPLIT END ####################", file=filehandle)
    
    