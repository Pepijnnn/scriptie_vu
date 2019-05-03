import pandas as pd
import numpy as np
import pickle
import time
import json
import umap

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.style as style
sns.set_context('paper')
style.use('seaborn-darkgrid')

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.externals import joblib
from tqdm import tqdm

import re

class Plots():

    def cluster_info(self, umap_pickle, whole_df):
        """ Input is a umap and its dataframe.
            - First it searches an amount of clusters with Kmeans clustering.
            - Then it searces for each cluster what the most prominent attributes are.
            Output are the per cluster these most distinct attributes."""
        # load umap and make clusters
        model = joblib.load(umap_pickle)
        print(f'length model = {len(model[:, 0])}')
        df = whole_df.iloc[:len(model[:, 0])]
        
        
        # print(pd.to_numeric(df['Avg(MIC)'].str.replace(',', '.').astype(float)).describe())
        # add cluster locations per row in df
        df['coordinates'] = list(zip(model[:, 0], model[:, 1]))

        # train the K_means on the model coordinates
        estimator = KMeans(n_clusters=3, random_state=0).fit(model) # model[:, 0],model[:, 1]
        print("Fitted")
        cluster_dict_first = {i: np.where(estimator.labels_ == i)[0] for i in range(estimator.n_clusters)}
        print(f'Dit is de eerste cluster dict{cluster_dict_first}')
        
        df_indices = dict()#{i: list(indices) for indices in df for i in range(estimator.n_clusters)}
        for i in range(estimator.n_clusters):
            cluster_coords = cluster_dict_first.get(i)
            # make new dataframe from rows that have the coordinates from cluster_coords
            # print(f"Length van de series = {len(pd.Series(df['coordinates']))}, length van cluster coords = {len(cluster_coords)}")
            sub_df = df.loc[df['coordinates'].isin(cluster_coords)]
            df_indices[i] = sub_df.index.values

        # add the indices per cluster to the dict
        cluster_dict = {i: (df_indices.get(i), np.where(estimator.labels_ == i)[0])[1] for i in range(estimator.n_clusters)}
        print(f'Dit zijn die clusters weer gecombineerd{cluster_dict}')
        
        # first look if the df and the umap pickle have the same amount of objects
        print(f'The shape of the df = {df.shape}')
        print(f'The length of the x coordinates = {len(model[:, 0])} and the y coordinates: {len(model[:, 1])}')

        # split df into n_cluster amount of dataframes
        df_clusters = list()
        for key in cluster_dict.keys():
            locs = [i for i in cluster_dict.get(key)]
            new_df = df.loc[locs]
            df_clusters.append(new_df)
            # print(pd.to_numeric(new_df['Avg(MIC)'].str.replace(',', '.').astype(float)).describe())
            print(f"#################################### CLUSTER {key}  ##########################################################################################")
            for column in new_df.columns:
                print(column, len(new_df[column].dropna()))
                # print(new_df[column].dropna())
                try:
                    print(pd.to_numeric(new_df[column].dropna().str.replace(',', '.').astype(float)).describe())
                except AttributeError:
                    print(pd.to_numeric(new_df[column].dropna().replace(',', '.').astype(float)).describe())
                except ValueError:
                    print(new_df[column].dropna().describe())

        # per cluster make a list which item is most prevalent per column
        most_frequent_item = dict()
        for index, cluster in tqdm(enumerate(df_clusters)):
            # TODO find new information about clusters
            most_freq_per_column = list()

            print(f"dit is de cluster {cluster}")
            # add the item that is most frequent per column and put it in the dict
            for column in list(cluster):
                print(type(column))
                print(f"we kijken in deze column {column}")
                print(column =="coordinates")
                if column == "coordinates":
                    most_freq_per_column.append("NVT")
                    continue
                # if for some reason there is no idxmax we put a 0 there
                try:
                    most_freq_per_column.append(df_clusters[index][column].value_counts().mean())
                except ValueError:
                    most_freq_per_column.append(0)

            most_frequent_item[index] = most_freq_per_column

        print(f"The most frequent items per dict are: \n {most_frequent_item} \
            the coordinates per clusters are: \n {cluster_dict} ")
        # # write the information per cluster to a file aswell as which points belong to which clusters
        # with open('cluster_dumps.txt', 'a') as filehandle:
        #     print("################## SPLIT START ####################", file=filehandle)
        #     # print(umap_pickle)
        #     print(most_frequent_item, file=filehandle)
        #     print(cluster_dict, file=filehandle)
        #     print(df['coordinates'].values, file=filehandle)
        #     print("################## SPLIT END ####################", file=filehandle)

        # show a nice plot of the clusters so we can see which rows belong to which clusters
        colormap = np.zeros(len(model[:, 0]))
        for key in cluster_dict.keys():
            for value in cluster_dict[key]:
                colormap[value] = key
        
        # plt.scatter(model[:, 0], model[:, 1], s=1, c=colormap, cmap='Set1')
        # ax = sns.scatterplot(x=model[:, 0], y=model[:, 1], s=6, palette=("Spectral"))
        # ax.set_title('K-Mean Clusters')
        
        df['Avg(MIC)'].fillna(0, inplace=True)
        df.loc[:, 'Avg(MIC)'] = pd.to_numeric(df['Avg(MIC)'].apply(lambda x: re.sub(',', '.', str(x))))

        def custom_round(x, base=5):
            return int(base * round(float(x)/base))
        def round_down(x):
            if int(x) > 47:
                return 48
            else:
                return x
        
        df.loc[:, 'Avg(MIC)'] = df['Avg(MIC)'].apply(lambda x: custom_round(x, base=2))
        df.loc[:, 'Avg(MIC)'] = df['Avg(MIC)'].apply(lambda x: round_down(x))
        # print(df['Avg(MIC)'].value_counts())
        # exit()
        ax = sns.scatterplot(data=df,x=model[:, 0], y=model[:, 1], s=6, hue="Avg(MIC)", legend="full")
        ax.set_title('Umap patient clusters coloured to Avg MIC value')
        plt.show()

    def umap_over_kmeansclusters(self, umap_pickle, whole_df):
        """ Input is a umap and its dataframe.
            - First it searches an amount of clusters with Kmeans clustering.
            - Then it makes again a umap over these subclusters and visualises it, and does k-means again.
            Output are for a subcluster the most prominant features."""
        # load umap and make clusters
        model = joblib.load(umap_pickle)
        print(f'length model = {len(model[:, 0])}')
        df = whole_df.iloc[:len(model[:, 0])]

        # add cluster locations per row in df
        df['coordinates'] = list(zip(model[:, 0], model[:, 1]))

        # train the K_means on the model coordinates
        estimator = KMeans(n_clusters=3, random_state=0).fit(model) # model[:, 0],model[:, 1]

        plt.subplot(321)
        plt.scatter(model[:, 0], model[:, 1], s=1)
        print("Fitted")
        cluster_dict_first = {i: np.where(estimator.labels_ == i)[0] for i in range(estimator.n_clusters)}
        
        df_indices = dict()#{i: list(indices) for indices in df for i in range(estimator.n_clusters)}
        for i in range(estimator.n_clusters):
            cluster_coords = cluster_dict_first.get(i)
            # make new dataframe from rows that have the coordinates from cluster_coords
            sub_df = df.loc[df['coordinates'].isin(cluster_coords)]
            df_indices[i] = sub_df.index.values

        # add the indices per cluster to the dict
        cluster_dict = {i: (df_indices.get(i), np.where(estimator.labels_ == i)[0])[1] for i in range(estimator.n_clusters)}

        # split df into n_cluster amount of dataframes
        df_clusters = list()
        for key in cluster_dict.keys():
            locs = [i for i in cluster_dict.get(key)]
            new_df = df.loc[locs]
            df_clusters.append(new_df)

        for key, _ in enumerate(df_clusters):
            focus_cluster = df_clusters[key]
            focus_cluster.drop(['Geslacht','IsOverleden','Postcode'], axis=1, inplace=True)
            for col in focus_cluster.columns:
                if focus_cluster[col].isnull().sum() > int(0.995*len(focus_cluster)):
                    # print("333", focus_cluster[col].isnull().sum() > (0.995*len(focus_cluster)))
                    focus_cluster.drop([col], axis=1, inplace=True)
                else:
                    focus_cluster[col] = pd.to_numeric(focus_cluster[col], errors='coerce')
            focus_cluster.fillna(0, inplace=True)
            u = umap.UMAP(metric="correlation", n_neighbors=30, n_components=2, min_dist=0.0, random_state=42).fit_transform(focus_cluster)
            plt.subplot(322)
            plt.scatter(u[:, 0], u[:, 1], s=1)
            print(f"cluster {key} done")

        plt.title("Subumap from the first subcluster from the patient only dataframe, 50k", fontsize=18)
        plt.show()
        

    def supervised(self, df3):
        # print(len(set(df3["MIC_RuweWaarde"].values)))
        # print(df3["MIC_RuweWaarde"])
        # exit()
        # combine part of the data frames 
        # df1 = df1[['MonsterNummer','IsolaatNummer','MicroOrganismeOuder', 'MateriaalDescription']].copy()
        # df2 = df2[['MonsterNummer','IsolaatNummer','AntibioticaNaam', 'RISV_Waarde']].copy()
        # df3 = pd.merge(df1,df2[['MonsterNummer','IsolaatNummer','AntibioticaNaam','RISV_Waarde']], on=["MonsterNummer", "IsolaatNummer"])
        df3 = df3[['RISV_Waarde','MIC_RuweWaarde','Family','AntibioticaNaam']].copy()
        # make sure that the amout of R and S rows are the same
        df3.drop(df3.loc[(df3['RISV_Waarde']=='V') | (df3['RISV_Waarde']=='I')].index, inplace=True)
        df3.sort_values(by=['RISV_Waarde'], ascending=False, inplace=True)
        rest = df3.RISV_Waarde.value_counts()['S'] - df3.RISV_Waarde.value_counts()['R']
        df3 = df3.iloc[rest:,]
        
        # drop the columns that we don't want to train on
        y = df3['RISV_Waarde']
        # df3.drop(['Monsternummer', 'IsolaatNummer', 'RISV_Waarde'], inplace=True, axis=1)
        df3.drop(['RISV_Waarde'], inplace=True, axis=1)
        # fill other NaN's with most frequent string in column an drop Not Important columns
        for col in df3.columns:
            df3[col].fillna("0", inplace=True)


        # put colours relative to amount AB res column and create the clusters using umap and show them
        # the dummies column keeps it headings so new data can be inserted in the right place
        X = pd.get_dummies(df3)
        y_label = preprocessing.LabelEncoder().fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y_label, test_size = 0.25, random_state = 40)
        # first create the umap class and fit the train data to it for dimens reduction
        # trans = umap.UMAP(metric=kwargs["metric"], n_neighbors=kwargs["nn"], n_components=2, min_dist=kwargs["min_dis"], random_state=42).fit(X_train)
        # then create the svc class on which we train the new data with the corresponding labels
        # svc = SVC(gamma = 'auto').fit(trans.embedding_, y_train)
        # finally transform the test data on the umap dimensions so the SVM can give a score
        # test_embedding = trans.transform(X_test)
        # print(svc.score(test_embedding, y_test))

        dims = 2
     
        start = time.time()
        print("Start umap")        
        trans = umap.UMAP(metric=kwargs["metric"], n_neighbors=kwargs["nn"], n_components=dims, min_dist=kwargs["min_dis"], random_state=42).fit(X_train, y_train)
        end = time.time()
        print(f"umap completed in {end - start} seconds")

        knn = KNeighborsClassifier().fit(trans.embedding_, y_train)
        test_embedding = trans.transform(X_test)

        svc = SVC(gamma = 'auto', random_state=42).fit(trans.embedding_, y_train)

        start = time.time()
        print("Start rf")  
        rf = RandomForestClassifier(n_estimators=400, max_depth=50, random_state=42).fit(trans.embedding_, y_train)
        end = time.time()
        print(f"umap completed in {end - start} seconds")
        # score is accuracy van predict tov y_test
        print(f"KNN Score : {knn.score(test_embedding, y_test)}")
        print(f"SVC Score: {svc.score(test_embedding, y_test)}")
        print(f"RF Score: {rf.score(test_embedding, y_test)}")
        print(f"############## DIT IS DE LENGTH ############# {len(y)}")
        print(f'shuffled version: the umap dimensions were: {dims}D the metric used was: {kwargs["metric"]}, the n_neighbors: {kwargs["nn"]}, the min_distance: {kwargs["min_dis"]}, and the amount: {kwargs["amount"]}')

        # labels for the training scatterplot
        # train_le = preprocessing.LabelEncoder()
        # train_label = train_le.fit_transform(y_train)
        # train_labels2 = train_le.fit(y_train)
        # train_le_name_map = dict(zip(train_labels2.transform(train_le.classes_),train_labels2.classes_))

        # labels for the testing scatterplot
        # test_le = preprocessing.LabelEncoder()
        # test_label = test_le.fit_transform(y_test)
        # test_labels2 = test_le.fit(y_test)
        # test_le_name_map = dict(zip(test_labels2.transform(test_le.classes_),test_labels2.classes_))

        # def draw_umap(title="umap prediction"):
        #     trans = umap.UMAP(metric=kwargs["metric"], n_neighbors=kwargs["nn"], n_components=2, min_dist=kwargs["min_dis"], random_state=42).fit(X_train)
        #     svc = SVC(gamma = 'auto').fit(trans.embedding_, y_train)
        #     test_embedding = trans.transform(X_test)
        #     print(svc.score(test_embedding, y_test))
        #     print(f'the metric used was: {kwargs["metric"]}, the n_neighbors: {kwargs["nn"]}, the min_distance: {kwargs["min_dis"]}')
        #     exit()

            # for printing the plots to see how well the train or the test proces went
            # mapper = umap.UMAP(metric=kwargs["metric"], n_neighbors=kwargs["nn"], n_components=2, min_dist=kwargs["min_dis"]).fit(X_train, y=train_label)
            # test_embedding = mapper.transform(X_test)
            
            # # load from file
            # model = joblib.load(joblib_file)
            
            # set the colourmap and amount of colours finally create the scatterplot
            #################### Train supervised ################
            # cmap = cm.get_cmap('jet', len(list(train_le_name_map.keys()))) 
            # scat = plt.scatter(*mapper.embedding_.T, c=train_label, s=5, cmap=cmap, alpha=1.0)
 
            # cb = plt.colorbar(scat, spacing='uniform', ticks=list(train_le_name_map.keys()))
            # cb.ax.set_yticklabels(list(train_le_name_map.values()))
            
            # plt.title(title, fontsize=18)
            # plt.show()

            ################# Test Supervised ##############
            # cmap = cm.get_cmap('jet', len(list(test_le_name_map.keys()))) 
            # scat = plt.scatter(*test_embedding.T, c=test_label, s=5, cmap=cmap, alpha=1.0)
 
            # cb = plt.colorbar(scat, spacing='uniform', ticks=list(test_le_name_map.keys()))
            # cb.ax.set_yticklabels(list(test_le_name_map.values()))
            
            # plt.title(title, fontsize=18)
            # plt.show()
            # print(df3.shape)

        # draw_umap(title=("S5_metric={}_nn={}_min_dis={}_amount={}_AB_Resistentie"
        #     .format(kwargs["metric"], kwargs["nn"], kwargs["min_dis"], kwargs["amount"])))