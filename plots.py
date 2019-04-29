import pandas as pd
import numpy as np
import pickle
import time
import umap

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.externals import joblib



class Plots():

    def cluster_info(self, umap_pickle, whole_df):
        """ Input is a umap and its dataframe.
            - First it searches an amount of clusters with Kmeans clustering.
            - Then it searces for each cluster what the most prominent attributes are.
            Output are the per cluster these most distinct attributes."""
        # load umap and make clusters
        model = joblib.load(umap_pickle)
        # print(f'length df = {whole_df.info()}')
        print(f'length model = {len(model[:, 0])}')
        df = whole_df.iloc[:len(model[:, 0])]

        # TODO add cluster locations per row in df
        df['coordinates'] = list(zip(model[:, 0], model[:, 1]))

        # model = joblib.load("S5_metric=yule_nn=80_min_dis=0.3_amount=100000_AB_Resistentie.pkl")
        # plt.scatter(*model.embedding_.T, s=5, alpha=1.0)

        # train the K_means on the model coordinates
        estimator = KMeans(n_clusters=3).fit(model) # model[:, 0],model[:, 1]
        cluster_dict_first = {i: np.where(estimator.labels_ == i)[0] for i in range(estimator.n_clusters)}
        print(f'Dit is de eerste cluster dict{cluster_dict_first}')
        
        df_indices = dict()#{i: list(indices) for indices in dffor i in range(estimator.n_clusters)}

        for i in range(estimator.n_clusters):
            cluster_coords = cluster_dict_first.get(i)
            # make new dataframe from rows that have the coordinates from cluster_coords
            # print(f"Length van de series = {len(pd.Series(df['coordinates']))}, length van cluster coords = {len(cluster_coords)}")
            sub_df = df.loc[df['coordinates'].isin(cluster_coords)]
            df_indices[i] = sub_df.index.values
        # print(f'Dit zijn de indices per cluster: {df_indices}')
        print(f'HIERDOORR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

        # oneliner:
        # df_indices = {i: df[df['coordinates'].isin(cluster_dict_first.get(i))].index.values for i in range(estimator.n_clusters)}

        # add the indices per cluster to the dict
        cluster_dict = {i: (df_indices.get(i), np.where(estimator.labels_ == i)[0]) for i in range(estimator.n_clusters)}
        print(f'Dit zijn die clusters weer gecombineerd{cluster_dict}')

        # TODO first look if the df and the umap pickle have the same amount of objects
        print(f'The shape of the df = {df.shape}')
        print(f'The length of the x coordinates = {len(model[:, 0])} and the y coordinates: {len(model[:, 1])}')

        # split df into n_cluster amount of dataframes
        df_clusters = list()
        for key in cluster_dict.keys():
        
            # TODO check if we can get the rows from the df with these coordinates
            # or do we need to put an index to the the dict for each point
            
            # presumed that the loc is a list of 
            locs = [i for i in cluster_dict.get(key)[1]]
            # print(cluster_dict.get(0))
            # print(len(cluster_dict.get(0)[1]))
            # print(len(locs))
            # exit()
            new_df = df.loc[locs]
            df_clusters.append(new_df)
        
        # print(f'dit zijn blijkbaar de clusters ofzo: {df_clusters}')
        most_frequent_item = dict()
        for index, cluster in enumerate(df_clusters):
            # TODO find new information about clusters
            most_freq_per_column = list()
            # add the item that is most frequent per column and put it in the dict
            print(df_clusters[index], index)
            # print(df_clusters[index].columns)
            for column in list(cluster):
                most_freq_per_column.append(df_clusters[index][column].value_counts().idxmax())
            
            most_frequent_item[index] = most_freq_per_column

        # print( cluster_dict.items())
        # exit()
        # print the output if it goes fast else put in in a txt file (or both)
        # print(most_freq_item)
        print(type(most_frequent_item))
        print(most_frequent_item)
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        print(most_frequent_item[0])
        exit()
        print(f"The most frequent items per dict are: \n {most_frequent_item} \
            the coordinates per clusters are: \n {cluster_dict} ")
        with open('cluster_dumps.txt', 'a') as filehandle:  
            filebuffer = list(most_frequent_item) #, list(cluster_dict.items()))
            filehandle.writelines("%s\n" % line for line in filebuffer) # in lst for lst


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

            # joblib_file = str(title) + "_{}_".format(len(list(test_le_name_map.keys()))) + ".pkl"
            # joblib.dump(test_embedding, joblib_file)
            
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