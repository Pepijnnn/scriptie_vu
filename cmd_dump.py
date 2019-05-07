import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.cluster import KMeans


class CMD_info():

    def cluster_info(self, umap_pickle, whole_df, n_clusters=7):
        """ Input is a umap and its dataframe.
            - First it searches an amount of clusters with Kmeans clustering.
            - Then it searces for each cluster what the most prominent attributes are.
            Output are the per cluster these most distinct attributes."""
        # load umap and make clusters
        model = joblib.load(umap_pickle)
        print(f'length model = {len(model[:, 0])}')
        # df = whole_df.iloc[:len(model[:, 0])]
        
        # add cluster locations per row in df
        # df['coordinates'] = list(zip(model[:, 0], model[:, 1]))

        # train the K_means on the model coordinates
        estimator = KMeans(n_clusters=n_clusters, random_state=0).fit(model) # model[:, 0],model[:, 1]
        print("Fitted")

        # dict van cluster naar pseudoID (als de index overeenkomt met pseudoID ander dat oldcode stuk erin plakken)
        cluster_dict = {i: np.where(estimator.labels_ == i)[0] for i in range(estimator.n_clusters)}
        print(f'Dit is de eerste cluster dict{cluster_dict}')
        
        # first look if the df and the umap pickle have the same amount of objects
        print(f'The shape of the df = {whole_df.shape}')
        print(f'The length of the x coordinates = {len(model[:, 0])} and the y coordinates: {len(model[:, 1])}')

        # split df into n_cluster amount of dataframes
        for key in cluster_dict.keys():
            locs = [i for i in cluster_dict.get(key)]
            # new_df = whole_df.loc[locs]
            new_df = whole_df[whole_df['Pseudo_id'].isin(locs)]
            # print(new_df.head())
            # print(whole_df.head())
            # exit()
            # print the information about each column for each cluster to the command prompt
            print(f"#################################### CLUSTER {key}  ##########################################################################################")
            # print(new_df['coordinates'].head(5))
            for column in new_df.columns:
                print(column, len(new_df[column].dropna()))
                try:
                    print(pd.to_numeric(new_df[column].dropna().str.replace(',', '.').astype(float)).describe())
                except AttributeError:
                    print(pd.to_numeric(new_df[column].dropna().replace(',', '.').astype(float)).describe())
                except ValueError:
                    print(new_df[column].dropna().describe())
                 


################################# OLD CODE ##############################

# df_indices = dict()
# for i in range(estimator.n_clusters):
#     cluster_coords = cluster_dict_first.get(i)
#     # make new dataframe from rows that have the coordinates from cluster_coords
#     sub_df = df.loc[df['coordinates'].isin(cluster_coords)]
#     df_indices[i] = sub_df.index.values

# add the indices per cluster to the dict
# cluster_dict = {i: (df_indices.get(i), np.where(estimator.labels_ == i)[0])[1] for i in range(estimator.n_clusters)}
# print(f'Dit zijn die clusters weer gecombineerd{cluster_dict}')