import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.cluster import KMeans


class CMD_info():

    def create_perc_df(self, df):
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

    def cluster_info_tho(self, umap_pickle, df, n_clusters):
        """ Input is a umap and its dataframe.
            - First it searches an amount of clusters with Kmeans clustering.
            - Then it searces for each cluster what the most prominent attributes are.
            Output are the per cluster these most distinct attributes."""
        # load umap and make clusters
        model = joblib.load(umap_pickle)
        print(f'length model = {len(model[:, 0])}')

        whole_df = self.create_perc_df(df)

        # add cluster locations per row in df
        whole_df['coordinates'] = list(zip(model[:, 0], model[:, 1]))

        # train the K_means on the model coordinates
        estimator = KMeans(n_clusters=n_clusters, random_state=0).fit(model) # model[:, 0],model[:, 1]
        # exit()
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
            new_df = whole_df.iloc[locs]
            
            # new_df = whole_df[whole_df['Pseudo_id'].isin(locs)]
            
            # print the information about each column for each cluster to the command prompt
            print(f"#################################### CLUSTER {key}  ##########################################################################################")
            print(new_df['coordinates'].head(5))
            for column in new_df.columns:
                print(column, len(new_df[column].dropna()))
                if column == 'Pseudo_id':
                    print(new_df[column].dropna().astype(object).describe())
                    continue
                try:
                    print(pd.to_numeric(new_df[column].dropna().str.replace(',', '.').astype(float)).describe())
                except AttributeError:
                    # print(pd.to_numeric(new_df[column].dropna().replace(',', '.').astype(object)).describe())
                    print(new_df[column].astype(float).describe())
                except ValueError:
                    
                    print(new_df[column].dropna().astype(object).describe())
                # exit()

        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib import pyplot as plt
        
        plt.scatter(model[:, 0], model[:, 1], s=1, c=estimator.labels_.astype(float), cmap='Spectral')
        plt.show()
                 

    def cluster_info(self, umap_pickle, whole_df, n_clusters=7):
        """ Input is a umap and its dataframe.
            - First it searches an amount of clusters with Kmeans clustering.
            - Then it searces for each cluster what the most prominent attributes are.
            Output are the per cluster these most distinct attributes."""
        # load umap and make clusters
        model = joblib.load(umap_pickle)
        print(f'length model = {len(model[:, 0])}')
        # df = whole_df.iloc[:len(model[:, 0])]
        
        print(whole_df['Pseudo_id'].astype(object).describe())

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
            
            # print the information about each column for each cluster to the command prompt
            print(f"#################################### CLUSTER {key}  ##########################################################################################")
            # print(new_df['coordinates'].head(5))
            for column in new_df.columns:
                print(column, len(new_df[column].dropna()))
                if column == 'Pseudo_id':
                    print(new_df[column].dropna().astype(object).describe())
                    continue
                try:
                    print(pd.to_numeric(new_df[column].dropna().str.replace(',', '.').astype(float)).describe())
                except AttributeError:
                    print(pd.to_numeric(new_df[column].dropna().replace(',', '.').astype(float)).describe())
                except ValueError:
                    print(new_df[column].dropna().describe())
                 


################################# OLD CODE ##############################
# def index_to_pseudo_dict(self, df, cluster_dict_first):
#     df_indices = dict()
#     for i in range(estimator.n_clusters):
#         cluster_coords = cluster_dict_first.get(i)
#         # make new dataframe from rows that have the coordinates from cluster_coords
#         sub_df = df.loc[df['coordinates'].isin(cluster_coords)]
#         df_indices[i] = sub_df.index.values

#     # add the indices per cluster to the dict
#     cluster_dict = {i: (df_indices.get(i), np.where(estimator.labels_ == i)[0])[1] for i in range(estimator.n_clusters)}
#     print(f'Dit zijn die clusters weer gecombineerd{cluster_dict}')
#     return cluster_dict