import pandas as pd
import matplotlib
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import itertools
import pickle
from sklearn.externals import joblib
from sklearn.svm import SVC
import numpy as np
import math
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import argparse
from sklearn import preprocessing
import matplotlib as mpl
import umap
from pylab import cm
import seaborn as sns
from tqdm import tqdm

#sel1_datatype = [["Monsternummer","StudieNummer","MateriaalShortName","WerkplekCode","BepalingCode","ArtsCode","AfdelingCodeAanvrager","Locatie","Waarde","Uitslag"],
#["Monsternummer","IsolaatNummer","MicroOrganismeCode","AfnameDatum","ArtsCode","AfdelingCodeAanvrager","AfdelingNaamAanvrager","AfdelingKliniekPoliAanvrager","OrganisatieCodeAanvrager","OrganisatieNaamAanvrager","StudieNummer","MicroOrganismeOuder","MicroOrganismeOuderOuder","MicroBiologieProcedureCode","MicroOrganismeName","MicroOrganismeType","MicroOrganismeParentCode","MateriaalCode","Kingdom","PhylumDivisionGroup","Class","Order","Family","Genus","MateriaalDescription","MateriaalShortName","ExternCommentaar","TimeStamp"],
#["Monsternummer","LabIndicator","AfnameDatum","BepalingsCode","IsolaatNummer","AntibioticaNaam","AB_Code","Methode","MIC_RuweWaarde","E_TestRuweWaarde","AgarDiffRuweWaarde","RISV_Waarde","TimeStamp"],
#["VoorschriftId","Pseudo_id","OpnameID","Startmoment","Status_naam","Snelheid","Snelheidseenheid","Dosis","DosisEenheid","Toedieningsroute","MedicatieArtikelCode","MedicatieArtikelNaam","MedicatieArtikelATCcode","MedicatieArtikelATCnaam","FarmaceutischeKlasse","FarmaceutischeSubklasse","TherapeutischeKlasse","Werkplek_code","Werkplek_omschrijving","Bron"],
#["Pseudo_id","Geslacht","Geboortedatum","Overlijdensdatum","IsOverleden","Land"]]  

def main(**kwargs):
    tab_one = pd.read_csv('../../offline_files/7 columns from mmi_Lab_MMI_Resistentie.txt', sep='\t', encoding="UTF-16")
    tab_two = pd.read_csv('../../offline_files/8 columns from mmi_Lab_MMI_BepalingenTekst.txt', sep='\t', encoding="UTF-16")
    tab_three = pd.read_csv('../../offline_files/9 columns from mmi_Lab_MMI_Isolaten.txt', sep='\t', encoding="UTF-16")
    tab_four = pd.read_csv('../../offline_files/alle columns mmi_Opname_Opname.txt', sep='\t', encoding="UTF-16")  
    tab_five = pd.read_csv('../../offline_files/mmi_Lab_MMI_Isolaten.txt', sep='\t', encoding="UTF-16")  
    tab_six = pd.read_csv('../../offline_files/mmi_Lab_MMI_Resistentie_5col.txt', sep='\t', encoding="UTF-16")  
    # tab_seven = pd.read_csv('../../offline_files/combined_df.txt', sep='\t', encoding="UTF-16")  
    tab_eight = pd.read_csv('../../offline_files/15 columns from BepalingTekstMetIsolatenResistentie_tot_103062.txt', sep='\t', encoding="UTF-16")  
    
    

    # drop non-important columns
    def drop_ni_columns(df):
        df = df.drop(['AfnameDatum'], axis=1)
        df = df.drop(['IsolaatNummer'], axis=1)
        df = df.drop(['Pseudo_id'], axis=1)
        df = df.drop(['MicroOrganismeCode'], axis=1)
        return df

    def create_new_text(df, name):
        tfile = open('../../offline_files/{}.txt'.format(name), 'w+')
        tfile.write(df.to_string())
        tfile.close()

    def two_to_one_df(df1, df2):
        df1 = df1[['MonsterNummer','IsolaatNummer','MicroOrganismeOuder', 'MateriaalDescription']].copy()
        df2 = df2[['MonsterNummer','IsolaatNummer','AntibioticaNaam', 'RISV_Waarde']].copy()
        df3 = pd.merge(df1,df2[['MonsterNummer','IsolaatNummer','AntibioticaNaam','RISV_Waarde']], on=["MonsterNummer", "IsolaatNummer"])
        create_new_text(df3, 'pandas_merge')

    # add the number of resistent bacterias found from every monster
    def combined_dataframe(df1, df2):
        # df = pd.read_csv('../../offline_files/mmi_Lab_MMI_Resistentie_5col.txt', sep='\t', encoding="UTF-16")  
        df1 = df1[['MonsterNummer','IsolaatNummer','MicroOrganismeOuder', 'MateriaalDescription']].copy()
        df2 = df2[['MonsterNummer','IsolaatNummer','AntibioticaNaam', 'RISV_Waarde']].copy()
        df3 = pd.merge(df1,df2[['MonsterNummer','IsolaatNummer','AntibioticaNaam','RISV_Waarde']], on=["MonsterNummer", "IsolaatNummer"])
        
        df3 = df3.drop(['MonsterNummer'], axis=1).copy()
        df3 = df3.drop(['IsolaatNummer'], axis=1).copy()

        # fill other NaN's with most frequent string in column an drop Not Important columns
        for col in df3.columns:
            df3[col].fillna("0", inplace=True)
            # df3[col]= df3[col].fillna(df3[col].value_counts().idxmax())
        # df3 = drop_ni_columns(df3)

        # put colours relative to amount AB res column and create the clusters using umap and show them
        focus_table = "RISV_Waarde"
        le = preprocessing.LabelEncoder()
        label = le.fit_transform(df3[focus_table])
        labels2 = le.fit(df3[focus_table])
        le_name_map = dict(zip(labels2.transform(le.classes_),labels2.classes_))
        one_hot_table = pd.get_dummies(df3[["AntibioticaNaam","MicroOrganismeOuder","MateriaalDescription"]])
        # one_hot_table = pd.get_dummies(df1)

        def draw_umap(n_neighbors=50, min_dist=0.5, n_components=2, metric='yule', title='Antibiotic Resistance'):
            fit = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric)
            u = fit.fit_transform(one_hot_table)

            # set the colourmap and amount of colours finally create the scatterplot
            cmap = cm.get_cmap('jet', len(list(le_name_map.keys()))) 
            scat = plt.scatter(u[:, 0], u[:, 1], c=label, s=10, cmap=cmap)
 
            cb = plt.colorbar(scat, spacing='uniform', ticks=list(le_name_map.keys()))
            cb.ax.set_yticklabels(list(le_name_map.values()))
            
            plt.title(title, fontsize=18)
            plt.show()
            print(df3.shape)

        draw_umap(metric=kwargs["metric"], title=("S4 metric:{}, nn:{}, min_dis:{}, amount:{} AB Resistentie"
                                .format(kwargs["metric"], kwargs["nn"], kwargs["min_dis"], kwargs["amount"])), 
            n_neighbors=kwargs["nn"], min_dist=kwargs["min_dis"])

    # unsupervised learning over Isolaten table coloured with different microorganisms
    def unsup_one_table(table):
        # sns.set(style='white', context='poster', rc={'figure.figsize':(14,10)})
        # print(list(table))
        # exit()
        # select the columns you want to calculate
        narrow_table = table[['AfnameDatum','MonsterNummer','IsolaatNummer','MicroOrganismeName', 'MateriaalCode', 'ArtsCode', 'AfdelingNaamAanvrager']]
        # fill NaN's with most frequent string from that column
        for col in narrow_table.columns:
            if col == "AfnameDatum" or col == "MonsterNummer" or col == "IsolaatNummer":
                narrow_table[col].fillna(0, inplace=True)
            else:
                narrow_table[col].fillna("0", inplace=True) #table[col].value_counts().idxmax()
        
        # Delete part of rows
        # narrow_table["AfdelingNaamAanvrager"] = narrow_table["AfdelingNaamAanvrager"].fillna("0", inplace=True)
        # short_table = narrow_table[narrow_table.AfdelingNaamAanvrager.str.contains("Polikliniek")]

        focus_table = "MicroOrganismeName"

        # delete most of the "MicroOrganismeName" table for visualisation
        m_set = list(set(narrow_table[focus_table]))
        # mm_set = m_set[:int(0.10*len(m_set))]
        rest = m_set[int(0.10*len(m_set)):]
        short_table = narrow_table[~narrow_table[focus_table].isin(rest)]
        
        # do the unsupervised learning where the micro organisms have different colours
        le = preprocessing.LabelEncoder()
        label = le.fit_transform(short_table[focus_table])
        labels2 = le.fit(short_table[focus_table])
        le_name_map = dict(zip(labels2.transform(le.classes_),labels2.classes_))
        one_hot_table = pd.get_dummies(short_table[['AfnameDatum','MonsterNummer','IsolaatNummer','AfdelingNaamAanvrager', 'MateriaalCode', 'ArtsCode']])

        def draw_umap(n_neighbors=50, min_dist=0.5, n_components=2, metric='yule', title='Antibiotic Resistance'):
            fit = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric)
            u = fit.fit_transform(one_hot_table)

            # set the colourmap and amount of colours finally create the scatterplot
            cmap = cm.get_cmap('jet', len(list(le_name_map.keys()))) 
            scat = plt.scatter(u[:, 0], u[:, 1], c=label, s=10, cmap=cmap)

            cb = plt.colorbar(scat, spacing='uniform', ticks=list(le_name_map.keys()))
            cb.ax.set_yticklabels(list(le_name_map.values()))
            
            plt.title(title, fontsize=18)
            plt.show()
            print(short_table.shape)

        draw_umap(metric=kwargs["metric"], 
            title=("S4 metric:{}, nn:{}, min_dis:{}, amount:{} MicroOrganismeNaam"
            .format(kwargs["metric"], kwargs["nn"], kwargs["min_dis"], kwargs["amount"])), 
            n_neighbors=kwargs["nn"], 
            min_dist=kwargs["min_dis"])

    def supervised(df1, df2):
        # combine part of the data frames 
        df1 = df1[['MonsterNummer','IsolaatNummer','MicroOrganismeOuder', 'MateriaalDescription']].copy()
        df2 = df2[['MonsterNummer','IsolaatNummer','AntibioticaNaam', 'RISV_Waarde']].copy()
        df3 = pd.merge(df1,df2[['MonsterNummer','IsolaatNummer','AntibioticaNaam','RISV_Waarde']], on=["MonsterNummer", "IsolaatNummer"])

        # make sure that the amout of R and S rows are the same
        df3.drop(df3.loc[(df3['RISV_Waarde']=='V') | (df3['RISV_Waarde']=='I')].index, inplace=True)
        df3.sort_values(by=['RISV_Waarde'], ascending=False, inplace=True)
        rest = df3.RISV_Waarde.value_counts()['S'] - df3.RISV_Waarde.value_counts()['R']
        df3 = df3.iloc[rest:,]
        
        # drop the columns that we don't want to train on
        df3.drop(['MonsterNummer', 'IsolaatNummer'], inplace=True, axis=1)

        # fill other NaN's with most frequent string in column an drop Not Important columns
        for col in df3.columns:
            df3[col].fillna("0", inplace=True)

        # put colours relative to amount AB res column and create the clusters using umap and show them
        X = pd.get_dummies(df3[["AntibioticaNaam","MicroOrganismeOuder","MateriaalDescription"]])
        y = df3.RISV_Waarde
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 40)

        le = preprocessing.LabelEncoder()
        label = le.fit_transform(y)
        labels2 = le.fit(y)
        le_name_map = dict(zip(labels2.transform(le.classes_),labels2.classes_))

        train_le = preprocessing.LabelEncoder()
        train_label = train_le.fit_transform(y_train)
        train_labels2 = train_le.fit(y_train)
        train_le_name_map = dict(zip(train_labels2.transform(train_le.classes_),train_labels2.classes_))

        test_le = preprocessing.LabelEncoder()
        test_label = test_le.fit_transform(y_test)
        test_labels2 = test_le.fit(y_test)
        test_le_name_map = dict(zip(test_labels2.transform(test_le.classes_),test_labels2.classes_))

        def draw_umap(metric=kwargs["metric"], n_neighbors=kwargs["nn"], n_components=2, min_dist=kwargs["min_dis"], title='Antibiotic Resistance'):
            trans = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric, random_state=42).fit(X_train)
            svc = SVC(gamma = 'auto').fit(trans.embedding_, y_train)
            test_embedding = trans.transform(X_test)
            print(svc.score(test_embedding, y_test))
            print(f'the metric used was: {kwargs["metric"]}, the n_neighbors: {kwargs["nn"]}, the min_distance: {kwargs["min_dis"]}')
            exit()

            # for printing the plots to see how well the train or the test proces went
            mapper = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric).fit(X_train, y=train_label)
            test_embedding = mapper.transform(X_test)

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
            cmap = cm.get_cmap('jet', len(list(test_le_name_map.keys()))) 
            scat = plt.scatter(*test_embedding.T, c=test_label, s=5, cmap=cmap, alpha=1.0)
 
            cb = plt.colorbar(scat, spacing='uniform', ticks=list(test_le_name_map.keys()))
            cb.ax.set_yticklabels(list(test_le_name_map.values()))
            
            plt.title(title, fontsize=18)
            plt.show()
            print(df3.shape)

        draw_umap(metric=kwargs["metric"], 
            title=("S5_metric={}_nn={}_min_dis={}_amount={}_AB_Resistentie"
            .format(kwargs["metric"], kwargs["nn"], kwargs["min_dis"], kwargs["amount"])), 
            n_neighbors=kwargs["nn"], 
            min_dist=kwargs["min_dis"])
    
    def load_pickle(pickl_name, colour_label, key_list, value_list):
        model = joblib.load(pickl_name)
        cmap = cm.get_cmap('jet', str(pickl_name).split("_ ")[-2]) 
        scat = plt.scatter(*model.T, c=colour_label, s=5, cmap=cmap, alpha=1.0)

        cb = plt.colorbar(scat, spacing='uniform', ticks=key_list)
        cb.ax.set_yticklabels(value_list)

        plt.title(str(pickl_name).split("_ ")[:-2], fontsize=18)
        plt.show()
        # "S5_metric=yule_nn=80_min_dis=0.3_amount=100000_AB_Resistentie"

    def best_cols(df, col_pred_amount):
        """ input is a large dataframe which contains certain columns and a 'RISV_Waarde' column
            output is .txt file consisting of 2+ columns with its prediction of the 'RISV_Waarde' column"""

        # make the database 50/50 of R and S counts in the RISV_Waarde column
        df.drop(df.loc[(df['RISV_Waarde']=='V') | (df['RISV_Waarde']=='I')].index, inplace=True)
        df.sort_values(by=['RISV_Waarde'], ascending=False, inplace=True)
        rest = df.RISV_Waarde.value_counts()['S'] - df.RISV_Waarde.value_counts()['R']
        df = df.iloc[rest:,].copy()

        for col in df.columns:
            df[col].fillna("0", inplace=True)

        # split the predicton column of the df and remove the biased columns
        y = df['RISV_Waarde']
        df.drop(['Monsternummer', 'IsolaatNummer', 'RISV_Waarde', 'Pseudo_id'], inplace=True, axis=1)

        # split the column in col_pred_amount amount of tuples for each possible combination
        df_col_combinations = list(set(itertools.combinations(list(df), col_pred_amount)))

        # test for each combination of columns how well they predict R or S
        columns_score = []
        for i in tqdm(range(len(df_col_combinations))):
            col_1, col_2  = df_col_combinations[i]
            X = pd.get_dummies(df[[col_1, col_2]])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 40)

            trans = umap.UMAP(n_neighbors=kwargs["nn"], min_dist=kwargs["min_dis"], n_components=2, metric=kwargs["metric"], random_state=42).fit(X_train)
            svc = SVC(gamma = 'auto').fit(trans.embedding_, y_train)
            test_embedding = trans.transform(X_test)
            print(df_col_combinations[i], svc.score(test_embedding, y_test))
            
            columns_score.append([str(col_1), str(col_2), svc.score(test_embedding, y_test)])
        
        # write results to a .txt file
        data_frame = pd.DataFrame(columns_score, columns = ['col_1', 'col_2', 'score'])
        data_frame.to_csv(f'best_cols_{kwargs["amount"]}.txt',index=False)

    def best_parameter(df1, df2):
        """ input are two specific datadrames that need to have the next columns:
            df1: ['MonsterNummer','IsolaatNummer','MicroOrganismeOuder', 'MateriaalDescription']
            df2: ['MonsterNummer','IsolaatNummer','AntibioticaNaam', 'RISV_Waarde']
            output is a .txt file where we see the best parameters of the umap for these dataframes"""

        # combine part of the data frames 
        df1 = df1[['MonsterNummer','IsolaatNummer','MicroOrganismeOuder', 'MateriaalDescription']].copy()
        df2 = df2[['MonsterNummer','IsolaatNummer','AntibioticaNaam', 'RISV_Waarde']].copy()
        df3 = pd.merge(df1,df2[['MonsterNummer','IsolaatNummer','AntibioticaNaam','RISV_Waarde']], on=["MonsterNummer", "IsolaatNummer"])

        # make sure that the amout of R and S rows are the same
        df3.drop(df3.loc[(df3['RISV_Waarde']=='V') | (df3['RISV_Waarde']=='I')].index, inplace=True)
        df3.sort_values(by=['RISV_Waarde'], ascending=False, inplace=True)
        rest = df3.RISV_Waarde.value_counts()['S'] - df3.RISV_Waarde.value_counts()['R']
        df3 = df3.iloc[rest:,]
        
        # drop the columns that we don't want to train on
        df3.drop(['MonsterNummer', 'IsolaatNummer'], inplace=True, axis=1)

        # fill other NaN's with most frequent string in column an drop Not Important columns
        for col in df3.columns:
            df3[col].fillna("0", inplace=True)

        # put colours relative to amount AB res column and create the clusters using umap and show them
        X = pd.get_dummies(df3[["AntibioticaNaam","MicroOrganismeOuder","MateriaalDescription"]])
        y = df3.RISV_Waarde
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 40)

        # these are the umap parameters the algorithm is going to calculate
        tuning_parameters = {'n_neighbors' : [2, 3, 4, 5, 6],
                              'min_dist' : [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
                              'metric' : ['sokalsneath', 'yule']
                            }
        # calculate the SVC of all the options and write them to a text file
        params = []
        for i in tuning_parameters.get('metric'):
            for j in tuning_parameters.get('n_neighbors'):
                for k in tuning_parameters.get('min_dist'):
                    trans = umap.UMAP(n_neighbors=j, min_dist=k, n_components=2, metric=i, random_state=41).fit(X_train)
                    svc = SVC(gamma = 'auto').fit(trans.embedding_, y_train)
                    test_embedding = trans.transform(X_test)
                    print(svc.score(test_embedding, y_test))
                    params.append([i, j, k, svc.score(test_embedding, y_test)])
                    print(f'the metric used was: {i}, the n_neighbors: {j}, the min_distance: {k}')
        data_frame = pd.DataFrame(params, columns = ['metric', 'nearest_neighbors', 'min_distance', 'score'])
        data_frame.to_csv(f'hyper_params_{kwargs["amount"]}.txt',index=False)


    if kwargs["f"] == "a":
        unsup_one_table(tab_five.loc[:kwargs["amount"]].copy())
    elif kwargs["f"] == "b":
        combined_dataframe(tab_five.loc[:kwargs["amount"]].copy(),tab_six.loc[:kwargs["amount"]].copy())  
    elif kwargs["f"] == "c":
        two_to_one_df(tab_five.copy(),tab_six.copy())
    elif kwargs["f"] == "d":
        supervised(tab_five.loc[:kwargs["amount"]].copy(),tab_six.loc[:kwargs["amount"]].copy()) #.loc[:kwargs["amount"]] .loc[:kwargs["amount"]]
    elif kwargs["f"] == "e":
        best_cols(tab_eight.loc[:(kwargs["amount"])].copy(), 2)
    elif kwargs["f"] == "f":
        best_parameter(tab_five.loc[:kwargs["amount"]].copy(),tab_six.loc[:kwargs["amount"]].copy()) #.loc[:kwargs["amount"]] .loc[:kwargs["amount"]]

if __name__ == '__main__':
    # optimal min_dis = 0.15, metric = yule, nn= 6
    parser = argparse.ArgumentParser(description = 'Unsupervised learning function')
    parser.add_argument("--f", default="f", help="select which function to use")
    parser.add_argument("--amount", default=50_000, help="select over how many rows you want to do the unsupervised learning")
    parser.add_argument("--nn", default = 6,  help="select the amount of nn cells for the umap")
    parser.add_argument("--min_dis", default = 0.15,  help="select the minimal distance for the umap")
    parser.add_argument("--metric", default = "yule",  help="select which metric for the umap you want to compute")
    args = parser.parse_args()

main(**vars(args))

# nn van 2 tot 200 (hoog) min_dist 0 tot 1 (hoog) ncomp = dimensions
# metric = Euclidean, manhattan, chebyshev, minkowski. Canberra, braycurtis(slecht), haversine(2d), mahalanobis(werkt niet), wminkowski, seuclidean(slecht). cosine, correlation. 
# Binary = hamming, jaccard, dice(y), russellrao, kulsinski, rogerstanimoto(y), sokalmichener(y), sokalsneath, yule(beste)



# TESTTTTTTTTTTTTTTTTTTT DEL AFTER
        # X = pd.get_dummies(df3)
        # le = preprocessing.LabelEncoder()
        # y = le.fit_transform(df3["RISV_Waarde"])
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 40)

        # tuning_parameters = [{'n_neighbors' : [2, 10, 20, 30, 40, 50, 75 , 100, 150],
        #                       'min_dist' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        #                       'metric' : ['hamming', 'jaccard', 'dice', 'russellrao', 'kulsinski', 'rogerstanimoto', 'sokalmichener', 'sokalsneath', 'yule']
        #                     }]
        # clf = GridSearchCV(umap.UMAP(), tuning_parameters, cv=3, scoring='balanced_accuracy')
        # clf.fit(X_train, y_train)
        # print(clf.best_params_)
        # exit()

