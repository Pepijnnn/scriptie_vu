import pandas as pd

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.cluster import KMeans

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import itertools
import pickle
import time
from sklearn.externals import joblib
from sklearn.svm import SVC
import numpy as np
import math
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import argparse
from sklearn import preprocessing
import matplotlib as mpl
import umap
from pylab import cm
import seaborn as sns
from tqdm import tqdm
import re

from plots import Plots
from cmd_dump import CMD_info
from umap_creator import Umaps
from random_forest_how_s_to_r import RF
from neural_network import NeuralNetwork as NN

def main(**kwargs):
    print("importing csv's")
    # tab_one = pd.read_csv('../../offline_files/7 columns from mmi_Lab_MMI_Resistentie.txt', sep='\t', encoding="UTF-16")
    # tab_two = pd.read_csv('../../offline_files/8 columns from mmi_Lab_MMI_BepalingenTekst.txt', sep='\t', encoding="UTF-16")
    # tab_three = pd.read_csv('../../offline_files/9 columns from mmi_Lab_MMI_Isolaten.txt', sep='\t', encoding="UTF-16")
    # tab_four = pd.read_csv('../../offline_files/alle columns mmi_Opname_Opname.txt', sep='\t', encoding="UTF-16")  

    # tab_five = pd.read_csv('../../offline_files/mmi_Lab_MMI_Isolaten.txt', sep='\t', encoding="UTF-16", low_memory=False)  
    # tab_six = pd.read_csv('../../offline_files/mmi_Lab_MMI_Resistentie_5col.txt', sep='\t', encoding="UTF-16", low_memory=False)  
    # pandas_fields = ['MonsterNummer', 'IsolaatNummer', 'MicroOrganismeOuder', 'MateriaalDescription', 'AntibioticaNaam', 'RISV_Waarde']
    # tab_seven = pd.read_csv('../../offline_files/15 columns from BepalingTekstMetIsolatenResistentie_1maart2018_1maart2019.txt', sep='\t', encoding="UTF-16")  
    tab_eight = pd.read_csv('../../offline_files/15 columns from BepalingTekstMetIsolatenResistentie_tot_103062.txt', sep='\t', encoding="UTF-16", low_memory=False)  
    # tab_nine = pd.read_csv('../../offline_files/12 columns from BepalingTekstMetIsolatenResistentie.txt', sep='\t', encoding="UTF-16", low_memory=False)  
    tab_ten = pd.read_csv('../../offline_files/Datafile voor pepijn.txt', sep='\t', encoding="UTF-16", low_memory=False)  
    pa = pd.read_csv('../../offline_files/Pepijn_Data_Pseudomonas_aeruginosa.txt', sep='\t', encoding="UTF-16", low_memory=False)
    # per_patient_problemlist = pd.read_csv('../../offline_files/per_patient_full_6 columns from mmi_Problemlist.txt', sep='\t', encoding="UTF-16", low_memory=False)  
    # per_patient_isolaten = pd.read_csv('../../offline_files/per_patient_full_10 columns from mmi_Lab_MMI_Isolaten.txt', sep='\t', encoding="UTF-16", low_memory=False)  
    # per_patient_meddiag = pd.read_csv('../../offline_files/per_patient_full_10 columns from mmi_MedischeDiagnose.txt', sep='\t', encoding="UTF-16", low_memory=False)  
    # per_patient_opname = pd.read_csv('../../offline_files/per_patient_full_15 columns from mmi_Opname_Opname.txt', sep='\t', encoding="UTF-16", low_memory=False)  
    # per_patient_opname_p_b = pd.read_csv('../../offline_files/per_patient_full_28 columns from mmi_Opname_Opnameperiode_PerBed.txt', sep='\t', encoding="UTF-16", low_memory=False)  
    kweken_ab_opnames_01 = pd.read_csv('../../offline_files/kweken_ab_opnames_0.1.txt', sep='\t', encoding="UTF-16", low_memory=False)  
    # kweken_ab_opnames_02 = pd.read_csv('../../offline_files/kweken_ab_opnames_0.2_pid.txt', sep='\t', encoding="UTF-16", low_memory=False) 
    # kweken_ab_opnames_03 = pd.read_csv('../../offline_files/Pepijn Data zelfdoen.txt', sep='\t', encoding="UTF-16", low_memory=False) 
    # kweken_ab_opnames_04 = pd.read_csv('../../offline_files/Pepijn Data zelfdoen2.txt', sep='\t', encoding="UTF-16", low_memory=False) 
    # kweken_ab_opnames_05 = pd.read_csv('../../offline_files/Pepijn Data zelfdoen3.txt', sep='\t', encoding="UTF-16", low_memory=False) 
    # kweken_ab_opnames_06 = pd.read_csv('../../offline_files/Pepijn Data zelfdoen 4 SS bias weg.txt', sep='\t', encoding="UTF-16", low_memory=False) 
    pepijn_data_1 = pd.read_csv('../../offline_files/Pepijn Data (1).txt', sep='\t', encoding="UTF-16", low_memory=False) 
    pepijn_data_2 = pd.read_csv('../../offline_files/Pepijn Data (2).txt', sep='\t', encoding="UTF-16", low_memory=False) 
    pepijn_data_2_2 = pd.read_csv('../../offline_files/cotrim_met_dagen_opgenomen.txt', sep='\t', encoding="UTF-16", low_memory=False) 
    pepijn_data_3 = pd.read_csv('../../offline_files/Pepijn Data (3).txt', sep='\t', encoding="UTF-16", low_memory=False) 
    

    print("done importing csv's")

    def create_new_text(df, name):
        tfile = open('../../offline_files/{}.txt'.format(name), 'w+')
        tfile.write(df.to_string())
        tfile.close()

    # unsupervised learning over Isolaten table coloured with different microorganisms
    def unsup_one_table(short_table):

        # print(short_table.info())
        # exit()
        # sns.set(style='white', context='poster', rc={'figure.figsize':(14,10)})
        # select the columns you want to calculate
        # narrow_table = table[['AfnameDatum','MonsterNummer','IsolaatNummer','MicroOrganismeName', 'MateriaalCode', 'ArtsCode', 'AfdelingNaamAanvrager']]
        # short_table.drop(['Pseudo_id','Monsternummer','IsolaatNummer','MicroOrganismeName','AfnameDatum','Genus'], axis=1, inplace=True)
        short_table = short_table[['MIC_RuweWaarde','Genus','AfdelingNaamAanvrager','AntibioticaNaam','RISV_Waarde']].copy()
        short_table.drop(short_table.loc[(short_table['RISV_Waarde']=='V') | (short_table['RISV_Waarde']=='I')].index, inplace=True)
        # short_table.sort_values(by=['RISV_Waarde'], ascending=False, inplace=True)
        # rest = short_table.RISV_Waarde.value_counts()['S'] - short_table.RISV_Waarde.value_counts()['R']
        # short_table = short_table.iloc[rest:,]
        # short_table.dropna(subset=['RISV_Waarde'], inplace=True)
        short_table.fillna("0", inplace=True)
        # short_table.dropna(inplace=True)
        
        focus_table = "RISV_Waarde"

        # delete most of the "MicroOrganismeName" table for visualisation
        # m_set = list(set(narrow_table[focus_table]))
        # mm_set = m_set[:int(0.10*len(m_set))]
        # rest = m_set[int(0.10*len(m_set)):]
        # short_table = narrow_table[~narrow_table[focus_table].isin(rest)]
        
        # do the unsupervised learning where the micro organisms have different colours
        le = preprocessing.LabelEncoder()
        label = le.fit_transform(short_table[focus_table])
        labels2 = le.fit(short_table[focus_table])
        le_name_map = dict(zip(labels2.transform(le.classes_),labels2.classes_))
        short_table.drop([focus_table], axis=1, inplace=True)
        one_hot_table = pd.get_dummies(short_table)
        # one_hot_table = pd.get_dummies(short_table[['AfnameDatum','MonsterNummer','IsolaatNummer','AfdelingNaamAanvrager', 'MateriaalCode', 'ArtsCode']])

        def draw_umap(n_neighbors=50, min_dist=0.5, n_components=2, metric='yule', title='Antibiotic Resistance'):
            print("start umap")
            start = time.time()
            fit = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric)
            end = time.time()
            print(f"umap completed in {end - start} seconds")        
        
            u = fit.fit_transform(one_hot_table)
            print(f'transform completed in {time.time()-end} seconds')
            joblib_file = f"{title}_{len(one_hot_table)}.pkl"
            joblib.dump(u, joblib_file)

            # set the colourmap and amount of colours finally create the scatterplot
            cmap = cm.get_cmap('jet', len(list(le_name_map.keys()))) 
            scat = plt.scatter(u[:, 0], u[:, 1], c=label, s=1, cmap=cmap)
            cb = plt.colorbar(scat, spacing='uniform', ticks=list(le_name_map.keys()))
            cb.ax.set_yticklabels(list(le_name_map.values()))
            plt.title(title, fontsize=18)
            plt.show()
            print(short_table.shape)

        draw_umap(metric=kwargs["metric"], 
            title=("tab7'MIC_RuweWaarde','Genus',' AfdelingNaamAanvrager','AntibioticaNaam','RISV_Waarde'_metric={}, nn={}, min_dis={}, amount= All"
            .format(kwargs["metric"], kwargs["nn"], kwargs["min_dis"])), 
            n_neighbors=kwargs["nn"], 
            min_dist=kwargs["min_dis"])        

    def ten(short_table):
        print(short_table.head(10))
        for col in short_table.columns:
            print(short_table[col].describe())
        exit()
        short_table.drop(['Pseudo_id','Geslacht','IsOverleden','Postcode'], axis=1, inplace=True)
        short_table.fillna("0", inplace=True)
        for col in short_table.columns:
            if short_table[col].isnull().sum() > int(0.995*len(short_table)):
                # print("333", short_table[col].isnull().sum() > (0.995*len(short_table)))
                short_table.drop([col], axis=1, inplace=True)
            else:
                # make it from string to numerical
                short_table.loc[:, col] = pd.to_numeric(short_table[col].apply(lambda x: re.sub(',', '.', str(x))))
                # normalize the columns
                short_table.loc[:, col] = short_table.loc[:, col] / short_table.loc[:, col].max()  
                
            #   short_table[col] = pd.to_numeric(short_table[col], errors='coerce')
                # .apply(lambda x: re.sub(',', '.', str(x))
    
        one_hot_table = short_table
        
        def draw_umap(n_neighbors=30, min_dist=0.5, n_components=2, metric='yule', title='Antibiotic Res Research'):
            print("start umap")
            start = time.time()
            fit = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric)
            end = time.time()
            print(f"umap completed in {end - start} seconds")        
        
            u = fit.fit_transform(one_hot_table)
            print(f'transform completed in {time.time()-end} seconds')
            

            # plt.scatter(u[:, 0], u[:, 1], s=4)
            ax = sns.scatterplot(data=short_table,x=u[:, 0], y=u[:, 1], s=7)
            ax.set_title(title)
            # plt.title(title, fontsize=18)
            plt.show()
            joblib_file = str(f"tab10_columns_all = {len(short_table.columns)} tab10_length= {len(one_hot_table)} metric = {kwargs['metric']}.pkl")
            joblib.dump(u, joblib_file)
            with open('tab_file_column_info.txt', 'a') as filehandle:
                print("################## SPLIT START ####################", file=filehandle)
                print(title, file=filehandle)
                print(joblib_file, file=filehandle)
                print(short_table.columns, file=filehandle)
                print("################## SPLIT END ####################", file=filehandle)
                
            print(short_table.shape)

        draw_umap(metric=kwargs["metric"], title=f"tab10_metric={kwargs['metric']}, nn={kwargs['nn']}, min_dis={kwargs['min_dis']}, amount= {len(short_table)}", n_neighbors=kwargs["nn"], min_dist=kwargs["min_dis"])  


    p = Plots()
    c = CMD_info()
    u = Umaps()
    r = RF()
    nn = NN()

    # guide to the function you want to use
    if kwargs["f"] == "a":
        ten(kweken_ab_opnames_01)
        # ten(per_patient_problemlist.copy()) #.iloc[:kwargs["amount"]]
    elif kwargs["f"] == "nn":
        nn.make_nn(pepijn_data_1)
    elif kwargs["f"] == "rf":
        # r.make_test_tree(pepijn_data_1)
        r.make_test_tree(pepijn_data_2_2)
        # r.make_test_tree(pepijn_data_3)
    elif kwargs["f"] == "cnumap":
        c.pseudo_aer(pa)
        # u.create_umap_kweken_ab_opname(kweken_ab_opnames_02)
        # exit()
        umap_pickle_name = "thomasfile_flitered_length_pid = 6591 metric = sokalmichener, nn = 20, md = 0.2.pkl"
        # c.cluster_info_tho("thomasfile_flitered_length = 6591 metric = sokalmichener, nn = 20, md = 0.2.pkl", kweken_ab_opnames_01, 8)
        c.cluster_info_tho(umap_pickle_name, kweken_ab_opnames_02, tab_eight, 7)
    elif kwargs["f"] == "cumap":
        u.create_umap_per_department(tab_eight.iloc[:kwargs["amount"]])
    elif kwargs["f"] == "cmdinfo":
        umap_pickle_name, whole_df = "tab10_columns_all = 40 tab10_length= 257735 metric = euclidean.pkl", per_patient_problemlist
        c.cluster_info(umap_pickle_name, whole_df, kwargs['kmclusters']) # per_patient_isolaten, per_patient_meddiag, per_patient_opname, per_patient_opname_p_b
    elif kwargs["f"] == "sp":
        # p.cluster_info("tab10_columns_all = 40 tab10_length= 257735 metric = cosine.pkl", tab_ten.copy())
        p.simple_pickle_viewer("tab10_columns_all = 40 tab10_length= 257735 metric = euclidean.pkl", tab_ten.copy())
    elif kwargs["f"] == "sk":
        p.simple_kmeans("tab10_columns_all = 40 tab10_length= 257735 metric = euclidean.pkl", kwargs['kmclusters'])
    elif kwargs["f"] == "d":
        p.umap_over_kmeansclusters("tab10_columns_all = 40 tab10_length= 257735 metric = euclidean.pkl", tab_ten.copy())
        # p.umap_over_kmeansclusters("tab10_columns = 31 tab10_length= 50000 metric = cosine.pkl", tab_ten[:50_000].copy())
    elif kwargs["f"] == "e":
        shuffled = tab_nine.sample(frac=1).reset_index(drop=True).copy()
        best_cols(shuffled.iloc[:(kwargs["amount"])], 3)
    elif kwargs["f"] == "f":
        shuffled_five = tab_five.sample(frac=1).reset_index(drop=True).copy()
        best_parameter(shuffled_five.iloc[:kwargs["amount"]],tab_six.copy()) #.loc[:kwargs["amount"]] .loc[:kwargs["amount"]]

# t distribution kernel instead of rgb for svm
if __name__ == '__main__':
    # optimal min_dis = 0.15, metric = yule, nn= 6
    parser = argparse.ArgumentParser(description = 'Unsupervised learning function')
    parser.add_argument("--f", default="rf", help="select which function to use")
    parser.add_argument("--amount", default=400_000, help="select over how many rows you want to do the unsupervised learning")
    parser.add_argument("--nn", default=25,  help="select the amount of nn cells for the umap")
    parser.add_argument("--min_dis", default=0.0,  help="select the minimal distance for the umap")
    parser.add_argument("--metric", default="yule",  help="select which metric for the umap you want to compute")
    parser.add_argument("--kmclusters", default=8, help="select how many clusters you want to look over")
    parser.add_argument("-bestparam", action='store_true', help='calculates the best parameters for the current settings (can take hours)')
    parser.add_argument("--bestcols", help='calculates the best columns (you need to specify which number of columns) from the datafile using the current settings for the umap (can take hours)')
    args = parser.parse_args()

main(**vars(args))

# nn van 2 tot 200 (hoog) min_dist 0 tot 1 (hoog) ncomp = dimensions
# metric = Euclidean, manhattan, chebyshev, minkowski. Canberra, braycurtis(slecht), haversine(2d), mahalanobis(werkt niet), wminkowski, seuclidean(slecht). cosine, correlation. 
# Binary = hamming, jaccard, dice(y), russellrao, kulsinski, rogerstanimoto(y), sokalmichener(y), sokalsneath, yule(beste)
# print("importing csv's")
#     # tab_one = pd.read_csv('../../offline_files/7 columns from mmi_Lab_MMI_Resistentie.txt', sep='\t', encoding="UTF-16")
#     # tab_two = pd.read_csv('../../offline_files/8 columns from mmi_Lab_MMI_BepalingenTekst.txt', sep='\t', encoding="UTF-16")
#     # tab_three = pd.read_csv('../../offline_files/9 columns from mmi_Lab_MMI_Isolaten.txt', sep='\t', encoding="UTF-16")
#     # tab_four = pd.read_csv('../../offline_files/alle columns mmi_Opname_Opname.txt', sep='\t', encoding="UTF-16")  
#     tab_five = pd.read_csv('../../offline_files/mmi_Lab_MMI_Isolaten.txt', sep='\t', encoding="UTF-16", low_memory=False)  
#     tab_six = pd.read_csv('../../offline_files/mmi_Lab_MMI_Resistentie_5col.txt', sep='\t', encoding="UTF-16", low_memory=False)  
#     tab_seven = pd.read_csv('../../offline_files/pandas_merge.txt', sep='\t', encoding='cp1252')  
#     tab_eight = pd.read_csv('../../offline_files/15 columns from BepalingTekstMetIsolatenResistentie_tot_103062.txt', sep='\t', encoding="UTF-16", low_memory=False)  
#     tab_nine = pd.read_csv('../../offline_files/12 columns from BepalingTekstMetIsolatenResistentie.txt', sep='\t', encoding="UTF-16", low_memory=False)  
#     print("done importing csv's")


 # # def load_pickle(pickl_name, colour_label, key_list, value_list):
    # def load_pickle():
    #     # model = joblib.load(pickl_name)
    #     model = joblib.load("S5_metric=yule_nn=80_min_dis=0.3_amount=100000_AB_Resistentie.pkl")
    #     plt.scatter(*model.embedding_.T, s=5, alpha=1.0)
    #     # cmap = cm.get_cmap('jet', str(pickl_name).split("_ ")[-2]) 
    #     # scat = plt.scatter(*model.T, c=colour_label, s=5, cmap=cmap, alpha=1.0)

    #     # cb = plt.colorbar(scat, spacing='uniform', ticks=key_list)
    #     # cb.ax.set_yticklabels(value_list)

    #     # plt.title(str(pickl_name).split("_ ")[:-2], fontsize=18)
    #     plt.show()
    #     # "S5_metric=yule_nn=80_min_dis=0.3_amount=100000_AB_Resistentie"

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

    # # add the number of resistent bacterias found from every monster
    #   def two_to_one_df(df1, df2):
    #     df1 = df1[['MonsterNummer','IsolaatNummer','MicroOrganismeOuder', 'MateriaalDescription']].copy()
    #     df2 = df2[['MonsterNummer','IsolaatNummer','AntibioticaNaam', 'RISV_Waarde']].copy()
    #     df3 = pd.merge(df1,df2[['MonsterNummer','IsolaatNummer','AntibioticaNaam','RISV_Waarde']], on=["MonsterNummer", "IsolaatNummer"])
    #     create_new_text(df3, 'pandas_merge')



    #     draw_umap(metric=kwargs["metric"], title=("S4 metric:{}, nn:{}, min_dis:{}, amount:{} AB Resistentie"
    #                             .format(kwargs["metric"], kwargs["nn"], kwargs["min_dis"], kwargs["amount"])), 
    #         n_neighbors=kwargs["nn"], min_dist=kwargs["min_dis"])
    # def combined_dataframe(df1, df2):
    #     # df = pd.read_csv('../../offline_files/mmi_Lab_MMI_Resistentie_5col.txt', sep='\t', encoding="UTF-16")  
    #     df1 = df1[['MonsterNummer','IsolaatNummer','MicroOrganismeOuder', 'MateriaalDescription']].copy()
    #     df2 = df2[['MonsterNummer','IsolaatNummer','AntibioticaNaam', 'RISV_Waarde']].copy()
    #     df3 = pd.merge(df1,df2[['MonsterNummer','IsolaatNummer','AntibioticaNaam','RISV_Waarde']], on=["MonsterNummer", "IsolaatNummer"])
        
    #     df3 = df3.drop(['MonsterNummer'], axis=1).copy()
    #     df3 = df3.drop(['IsolaatNummer'], axis=1).copy()

    #     # fill other NaN's with most frequent string in column an drop Not Important columns
    #     for col in df3.columns:
    #         df3[col].fillna("0", inplace=True)

    #     # put colours relative to amount AB res column and create the clusters using umap and show them
    #     focus_table = "RISV_Waarde"
    #     le = preprocessing.LabelEncoder()
    #     label = le.fit_transform(df3[focus_table])
    #     labels2 = le.fit(df3[focus_table])
    #     le_name_map = dict(zip(labels2.transform(le.classes_),labels2.classes_))
    #     one_hot_table = pd.get_dummies(df3[["AntibioticaNaam","MicroOrganismeOuder","MateriaalDescription"]])
    #     # one_hot_table = pd.get_dummies(df1)

    #     def draw_umap(n_neighbors=50, min_dist=0.5, n_components=2, metric='yule', title='Antibiotic Resistance'):
    #         fit = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric)
    #         u = fit.fit_transform(one_hot_table)

    #         # set the colourmap and amount of colours finally create the scatterplot
    #         cmap = cm.get_cmap('jet', len(list(le_name_map.keys()))) 
    #         scat = plt.scatter(u[:, 0], u[:, 1], c=label, s=10, cmap=cmap)
 
    #         cb = plt.colorbar(scat, spacing='uniform', ticks=list(le_name_map.keys()))
    #         cb.ax.set_yticklabels(list(le_name_map.values()))
            
    #         plt.title(title, fontsize=18)
    #         plt.show()
    #         print(df3.shape)
    #sel1_datatype = [["Monsternummer","StudieNummer","MateriaalShortName","WerkplekCode","BepalingCode","ArtsCode","AfdelingCodeAanvrager","Locatie","Waarde","Uitslag"],
    #["Monsternummer","IsolaatNummer","MicroOrganismeCode","AfnameDatum","ArtsCode","AfdelingCodeAanvrager","AfdelingNaamAanvrager","AfdelingKliniekPoliAanvrager","OrganisatieCodeAanvrager","OrganisatieNaamAanvrager","StudieNummer","MicroOrganismeOuder","MicroOrganismeOuderOuder","MicroBiologieProcedureCode","MicroOrganismeName","MicroOrganismeType","MicroOrganismeParentCode","MateriaalCode","Kingdom","PhylumDivisionGroup","Class","Order","Family","Genus","MateriaalDescription","MateriaalShortName","ExternCommentaar","TimeStamp"],
    #["Monsternummer","LabIndicator","AfnameDatum","BepalingsCode","IsolaatNummer","AntibioticaNaam","AB_Code","Methode","MIC_RuweWaarde","E_TestRuweWaarde","AgarDiffRuweWaarde","RISV_Waarde","TimeStamp"],
    #["VoorschriftId","Pseudo_id","OpnameID","Startmoment","Status_naam","Snelheid","Snelheidseenheid","Dosis","DosisEenheid","Toedieningsroute","MedicatieArtikelCode","MedicatieArtikelNaam","MedicatieArtikelATCcode","MedicatieArtikelATCnaam","FarmaceutischeKlasse","FarmaceutischeSubklasse","TherapeutischeKlasse","Werkplek_code","Werkplek_omschrijving","Bron"],
    #["Pseudo_id","Geslacht","Geboortedatum","Overlijdensdatum","IsOverleden","Land"]]  