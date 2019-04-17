import pandas as pd
import matplotlib
from sklearn.utils import shuffle
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
    tab_eight = pd.read_csv('../../offline_files/pandas_merge.txt', sep='\t', encoding="UTF-16-be")  
    
    

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
        # # add risv value as column to df1
        # # df1["RISV_Waarde"] = np.nan
        # # for mon_nr, isolaat, risv in zip(df2["MonsterNummer"], df2["IsolaatNummer"], df2["RISV_Waarde"]):
        # #     if str(risv) == "R":
        # #         for i, (mon_nr2, isolaat2) in enumerate(zip(df1["MonsterNummer"], df1["IsolaatNummer"])):
        # #             if mon_nr == mon_nr2 and isolaat == isolaat2: 
        # #                 df1["RISV_Waarde"][i] = risv
        # # df1["RISV_Waarde"].fillna(0, inplace = True)

        # # count for each monsternummer there was a resistant bacteria
        # mns = list()
        # for c, num in enumerate(df2["RISV_Waarde"]):
        #     if str(num) == "R":
        #         mns.append((df2["MonsterNummer"][c], df2["IsolaatNummer"][c]))
        # count_dict = {x:mns.count(x) for x in mns}
        
        # # add the antibioticanaam as a new column concat join
        # df1["AntibioticaNaam"] = np.nan
        # for mon_nr, isolaat, abn in zip(df2["MonsterNummer"], df2["IsolaatNummer"], df2["AntibioticaNaam"]):
        #     for i, (mon_nr2, isolaat2) in enumerate(zip(df1["MonsterNummer"], df1["IsolaatNummer"])):
        #         if mon_nr == mon_nr2 and isolaat == isolaat2: 
        #             df1["AntibioticaNaam"][i] = abn
        #             break
        # df1["AntibioticaNaam"].fillna("0", inplace = True)
        
        # # add new colomn of previous count file and fill the NaNs with 0's
        # df1["ResAB_amount"] = np.nan
        # for c, (mn, isn) in tqdm(enumerate(zip(df1["MonsterNummer"], df1["IsolaatNummer"]))):
        #     df1["ResAB_amount"].loc[c] = count_dict.get((mn, isn), 0) # build-in else statement

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

        draw_umap(metric=kwargs["metric"], title="S4 metric:{}, nn:{}, min_dis:{}, amount:{} AB Resistentie".format(kwargs["metric"], kwargs["nn"], kwargs["min_dis"], kwargs["amount"]), 
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
    
        # nn van 2 tot 200 (hoog) min_dist 0 tot 1 (hoog) ncomp = dimensions
        # metric = Euclidean, manhattan, chebyshev, minkowski. Canberra, braycurtis(slecht), haversine(2d), mahalanobis(werkt niet), wminkowski, seuclidean(slecht). cosine, correlation. 
        # Binary = hamming, jaccard, dice(y), russellrao, kulsinski, rogerstanimoto(y), sokalmichener(y), sokalsneath, yule(beste)

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

        draw_umap(metric=kwargs["metric"], title="S4 metric:{}, nn:{}, min_dis:{}, amount:{} MicroOrganismeNaam".format(kwargs["metric"], kwargs["nn"], kwargs["min_dis"], kwargs["amount"]), 
            n_neighbors=kwargs["nn"], min_dist=kwargs["min_dis"])

    # don't know yet what is going to happen here
    def extravalues(df1, df2, df3, df4):
        # count for each monsternummer there was a resistant bacteria and add the ab_name
        mns, ab_naam = [], []
        for c, num in enumerate(df2["RISV_Waarde"]):
            if str(num) == "R":
                mns.append(df2["MonsterNummer"][c])
                ab_naam.append([df2["MonsterNummer"][c],df2["IsolaatNummer"][c],df2["AntibioticaNaam"][c]])
        count_dict = {x:mns.count(x) for x in mns}
        
        df1["ResAB_amount"] = np.nan
        df1["Res_antibiotica"] = np.nan
        df1["Res_antibiotica2"] = np.nan
        for c, num in enumerate(df1["MonsterNummer"]):
            # if count_dict.get(num):
            #     df1["ResAB_amount"].loc[c] = count_dict.get(num)
            for x, y, z in ab_naam:
                if x == num and df1["IsolaatNummer"][c] == y:
                    if count_dict.get(num):
                        df1["ResAB_amount"].loc[c] = count_dict.get(num)
                    if pd.isna(df1["Res_antibiotica"].loc[c]):
                        df1["Res_antibiotica"].loc[c] = z
                    else:
                        df1["Res_antibiotica2"].loc[c] = z
        # print(df1)
        # exit()
        df1 = drop_ni_columns(df1)
        df1["Res_antibiotica2"].fillna(0, inplace = True)
        df1["Res_antibiotica"].fillna(0, inplace = True)
        df1["ResAB_amount"].fillna(0, inplace = True)
        for col in df1.columns:
            df1[col]= df1[col].fillna(df1[col].value_counts().idxmax())

        print(df1)
        le = preprocessing.LabelEncoder()
        labels = le.fit_transform(df1["ResAB_amount"])
        one_hot_table = pd.get_dummies(df1)
        standard_embedding = umap.UMAP(random_state=42).fit_transform(one_hot_table)
        plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=labels, s=30, cmap='Spectral')
        plt.show()

    if kwargs["f"] == "a":
        unsup_one_table(tab_five.loc[:kwargs["amount"]].copy())
    elif kwargs["f"] == "b":
        combined_dataframe(tab_five.loc[:kwargs["amount"]].copy(),tab_six.loc[:kwargs["amount"]].copy())  
    elif kwargs["f"] == "c":
        extravalues(tab_three, tab_one, tab_two, tab_four)
    elif kwargs["f"] == "d":
        two_to_one_df(tab_five.copy(),tab_six.copy())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Unsupervised learning function')
    parser.add_argument("--f", default="b", help="select which function to use")
    parser.add_argument("--amount", default=300_000, help="select over how many rows you want to do the unsupervised learning")
    parser.add_argument("--nn", default = 30,  help="select the amount of nn cells for the umap")
    parser.add_argument("--min_dis", default = 0.2,  help="select the minimal distance for the umap")
    parser.add_argument("--metric", default = "sokalsneath",  help="select which metric for the umap you want to compute")
    args = parser.parse_args()

main(**vars(args))




