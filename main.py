import pandas as pd
import matplotlib
import numpy as np
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import argparse
from sklearn import preprocessing
import umap

sel1_datatype = [["Monsternummer","StudieNummer","MateriaalShortName","WerkplekCode","BepalingCode","ArtsCode","AfdelingCodeAanvrager","Locatie","Waarde","Uitslag"],
["Monsternummer","IsolaatNummer","MicroOrganismeCode","AfnameDatum","ArtsCode","AfdelingCodeAanvrager","AfdelingNaamAanvrager","AfdelingKliniekPoliAanvrager","OrganisatieCodeAanvrager","OrganisatieNaamAanvrager","StudieNummer","MicroOrganismeOuder","MicroOrganismeOuderOuder","MicroBiologieProcedureCode","MicroOrganismeName","MicroOrganismeType","MicroOrganismeParentCode","MateriaalCode","Kingdom","PhylumDivisionGroup","Class","Order","Family","Genus","MateriaalDescription","MateriaalShortName","ExternCommentaar","TimeStamp"],
["Monsternummer","LabIndicator","AfnameDatum","BepalingsCode","IsolaatNummer","AntibioticaNaam","AB_Code","Methode","MIC_RuweWaarde","E_TestRuweWaarde","AgarDiffRuweWaarde","RISV_Waarde","TimeStamp"],
["VoorschriftId","Pseudo_id","OpnameID","Startmoment","Status_naam","Snelheid","Snelheidseenheid","Dosis","DosisEenheid","Toedieningsroute","MedicatieArtikelCode","MedicatieArtikelNaam","MedicatieArtikelATCcode","MedicatieArtikelATCnaam","FarmaceutischeKlasse","FarmaceutischeSubklasse","TherapeutischeKlasse","Werkplek_code","Werkplek_omschrijving","Bron"],
["Pseudo_id","Geslacht","Geboortedatum","Overlijdensdatum","IsOverleden","Land"]]  

def main(**kwargs):
    tab_one = pd.read_csv('../../offline_files/7 columns from mmi_Lab_MMI_Resistentie.txt', sep='\t', encoding="UTF-16")
    tab_two = pd.read_csv('../../offline_files/8 columns from mmi_Lab_MMI_BepalingenTekst', sep='\t', encoding="UTF-16")
    tab_three = pd.read_csv('../../offline_files/9 columns from mmi_Lab_MMI_Isolaten.txt', sep='\t', encoding="UTF-16")
    tab_four = pd.read_csv('../../offline_files/alle columns mmi_Opname_Opname', sep='\t', encoding="UTF-16")  

    def drop_ni_columns(df):
        df = df.drop(['AfnameDatum'], axis=1)
        df = df.drop(['Pseudo_id'], axis=1)
        return df

    # add the number of resistent bacterias found from every monster
    def add_res_amount(df1, df2):
        # count for each monsternummer there was a resistant bacteria
        mns = []
        for c, num in enumerate(df2["RISV_Waarde"]):
            if str(num) == "R":
                mns.append(df2["MonsterNummer"][c])
        count_dict = {x:mns.count(x) for x in mns}

        # add new colomn of previous count file and fill the NaNs with 0's
        df1["ResAB_amount"] = np.nan
        for c, num in enumerate(df1["MonsterNummer"]):
            if count_dict.get(num):
                df1["ResAB_amount"].loc[c] = count_dict.get(num)
        df1["ResAB_amount"].fillna(0, inplace = True)

        # fill other NaN's with most frequent string in column an drop Not Important columns
        for col in df1.columns:
            df1[col]= df1[col].fillna(df1[col].value_counts().idxmax())
        df1 = drop_ni_columns(df1)

        # put colours relative to amount AB res column and create the clusters using umap and show them
        le = preprocessing.LabelEncoder()
        labels = le.fit_transform(df1["ResAB_amount"])
        one_hot_table = pd.get_dummies(df1)
        standard_embedding = umap.UMAP(random_state=42).fit_transform(one_hot_table)
        plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=labels, s=30, cmap='Spectral')
        plt.show()

    # unsupervised learning over Isolaten table coloured with different microorganisms
    def unsup_one_table(table):
        # fill NaN's with most frequent string from that column
        for col in table.columns:
            table[col]= table[col].fillna(table[col].value_counts().idxmax())

        # do the unsupervised learning where the micro organisms have different colours
        le = preprocessing.LabelEncoder()
        labels = le.fit_transform(table["MicroOrganismeName"])
        one_hot_table = pd.get_dummies(table)
        standard_embedding = umap.UMAP(random_state=42).fit_transform(one_hot_table)
        plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=labels, s=30, cmap='Spectral')
        plt.show()

    def extravalues(df1, df2, df3, df4):
        # count for each monsternummer there was a resistant bacteria
        mns = []
        for c, num in enumerate(df2["RISV_Waarde"]):
            if str(num) == "R":
                mns.append(df2["MonsterNummer"][c])
        count_dict = {x:mns.count(x) for x in mns}


    if kwargs["f"] == "a":
        print("hoi")
        unsup_one_table(tab_three)
    elif kwargs["f"] == "b":
        add_res_amount(tab_three, tab_one)  
    elif kwargs["f"] == "c":
        extravalues(tab_three, tab_one, tab_two, tab_four)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Unsupervised learning function')
    parser.add_argument("--f", default="a", help="select which function to use")
    # parser.add_argument("--one", default="y", help="unsupervised learning over 1 table")
    # parser.add_argument("--resadd", default = "n",  help="unsupervised learning over 1 table plus amount of res bacteria added per monstersample")
    args = parser.parse_args()

main(**vars(args))