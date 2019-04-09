import pandas as pd
import matplotlib
import numpy as np
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import argparse
import umap

sel1_datatype = [["Monsternummer","StudieNummer","MateriaalShortName","WerkplekCode","BepalingCode","ArtsCode","AfdelingCodeAanvrager","Locatie","Waarde","Uitslag"],
["Monsternummer","IsolaatNummer","MicroOrganismeCode","AfnameDatum","ArtsCode","AfdelingCodeAanvrager","AfdelingNaamAanvrager","AfdelingKliniekPoliAanvrager","OrganisatieCodeAanvrager","OrganisatieNaamAanvrager","StudieNummer","MicroOrganismeOuder","MicroOrganismeOuderOuder","MicroBiologieProcedureCode","MicroOrganismeName","MicroOrganismeType","MicroOrganismeParentCode","MateriaalCode","Kingdom","PhylumDivisionGroup","Class","Order","Family","Genus","MateriaalDescription","MateriaalShortName","ExternCommentaar","TimeStamp"],
["Monsternummer","LabIndicator","AfnameDatum","BepalingsCode","IsolaatNummer","AntibioticaNaam","AB_Code","Methode","MIC_RuweWaarde","E_TestRuweWaarde","AgarDiffRuweWaarde","RISV_Waarde","TimeStamp"],
["VoorschriftId","Pseudo_id","OpnameID","Startmoment","Status_naam","Snelheid","Snelheidseenheid","Dosis","DosisEenheid","Toedieningsroute","MedicatieArtikelCode","MedicatieArtikelNaam","MedicatieArtikelATCcode","MedicatieArtikelATCnaam","FarmaceutischeKlasse","FarmaceutischeSubklasse","TherapeutischeKlasse","Werkplek_code","Werkplek_omschrijving","Bron"],
["Pseudo_id","Geslacht","Geboortedatum","Overlijdensdatum","IsOverleden","Land"]]  

def main(**kwargs):
    tab_one = pd.read_csv('7 columns from mmi_Lab_MMI_Resistentie.txt', sep='\t', encoding="UTF-16")
    # tab_two = pd.read_csv('7 columns from mmi_Lab_MMI_Resistentie.txt', sep='\t', encoding="UTF-16")
    tab_three = pd.read_csv('9 columns from mmi_Lab_MMI_Isolaten.txt', sep='\t', encoding="UTF-16")
    # tab_four = pd.read_csv('7 columns from mmi_Lab_MMI_Resistentie.txt', sep='\t', encoding="UTF-16")
    # tab_five = pd.read_csv('7 columns from mmi_Lab_MMI_Resistentie.txt', sep='\t', encoding="UTF-16")    

    # add the number of resistent bacterias found from every monster
    def add_res_amount(df1, df2):
        mns = []
        for c, num in enumerate(df2["RISV_Waarde"]):
            if str(num) == "R":
                mns.append(df2["MonsterNummer"][c])
        count_dict = {x:mns.count(x) for x in mns}
        # print(count_dict.get(165238001801))
        # exit()
        df1["ResAB_amount"] = np.nan
        for c, num in enumerate(df1["MonsterNummer"]):
            if count_dict.get(num):
                df1["ResAB_amount"].loc[c] = count_dict.get(num)
        df1["ResAB_amount"].fillna(0, inplace = True)
        print(df1)

    # def unsup_one_table(one_table):
    #     table = pd.DataFrame(one_table)
    #     for col in table.columns:
    #         table[str(col)].fillna(method ='ffill', inplace = True) 

    #     one_hot_table = pd.get_dummies(table)
    #     standard_embedding = umap.UMAP(random_state=42).fit_transform(one_hot_table)
    #     plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], s=30, cmap='Spectral')
    #     plt.show()

    #unsup_one_table(tab_three)  
    add_res_amount(tab_three, tab_one)  

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Unsupervised learning function')
    parser.add_argument("--f", default="bac", help="select which function to use")
    args = parser.parse_args()

main(**vars(args))