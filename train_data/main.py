import pandas as pd
from sklearn.cluster import KMeans

from stack_for_visual import Store


def main(**kwargs):
    lab_res = pd.read_csv('7 columns from mmi_Lab_MMI_Resistentie.txt', sep='\t', encoding="UTF-16")
    bep_tekst = pd.read_csv('8 columns from mmi_Lab_MMI_BepalingenTekst.txt', sep='\t', encoding="UTF-16")
    isolaten = pd.read_csv('9 columns from mmi_Lab_MMI_Isolaten.txt', sep='\t', encoding="UTF-16")
    opname = pd.read_csv('alle columns mmi_Opname_Opname.txt', sep='\t', encoding="UTF-16")

    mns = []
    for c, num in enumerate(lab_res["RISV_Waarde"]):
        if str(num) == "R":
            mns.append((lab_res["AfnameDatum"][c],lab_res["MonsterNummer"][c]))
    mns = list(set(mns))

    orgs = []
    print(mns)
    for mon in mns:
        orgs.append([isolaten["MicroOrganismeName"][c] for (c,num) in enumerate(isolaten["MonsterNummer"]) if (mon[1] == num and isolaten["PhylumDivisionGroup"][c] == "Proteobacteria")])

    print(orgs)
    for date in mns:
        print("On ", date[0], " we have found the following organisms: ", orgs)
    # print(bep_tekst)
    # print(isolaten)
    # print(lab_res)
    # print(opname)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Visualizer for antibiotic resistance in medical cultures')

main(**vars(args))

##################### OLD (maybe later functional) ############################
# for c, num in enumerate(isolaten["MonsterNummer"]):
#     if mon[1] == num and isolaten["PhylumDivisionGroup"][c] == "Proteobacteria":
#         orgs.append(isolaten["MicroOrganismeName"][c])
    # for c, num in enumerate(bep_tekst["MonsterNummer"]):
#     if mon == num and bep_tekst[]