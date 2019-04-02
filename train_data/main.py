import nltk
import pandas as pd

lab_res = pd.read_csv('7 columns from mmi_Lab_MMI_Resistentie.txt', sep='\t', encoding="UTF-16")
bep_tekst = pd.read_csv('8 columns from mmi_Lab_MMI_BepalingenTekst.txt', sep='\t', encoding="UTF-16")
isolaten = pd.read_csv('9 columns from mmi_Lab_MMI_Isolaten.txt', sep='\t', encoding="UTF-16")
opname = pd.read_csv('alle columns mmi_Opname_Opname.txt', sep='\t', encoding="UTF-16")

print(bep_tekst)
print(isolaten)
print(lab_res)
print(opname)