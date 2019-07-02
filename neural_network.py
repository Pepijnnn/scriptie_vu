import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

class NeuralNetwork():

    def dfs_to_Xy(self, SS_df, SR_df):
        SS_df['group_label'] = np.zeros(SS_df.shape[0], dtype=int)
        SR_df['group_label'] = np.ones(SR_df.shape[0], dtype=int)

        X = pd.concat([SS_df, SR_df])
        y = X.pop("group_label")
        return X, y

    def mlp_nn(self, SS_df, SR_df):
        X, y = self.dfs_to_Xy(SS_df, SR_df)
        feature_list = list(X.columns)
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

        # mlp = MLPRegressor(hidden_layer_sizes=(10,5,10),
        #                              activation='relu',
        #                              solver='adam',
        #                              learning_rate='adaptive',
        #                              max_iter=1000,
        #                              learning_rate_init=0.01,
        #                              alpha=0.01)
        mlp = MLPClassifier(hidden_layer_sizes=(10,2),
                                     activation='relu',
                                     solver='sgd',
                                     nesterovs_momentum=True,
                                     learning_rate='adaptive',
                                     max_iter=1000,
                                     learning_rate_init=0.01,
                                     alpha=0.01)

        mlp.fit(X_train, y_train)

        # y_pred = nn.predict(X_test)
        print(f"Training score is: {mlp.score(X_train, y_train)}")
        
        print(f"Test score is: {mlp.score(X_test, y_test)}")
        exit()



    def make_clean_pers_df(self, df):
        leeftijd_df = df["Leeftijd"]
        one_hot_geslacht = pd.get_dummies(df["Geslacht"])
        #one_hot_land = pd.get_dummies(df["Land"])
        return pd.concat([leeftijd_df, one_hot_geslacht], axis=1)

    # fluorquinolonen (moxifloxacine, levofloxacine, ciprofloxacine) met Entero I en Entero II en Pseudomonas
    def make_pep_data_3_nn(self, df, even):
        # make percentages of departments and add a total column
        dep_df = df[[col for col in df.columns if "VUMC" in str(col)]]
        ab_df = df[["Moxifloxacine", "Levofloxacine", "Ciprofloxacine", "Cotrimoxazol", "Meropenem", "Imipenem/cilastatine", "Vancomycine", "Fluoroquinolonen",\
            "Penicillinen, smalspectrum", "Penicillinen, breedspectrum met beta-lactamase inhibitor", "Penicillinen, breedspectrum",\
            "Cefalosporines, 3e generatie", "Cefalosporines, 1e generatie"]]
        pers_df = self.make_clean_pers_df(df[["Geslacht", "Land", "Leeftijd"]])

        # AB hierbij is cefotaxim/ceftazidim
        fq_df = df[[col for col in df.columns if ("moxifloxacine" in str(col) or "levofloxacine" in str(col) or "ciprofloxacine" in str(col))]]
        R_fq_df = fq_df[[col for col in fq_df.columns if str(col)[-1] == "R"]]
        
        # Verdeel de database in mensen die in beide kolommen 0 hebben en tenminste één 1 in een van beide
        # gebruikt voor de splitsing van de andere databases
        SR_fq_df = R_fq_df.loc[~(R_fq_df==0).all(axis=1)]
        SS_fq_df = R_fq_df.loc[(R_fq_df==0).all(axis=1)].sample(n=int(SR_fq_df.shape[0]), random_state=42) if even else R_fq_df.loc[(R_fq_df==0).all(axis=1)]
        print(len(SR_fq_df)/(len(SS_fq_df)+len(SR_fq_df)))
        print(len(SR_fq_df))
        
        # subdataframe van de departments van de vorige indexen van de data tables
        SR_deps = dep_df.loc[list(SR_fq_df.index.values)]
        SS_deps = dep_df.loc[list(SS_fq_df.index.values)]

        # subdataframes van de ab
        SR_ab = ab_df.loc[list(SR_fq_df.index.values)]
        SS_ab = ab_df.loc[list(SS_fq_df.index.values)]

        # subdataframes van de personen
        SR_pers = pers_df.loc[list(SR_fq_df.index.values)]
        SS_pers = pers_df.loc[list(SS_fq_df.index.values)]

        # de datatabellen weer bi elkaar
        SS_df = pd.concat([SS_ab, SS_pers, SS_deps], axis=1)
        SR_df = pd.concat([SR_ab, SR_pers, SR_deps], axis=1)

        self.mlp_nn(SS_df, SR_df)

    # Cotrimoxazol R met Entero I en Entero II
    def make_pep_data_2_nn(self, df, even):
        # make percentages of departments and add a total column
        dep_df = df[[col for col in df.columns if "VUMC" in str(col)]]
        ab_df = df[["Cotrimoxazol", "Meropenem", "Imipenem/cilastatine", "Vancomycine", "Fluoroquinolonen",\
            "Penicillinen, smalspectrum", "Penicillinen, breedspectrum met beta-lactamase inhibitor", "Penicillinen, breedspectrum",\
            "Cefalosporines, 3e generatie", "Cefalosporines, 1e generatie"]]
        pers_df = self.make_clean_pers_df(df[["Geslacht", "Land", "Leeftijd"]])

        # AB hierbij is cefotaxim/ceftazidim
        cotrim_df = df[[col for col in df.columns if "cotrimoxazol" in str(col)]]
        R_cotrim_df = cotrim_df[[col for col in cotrim_df.columns if str(col)[-1] == "R"]]
        
        # Verdeel de database in mensen die in beide kolommen 0 hebben en tenminste één 1 in een van beide
        # gebruikt voor de splitsing van de andere databases
        SR_cotrim_df = R_cotrim_df.loc[~(R_cotrim_df==0).all(axis=1)]
        SS_cotrim_df = R_cotrim_df.loc[(R_cotrim_df==0).all(axis=1)].sample(n=int(SR_cotrim_df.shape[0]), random_state=42) if even else R_cotrim_df.loc[(R_cotrim_df==0).all(axis=1)]
        print(len(SR_cotrim_df)/(len(SS_cotrim_df)+len(SR_cotrim_df)))
        print(len(SR_cotrim_df))
        
        # subdataframe van de departments van de vorige indexen van de data tables
        SR_deps = dep_df.loc[list(SR_cotrim_df.index.values)]
        SS_deps = dep_df.loc[list(SS_cotrim_df.index.values)]

        # subdataframes van de ab
        SR_ab = ab_df.loc[list(SR_cotrim_df.index.values)]
        SS_ab = ab_df.loc[list(SS_cotrim_df.index.values)]

        # subdataframes van de personen
        SR_pers = pers_df.loc[list(SR_cotrim_df.index.values)]
        SS_pers = pers_df.loc[list(SS_cotrim_df.index.values)]

        # de datatabellen weer bi elkaar
        SS_df = pd.concat([SS_ab, SS_pers, SS_deps], axis=1)
        SR_df = pd.concat([SR_ab, SR_pers, SR_deps], axis=1)

        self.mlp_nn(SS_df, SR_df)

    # Cefotaxim/Ceftazidim R met Entero I en Pseudomonas
    def make_pep_data_1_nn(self, df, even):
        # make percentages of departments and add a total column
        dep_df = df[[col for col in df.columns if "VUMC" in str(col)]]
        ab_df = df[["Cefotaxim", "Ceftazidim", "Cotrimoxazol", "Meropenem", "Imipenem/cilastatine", "Vancomycine", "Fluoroquinolonen",\
            "Penicillinen, smalspectrum", "Penicillinen, breedspectrum met beta-lactamase inhibitor", "Penicillinen, breedspectrum",\
            "Cefalosporines, 3e generatie", "Cefalosporines, 1e generatie"]]
        pers_df = self.make_clean_pers_df(df[["Geslacht", "Land", "Leeftijd"]])

        # AB hierbij is cefotaxim/ceftazidim
        cef_df = df[[col for col in df.columns if ("cefotaxim" in str(col) or "ceftazidim" in str(col))]]
        R_cef_df = cef_df[[col for col in cef_df.columns if str(col)[-1] == "R"]]
        
        # Verdeel de database in mensen die in beide kolommen 0 hebben en tenminste één 1 in een van beide
        # gebruikt voor de splitsing van de andere databases
        SR_cef_df = R_cef_df.loc[~(R_cef_df==0).all(axis=1)]
        SS_cef_df = R_cef_df.loc[(R_cef_df==0).all(axis=1)].sample(n=int(SR_cef_df.shape[0]), random_state=42) if even else R_cef_df.loc[(R_cef_df==0).all(axis=1)]
        print(len(SR_cef_df)/(len(SS_cef_df)+len(SR_cef_df)))
        print(len(SR_cef_df))
        
        # subdataframe van de departments van de vorige indexen van de data tables
        SR_deps = dep_df.loc[list(SR_cef_df.index.values)]
        SS_deps = dep_df.loc[list(SS_cef_df.index.values)]

        # subdataframes van de ab
        SR_ab = ab_df.loc[list(SR_cef_df.index.values)]
        SS_ab = ab_df.loc[list(SS_cef_df.index.values)]

        # subdataframes van de personen
        SR_pers = pers_df.loc[list(SR_cef_df.index.values)]
        SS_pers = pers_df.loc[list(SS_cef_df.index.values)]

        # de datatabellen weer bi elkaar
        SS_df = pd.concat([SS_ab, SS_pers, SS_deps], axis=1)
        SR_df = pd.concat([SR_ab, SR_pers, SR_deps], axis=1)

        self.mlp_nn(SS_df, SR_df)

    def make_nn(self, df):
        self.make_pep_data_1_nn(df, even=True)
        # self.make_pep_data_2_nn(df, even=True)
        # self.make_pep_data_3_nn(df, even=True)