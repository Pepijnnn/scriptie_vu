import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Globals
max_depth = 5
n_estimators = 100
tree_name = "tree_X"


# source https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
class RF():

    def visualize_tree(self, rf, feature_list):
        # Import tools needed for visualization
        from sklearn.tree import export_graphviz
        import pydot

        # Pull out one tree from the forest
        tree = rf.estimators_[5]

        # Export the image to a dot file
        export_graphviz(tree, out_file = f'trees/{tree_name}.dot', feature_names = feature_list, rounded = True, precision = 1)

        # Use dot file to create a graph
        (graph, ) = pydot.graph_from_dot_file(f'trees/{tree_name}.dot')

        # Write graph to a png file
        graph.write_png(f'trees/{tree_name}.png')

    def dfs_to_Xy(self, SS_df, SR_df):
        SS_df['group_label'] = np.zeros(SS_df.shape[0], dtype=int)
        SR_df['group_label'] = np.ones(SR_df.shape[0], dtype=int)

        X = pd.concat([SS_df, SR_df])
        y = X.pop("group_label")
        return X, y

    def decisions_to_R(self, SS_df, SR_df):
        X, y = self.dfs_to_Xy(SS_df, SR_df)
        feature_list = list(X.columns)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

        # Instantiate model with 100 decision trees
        rf = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth,  random_state = 42, min_samples_split=50, min_samples_leaf=20)

        
        # Train the model on training data
        rf.fit(X_train, y_train)

        from sklearn.metrics import average_precision_score
        from sklearn.metrics import mean_squared_error
        from treeinterpreter import treeinterpreter as ti

        y_pred = rf.predict(X_test)
        #prediction, bias, contributions = ti.predict(rf, X_test)
        #print(prediction, bias, contributions)
        #print(contributions)
    
        print(f"Avg prec score = {average_precision_score(y_test, y_pred)}")
        #print(f"Avg prec score = {mean_squared_error(y_test, y_pred) }")
        self.visualize_tree(rf, feature_list)
    
    def make_imi_tree(self, df):
        # make percentages of departments and add a total column
        dep_df = df[[col for col in df.columns if "VUMC" in str(col)]]
        ab_df = df[["Cotrimoxazol", "Meropenem", "Imipenem/cilastatine", "Vancomycine", "Fluoroquinolonen"]]

        imi_df = df[[col for col in df.columns if "imipenem" in str(col)]]
        R_imi_df = imi_df[[col for col in imi_df.columns if str(col)[-1] == "R"]]
        
        # I heeft 6552/6591 0, II heeft 6556/6591 0
        # Verdeel de database in mensen die in beide kolommen 0 hebben en tenminste één 1 in een van beide
        SR_imi_df = R_imi_df.loc[~(R_imi_df==0).all(axis=1)]
        SS_imi_df = R_imi_df.loc[(R_imi_df==0).all(axis=1)] # .iloc[:int(SR_imi_df.shape[0])]

        # subdataframe van de departments van de vorige indexen van de data tables
        SR_deps = dep_df.loc[list(SR_imi_df.index.values)]
        SS_deps = dep_df.loc[list(SS_imi_df.index.values)]

        # subdataframes van de ab
        SR_ab = ab_df.loc[list(SR_imi_df.index.values)]
        SS_ab = ab_df.loc[list(SS_imi_df.index.values)]

        # de datatabellen weer bi elkaar
        SS_df = pd.concat([SS_ab, SS_deps], axis=1) # SS_imi_df
        SR_df = pd.concat([SR_ab, SR_deps], axis=1) # SR_imi_df

        self.decisions_to_R(SS_df, SR_df)
    
    def make_clean_pers_df(self, df):
        leeftijd_df = df["Leeftijd"]
        one_hot_geslacht = pd.get_dummies(df["Geslacht"])
        one_hot_land = pd.get_dummies(df["Land"])
        return pd.concat([leeftijd_df, one_hot_geslacht, one_hot_land], axis=1)

    def make_cipro_tree(self, df):
        # make percentages of departments and add a total column
        dep_df = df[[col for col in df.columns if "VUMC" in str(col)]]
        ab_df = df[["Cotrimoxazol", "Meropenem", "Imipenem/cilastatine", "Vancomycine", "Fluoroquinolonen",\
            "Penicillinen, smalspectrum", "Penicillinen, breedspectrum met beta-lactamase inhibitor", "Penicillinen, breedspectrum",\
            "Cefalosporines, 3e generatie", "Cefalosporines, 1e generatie"]]
        pers_df = self.make_clean_pers_df(df[["Geslacht", "Land", "Leeftijd"]])

        # AB hierbij is fluorquinolonen
        cipro_df = df[[col for col in df.columns if "ciprofloxacine" in str(col)]]
        R_cipro_df = cipro_df[[col for col in cipro_df.columns if str(col)[-1] == "R"]]
        
        # Verdeel de database in mensen die in beide kolommen 0 hebben en tenminste één 1 in een van beide
        # gebruikt voor de splitsing van de andere databases
        SR_cipro_df = R_cipro_df.loc[~(R_cipro_df==0).all(axis=1)]
        SS_cipro_df = R_cipro_df.loc[(R_cipro_df==0).all(axis=1)] #.iloc[:int(SR_cipro_df.shape[0])]
        print(len(SR_cipro_df)/(len(SS_cipro_df)+len(SR_cipro_df)))
        print(len(SS_cipro_df))
        
        # subdataframe van de departments van de vorige indexen van de data tables
        SR_deps = dep_df.loc[list(SR_cipro_df.index.values)]
        SS_deps = dep_df.loc[list(SS_cipro_df.index.values)]

        # subdataframes van de ab
        SR_ab = ab_df.loc[list(SR_cipro_df.index.values)]
        SS_ab = ab_df.loc[list(SS_cipro_df.index.values)]

        # subdataframes van de personen
        SR_pers = pers_df.loc[list(SR_cipro_df.index.values)]
        SS_pers = pers_df.loc[list(SS_cipro_df.index.values)]

        # de datatabellen weer bi elkaar
        SS_df = pd.concat([SS_ab, SS_pers], axis=1)
        SR_df = pd.concat([SR_ab, SR_pers], axis=1)

        self.decisions_to_R(SS_df, SR_df)

    def make_test_tree(self, df):
        global tree_name, max_depth, n_estimators
        tree_name = "tree13"
        max_depth = 4
        n_estimators = 100
        #self.make_imi_tree(df)
        self.make_cipro_tree(df)
        
        
    # TODO put all the right names here still
    # def make_vanco_tree(self, df):
    #     # make percentages of departments and add a total column
    #     dep_df = df[[col for col in df.columns if "VUMC" in str(col)]]
    #     ab_df = df[["Cotrimoxazol", "Meropenem", "Imipenem/cilastatine", "Vancomycine", "Fluoroquinolonen",\
    #         "Penicillinen, smalspectrum", "Penicillinen, breedspectrum met beta-lactamase inhibitor", "Penicillinen, breedspectrum",\
    #         "Cefalosporines, 3e generatie", "Cefalosporines, 1e generatie"]]

    #     # AB hierbij is vanco
    #     imi_df = df[[col for col in df.columns if "vancomycine" in str(col)]]
    #     R_imi_df = imi_df[[col for col in imi_df.columns if str(col)[-1] == "R"]]
        
    #     # Verdeel de database in mensen die in beide kolommen 0 hebben en tenminste één 1 in een van beide
    #     SR_imi_df = R_imi_df.loc[~(R_imi_df==0).all(axis=1)]
    #     SS_imi_df = R_imi_df.loc[(R_imi_df==0).all(axis=1)] #.iloc[:int(SR_imi_df.shape[0])]
    #     print(len(SR_imi_df)/(len(SS_imi_df)+len(SR_imi_df)))
    #     print(len(SS_imi_df))
        

    #     # subdataframe van de departments van de vorige indexen van de data tables
    #     # SR_deps = dep_df.loc[list(SR_imi_df.index.values)]
    #     # SS_deps = dep_df.loc[list(SS_imi_df.index.values)]

    #     # subdataframes van de ab
    #     SR_ab = ab_df.loc[list(SR_imi_df.index.values)]
    #     SS_ab = ab_df.loc[list(SS_imi_df.index.values)]

    #     # de datatabellen weer bi elkaar
    #     SS_df = pd.concat([SS_ab], axis=1) # SS_imi_df
    #     SR_df = pd.concat([SR_ab], axis=1) # SR_imi_df

    #     self.decisions_to_R(SS_df, SR_df)