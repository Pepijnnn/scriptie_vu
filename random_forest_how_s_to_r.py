import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Globals
max_depth = 5
n_estimators = 100
tree_name = "tree_X"


# source https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
class RF():
    def kianho(self, X, y):
        import matplotlib.pyplot as plt

        from collections import OrderedDict
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier

        # Author: Kian Ho <hui.kian.ho@gmail.com>
        #         Gilles Louppe <g.louppe@gmail.com>
        #         Andreas Mueller <amueller@ais.uni-bonn.de>
        #
        # License: BSD 3 Clause

        print(__doc__)

        RANDOM_STATE = 40

        # Generate a binary classification dataset.
        # X, y = make_classification(n_samples=500, n_features=25,
        #                         n_clusters_per_class=1, n_informative=15,
        #                         random_state=RANDOM_STATE)

        # NOTE: Setting the `warm_start` construction parameter to `True` disables
        # support for parallelized ensembles but is necessary for tracking the OOB
        # error trajectory during training.
        ensemble_clfs = [
            ("RandomForestClassifier, max_features=None", RandomForestClassifier(n_estimators=1000, warm_start=True, max_features=None, oob_score=True, random_state=RANDOM_STATE)),
            #("RandomForestClassifier, max_features='log2'", RandomForestClassifier(n_estimators=1000, warm_start=True, max_features='log2', oob_score=True, random_state=RANDOM_STATE)),
            ("RandomForestClassifier, max_features='sqrt", RandomForestClassifier(n_estimators=1000, warm_start=True, max_features='sqrt', oob_score=True, random_state=RANDOM_STATE)),
            ("RandomForestClassifier, max_features=2", RandomForestClassifier(n_estimators=1000, warm_start=True, max_features=2, oob_score=True, random_state=RANDOM_STATE))
        ]

        # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
        error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

        # Range of `n_estimators` values to explore.
        min_estimators = 10
        max_estimators = 1000

        for label, clf in ensemble_clfs:
            for i in range(min_estimators, max_estimators + 5):
                clf.set_params(n_estimators=i)
                clf.fit(X, y)

                # Record the OOB error for each `n_estimators=i` setting.
                oob_error = 1 - clf.oob_score_
                error_rate[label].append((i, oob_error))

        # Generate the "OOB error rate" vs. "n_estimators" plot.
        for label, clf_err in error_rate.items():
            xs, ys = zip(*clf_err)
            plt.plot(xs, ys, label=label)

        plt.xlim(min_estimators, max_estimators)
        plt.xlabel("n_estimators")
        plt.ylabel("OOB error rate")
        plt.legend(loc="upper right")
        plt.show()

    def visualize_tree(self, rf, feature_list, y):
        # print(y.apply(str))
        # exit()
        # Import tools needed for visualization
        from sklearn.tree import export_graphviz
        import pydot

        # Pull out one tree from the forest
        tree = rf.estimators_[25]

        # Export the image to a dot file
        export_graphviz(tree, out_file = f'trees/{tree_name}.dot', feature_names = feature_list, rounded = True, precision = 2, class_names=True)
        # Export as dot file
        # export_graphviz(tree, f'trees/{tree_name}.dot', 
        #         feature_names = feature_list,
        #         class_names = y,
        #         rounded = True, 
        #         precision = 2, filled = True)

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
        # print(SS_df['Cefalosporines, 3e generatie'].astype(bool).sum(axis=0))
        
        X, y = self.dfs_to_Xy(SS_df, SR_df)
        feature_list = list(X.columns)
        print(len(SS_df.columns))
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
        # self.kianho(X_train, y_train)
        # exit()
        # Instantiate model with 100 decision trees
        # rf = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, min_samples_split=50, min_samples_leaf=20,  random_state = 42, oob_score = True, bootstrap = True)
        rf = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, min_samples_split=40, min_samples_leaf=20,  random_state = 42, oob_score = True, bootstrap = True)
        rf.fit(X_train, y_train)
        print("Xtrain, Xtest are ", len(X_train), len(X_test))
        

        from sklearn.model_selection import cross_val_score  
        for score in ["precision", "recall"]:
            print(score)
            all_accuracies = cross_val_score(estimator=rf, X=X_train, scoring=score, y=y_train, cv=5)  
            print(f"cross_val_score = {all_accuracies}")
            print(f'averaged = {np.sum(all_accuracies)/len(all_accuracies)}')
        
        # Train the model on training data
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        print(f"F1_score = {f1_score(y_test, y_pred, average='binary')}")
        from sklearn.metrics import classification_report
        print(classification_report(y_test, y_pred, target_names=['class 0', 'class 1']))
        
        self.visualize_tree(rf, feature_list, y_train)
        
        from sklearn.base import clone 
        import eli5
        from sklearn.metrics import average_precision_score
        def drop_col_feat_imp(model, X_train, y_train, random_state = 42):
            
            # clone the model to have the exact same specification as the one initially trained
            model_clone = clone(model)
            # set random_state for comparability
            model_clone.random_state = random_state
            # training and scoring the benchmark model
            model_clone.fit(X_train, y_train)
            benchmark_score = f1_score(y_test, model_clone.predict(X_test)) # model_clone.score(X_train, y_train)
            # list for storing feature importances
            importances = []
            
            # iterating over all columns and storing feature importance (difference between benchmark and new model)
            for col in X_train.columns:
                model_clone = clone(model)
                model_clone.random_state = random_state
                model_clone.fit(X_train.drop(col, axis = 1), y_train)
                drop_col_score = f1_score(y_test, model_clone.predict(X_test.drop(col,axis=1)))# model_clone.score(X_train.drop(col, axis = 1), y_train)
                importances.append(benchmark_score - drop_col_score)
            
            print(X_train.columns)
            print(importances)
            drop_indices = np.argsort(importances)

            return (X_train.columns, importances, drop_indices)
        drop_columns, drop_importances, drop_indices = drop_col_feat_imp(rf, X_train, y_train)

        from sklearn.metrics import average_precision_score
        from sklearn.metrics import mean_squared_error
        from treeinterpreter import treeinterpreter as ti

        y_pred = rf.predict(X_test)
        features = X.columns.values
        importances = rf.feature_importances_
        indices = np.argsort(importances)
        # print(features)
        # print(importances[indices])
        # print(features[indices])
        # for i,j in zip(importances[indices], features[indices]):
        #     print(i,j)
        
        # features[8] = "Pen, breedspectrum B-lact inhib"
        #features.replace("Penicillinen, breedspectrum met beta-lactamase inhibitor", "Penicillinen, brspec B-lact inhib")
        # print(features)
        # exit()
        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib import pyplot as plt
        import seaborn as sns

        sns.set(context="paper")
        # Z = [x for _,x in sorted(zip(drop_importances, drop_columns))]
        # print(Z)
        # print([(x,y) for y,x in sorted(zip(drop_importances, drop_columns))])
        # ax = sns.barplot(sorted(drop_importances), Z)
        # ax.set_yticklabels(Z, fontsize=6)

        # plt.barh(range(len(indices)), importances[indices],color='#8f63f4', align='center') #, color='#8f63f4'
        ax = sns.barplot(x=importances[indices], y=features[indices])
        ax.set_title('Feature importances', fontsize=30)
        ax.invert_yaxis()
        plt.yticks(range(len(indices)), features[indices])
        plt.xlabel('Drop Importance')
        plt.show()
        # exit()
        #prediction, bias, c/ontributions = ti.predict(rf, X_test)
        # for i in range(len(instances)):
        #     print("Instance", i)
        #     print("Bias (trainset mean)", biases[i])
        #     print "Feature contributions:")
        #     for c, feature in sorted(zip(contributions[i], 
        #                                 boston.feature_names), 
        #                             key=lambda x: -abs(x[0])):
        #         print feature, round(c, 2)
        #     print "-"*20 

        # print("Predictions")
        # print(np.allclose(prediction, bias + np.sum(contributions, axis=1)))
        # print(np.allclose(y_pred, bias + np.sum(contributions, axis=1)))
        # print(bias + np.sum(contributions, axis=1))
        # print(prediction - (bias + np.sum(contributions, axis=1)))
        # print(y_pred - (bias + np.sum(contributions, axis=1)))

        # print(f"Avg prec score = {average_precision_score(y_test, y_pred)}")
        #print(f"Avg prec score = {mean_squared_error(y_test, y_pred) }")
        self.visualize_tree(rf, feature_list, y_train)
    
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
        return pd.concat([leeftijd_df, one_hot_geslacht], axis=1)

    # fluorquinolonen (moxifloxacine, levofloxacine, ciprofloxacine) met Entero I en Entero II en Pseudomonas
    def make_pep_data_3_tree(self, df, even):
        # make percentages of departments and add a total column
        dep_df = df[[col for col in df.columns if "VUMC" in str(col)]]
        ab_df = df[["Moxifloxacine", "Levofloxacine", "Ciprofloxacine", "Cotrimoxazol", "Meropenem", "Imipenem/cilastatine", "Vancomycine",\
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

        self.decisions_to_R(SS_df, SR_df)

    # Cotrimoxazol R met Entero I en Entero II
    def make_pep_data_2_tree(self, df, even):
        print('orig df =', len(df))
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
        print("division sr/total = ", len(SR_cotrim_df)/(len(SS_cotrim_df)+len(SR_cotrim_df)))
        
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
        print("end df ss = ", len(SS_df))
        print("end df sr = ", len(SR_df))

        self.decisions_to_R(SS_df, SR_df)

    # Cefotaxim/Ceftazidim R met Entero I en Pseudomonas 
    def make_pep_data_1_tree(self, df, even):
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
        SS_cef_df = R_cef_df.loc[(R_cef_df==0).all(axis=1)].sample(n=int(SR_cef_df.shape[0]), random_state=40) if even else R_cef_df.loc[(R_cef_df==0).all(axis=1)]
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

        self.decisions_to_R(SS_df, SR_df)

    def make_test_tree(self, df):
        import seaborn as sns
        import matplotlib.pyplot as plt
        height = [241, 1379, 426, 2254, 897, 536, 101, 757]
        bars = ('0', '1', '2', '3', '4', '5', '6', '7')
        y_pos = np.arange(len(bars))
        sns.set(context="paper")
        ax = sns.barplot(x=y_pos, y=height)
        plt.xticks(y_pos, bars)
        plt.ylabel('Patients')
        plt.xlabel('Cluster')
        plt.show()
        exit()

        # print(df["Cotrimoxazol"].astype(bool).sum(axis=0))
        # exit()
        global tree_name, max_depth, n_estimators
        tree_name = "tree28_even_true"
        max_depth = 5
        n_estimators = 600
        #self.make_imi_tree(df)
        # self.make_cipro_tree(df, even=True)
        # self.make_pep_data_1_tree(df, even=True)
        # print(df.columns)
        # exit()
        self.make_pep_data_2_tree(df, even=True)
        # self.make_pep_data_3_tree(df, even=True)

    # def make_cipro_tree(self, df, even):
    #     # make percentages of departments and add a total column
    #     dep_df = df[[col for col in df.columns if "VUMC" in str(col)]]
        # ab_df = df[["Cotrimoxazol", "Meropenem", "Imipenem/cilastatine", "Vancomycine", "Fluoroquinolonen",\
        #     "Penicillinen, smalspectrum", "Penicillinen, breedspectrum met beta-lactamase inhibitor", "Penicillinen, breedspectrum",\
        #     "Cefalosporines, 3e generatie", "Cefalosporines, 1e generatie"]]
    #     pers_df = self.make_clean_pers_df(df[["Geslacht", "Land", "Leeftijd"]])

    #     # AB hierbij is fluorquinolonen
    #     cipro_df = df[[col for col in df.columns if "ciprofloxacine" in str(col)]]
    #     R_cipro_df = cipro_df[[col for col in cipro_df.columns if str(col)[-1] == "R"]]
        
    #     # Verdeel de database in mensen die in beide kolommen 0 hebben en tenminste één 1 in een van beide
    #     # gebruikt voor de splitsing van de andere databases
    #     SR_cipro_df = R_cipro_df.loc[~(R_cipro_df==0).all(axis=1)]
    #     SS_cipro_df = R_cipro_df.loc[(R_cipro_df==0).all(axis=1)].sample(n=int(SR_cipro_df.shape[0]), random_state=42) if even else R_cipro_df.loc[(R_cipro_df==0).all(axis=1)]
    #     print(len(SR_cipro_df)/(len(SS_cipro_df)+len(SR_cipro_df)))
    #     print(len(SR_cipro_df))
        
    #     # subdataframe van de departments van de vorige indexen van de data tables
    #     SR_deps = dep_df.loc[list(SR_cipro_df.index.values)]
    #     SS_deps = dep_df.loc[list(SS_cipro_df.index.values)]

    #     # subdataframes van de ab
    #     SR_ab = ab_df.loc[list(SR_cipro_df.index.values)]
    #     SS_ab = ab_df.loc[list(SS_cipro_df.index.values)]

    #     # subdataframes van de personen
    #     SR_pers = pers_df.loc[list(SR_cipro_df.index.values)]
    #     SS_pers = pers_df.loc[list(SS_cipro_df.index.values)]

    #     # de datatabellen weer bi elkaar
    #     SS_df = pd.concat([SS_ab, SS_pers, SS_deps], axis=1)
    #     SR_df = pd.concat([SR_ab, SR_pers, SR_deps], axis=1)

    #     self.decisions_to_R(SS_df, SR_df)
        
        
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