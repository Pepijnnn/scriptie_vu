import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pandas as pd
import umap
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

class Optparams():

    def pep_best_parameter(self, df1, df2):
        """ input are two specific datadrames that need to have the next columns:
            df1: ['MonsterNummer','IsolaatNummer','MicroOrganismeOuder', 'MateriaalDescription']
            df2: ['MonsterNummer','IsolaatNummer','AntibioticaNaam', 'RISV_Waarde']
            output is a .txt file where we see the best parameters of the umap for these dataframes"""

        # combine part of the data frames 
        df1 = df1[['MonsterNummer','IsolaatNummer','MicroOrganismeOuder', 'MateriaalDescription', 'AfdelingNaamAanvrager']].copy()
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
        X = pd.get_dummies(df3[["AntibioticaNaam","MicroOrganismeOuder","MateriaalDescription","AfdelingNaamAanvrager"]])
        y = df3["RISV_Waarde"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 40)

        # these are the umap parameters the algorithm is going to calculate
        tuning_parameters = {'n_neighbors' : [2, 4, 6, 10, 15, 20, 40],
                                'min_dist' : [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
                                'metric' : ['sokalsneath', 'yule']
                            }
        # calculate the SVC of all the options and write them to a text file
        params = []
        for i in tqdm(tuning_parameters.get('metric')):
            for j in tqdm(tuning_parameters.get('n_neighbors')):
                for k in tqdm(tuning_parameters.get('min_dist')):
                    trans = umap.UMAP(n_neighbors=j, min_dist=k, n_components=2, metric=i, random_state=41).fit(X_train)
                    svc = SVC(gamma = 'auto').fit(trans.embedding_, y_train)
                    test_embedding = trans.transform(X_test)
                    print(svc.score(test_embedding, y_test))
                    params.append([i, j, k, svc.score(test_embedding, y_test)])
                    print(f'the metric used was: {i}, the n_neighbors: {j}, the min_distance: {k}')
        data_frame = pd.DataFrame(params, columns = ['metric', 'nearest_neighbors', 'min_distance', 'score'])
        data_frame.to_csv(f'hyper_params_2_{kwargs["amount"]}.txt',index=False)

    def grid_search(self):
        df = pd.read_csv('../../offline_files/15 columns from BepalingTekstMetIsolatenResistentie_tot_103062.txt', sep='\t', encoding="UTF-16")  
        df3 = df.iloc[:100_000]
        df3.drop(df3.loc[(df3['RISV_Waarde']=='V') | (df3['RISV_Waarde']=='I')].index, inplace=True)
        df3.sort_values(by=['RISV_Waarde'], ascending=False, inplace=True)
        rest = df3.RISV_Waarde.value_counts()['S'] - df3.RISV_Waarde.value_counts()['R']
        df3 = df3.iloc[rest:,]

        # drop the columns that we don't want to train on
        y = df3['RISV_Waarde']
        df3.drop(['Monsternummer', 'IsolaatNummer', 'RISV_Waarde'], inplace=True, axis=1)

        # fill other NaN's with most frequent string in column an drop Not Important columns
        for col in df3.columns:
            df3[col].fillna("0", inplace=True)

        # put colours relative to amount AB res column and create the clusters using umap and show them
        X = pd.get_dummies(df3)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 40)

        trans = umap.UMAP(metric='yule', n_neighbors=6, n_components=2, min_dist=0.15, random_state=42).fit(X_train)
        svc = SVC(gamma = 'auto').fit(trans.embedding_, y_train)
        test_embedding = trans.transform(X_test)

        # Utility function to move the midpoint of a colormap to be around
        # the values of interest.

        class MidpointNormalize(Normalize):

            def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
                self.midpoint = midpoint
                Normalize.__init__(self, vmin, vmax, clip)

            def __call__(self, value, clip=None):
                x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
                return np.ma.masked_array(np.interp(value, x, y))

        # #############################################################################
        # Load and prepare data set
        #
        # dataset for grid search

        iris = load_iris()
        X = iris.data
        y = iris.target

        # Dataset for decision function visualization: we only keep the first two
        # features in X and sub-sample the dataset to keep only 2 classes and
        # make it a binary classification problem.

        X_2d = X[:, :2]
        X_2d = X_2d[y > 0]
        y_2d = y[y > 0]
        y_2d -= 1

        # It is usually a good idea to scale the data for SVM training.
        # We are cheating a bit in this example in scaling all of the data,
        # instead of fitting the transformation on the training set and
        # just applying it on the test set.

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_2d = scaler.fit_transform(X_2d)

        # #############################################################################
        # Train classifiers
        #
        # For an initial search, a logarithmic grid with basis
        # 10 is often helpful. Using a basis of 2, a finer
        # tuning can be achieved but at a much higher cost.

        C_range = np.logspace(-2, 10, 13)
        gamma_range = np.logspace(-9, 3, 13)
        param_grid = dict(gamma=gamma_range, C=C_range)
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
        grid.fit(X, y)

        print("The best parameters are %s with a score of %0.2f"
            % (grid.best_params_, grid.best_score_))

        # Now we need to fit a classifier for all parameters in the 2d version
        # (we use a smaller set of parameters here because it takes a while to train)

        C_2d_range = [1e-2, 1, 1e2]
        gamma_2d_range = [1e-1, 1, 1e1]
        classifiers = []
        for C in C_2d_range:
            for gamma in gamma_2d_range:
                clf = SVC(C=C, gamma=gamma)
                clf.fit(X_2d, y_2d)
                classifiers.append((C, gamma, clf))

        # #############################################################################
        # Visualization
        #
        # draw visualization of parameter effects

        plt.figure(figsize=(8, 6))
        xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
        for (k, (C, gamma, clf)) in enumerate(classifiers):
            # evaluate decision function in a grid
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            # visualize decision function for these parameters
            plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
            plt.title("gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)),
                    size='medium')

            # visualize parameter's effect on decision function
            plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
            plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdBu_r,
                        edgecolors='k')
            plt.xticks(())
            plt.yticks(())
            plt.axis('tight')

        scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                            len(gamma_range))

        # Draw heatmap of the validation accuracy as a function of gamma and C
        #
        # The score are encoded as colors with the hot colormap which varies from dark
        # red to bright yellow. As the most interesting scores are all located in the
        # 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
        # as to make it easier to visualize the small variations of score values in the
        # interesting range while not brutally collapsing all the low score values to
        # the same color.

        plt.figure(figsize=(8, 6))
        plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
        plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
                norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
        plt.xlabel('gamma')
        plt.ylabel('C')
        plt.colorbar()
        plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
        plt.yticks(np.arange(len(C_range)), C_range)
        plt.title('Validation accuracy')
        plt.show()