import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# source https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
class RF():

    def visualize_tree(self, rf, feature_list):
        # Import tools needed for visualization
        from sklearn.tree import export_graphviz
        import pydot

        # Pull out one tree from the forest
        tree = rf.estimators_[5]

        # Import tools needed for visualization
        from sklearn.tree import export_graphviz
        import pydot

        # Pull out one tree from the forest
        tree = rf.estimators_[5]

        # Export the image to a dot file
        export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)

        # Use dot file to create a graph
        (graph, ) = pydot.graph_from_dot_file('tree.dot')

        # Write graph to a png file
        graph.write_png('tree_1.png')

    def dfs_to_Xy(self, SS_df, SR_df):
        SS_df['group_label'] = np.zeros(SS_df.shape[0], dtype=int)
        SR_df['group_label'] = np.ones(SR_df.shape[0], dtype=int)

        X = pd.concat(SS_df, SR_df)
        y = X["group_label"]
        X.drop(["group_label"], axis=1, inplace=True)
        return X, y

    def decisions_to_R(self, SS_df, SR_df):
        X, y = self.dfs_to_Xy(SS_df, SR_df)
        feature_list = list(X.columns)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

        # Instantiate model with 1000 decision trees
        rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

        # Train the model on training data
        rf.fit(X_train, y_train)

        self.visualize_tree(rf, feature_list)
    
    def make_test_tree(self, df):
        

        self.decisions_to_R(SS_df, SR_df)
        