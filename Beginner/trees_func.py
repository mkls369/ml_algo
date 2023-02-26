import numpy as np
from collections import Counter
import statistics as st


def calc_entropy(arr):
    freq = np.bincount(arr)
    prob = freq / len(arr)
    
    entropy = 0
    for p in prob:
        if p > 0:
            entropy += p * np.log2(p)
    return -entropy

def calc_information_gain(parent, child1, child2):
    weight1 = len(child1) / len(parent)
    weight2 = len(child2) / len(parent)
    
    ig = calc_entropy(parent) - (weight1 * calc_entropy(child1) + weight2 * calc_entropy(child2))
    return ig

#Creating a class to collect values from our reuslts
class Node:
    '''
    Helper class which implements a single tree node.
    '''
    def __init__(self, feature=None, threshold=None, data_left=None, data_right=None, gain=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.data_left = data_left
        self.data_right = data_right
        self.gain = gain
        self.value = value
        

def get_best_split(X, y):
    best_split = {}
    best_info_gain = -1
    n_rows, n_cols = X.shape

    df = np.concatenate((X, y.reshape(1, -1).T), axis=1)
    # For every dataset feature
    for f_idx in range(n_cols):
        X_curr = X[:, f_idx]
        # For every unique value of that feature
        for threshold in np.unique(X_curr):
            # Construct a dataset and split it to the left and right parts
            # Left part includes records lower or equal to the threshold
            # Right part includes records higher than the threshold
            
            df_left = np.array([row for row in df if row[f_idx] <= threshold])
            df_right = np.array([row for row in df if row[f_idx] > threshold])

            # Do the calculation only if there's data in both subsets
            if len(df_left) > 0 and len(df_right) > 0:
                # Obtain the value of the target variable for subsets
                y = df[:, -1].astype('int64')
                y_left = df_left[:, -1].astype('int64')
                y_right = df_right[:, -1].astype('int64')

                # Caclulate the information gain and save the split parameters
                # if the current split if better then the previous best
                #import ipdb; ipdb.set_trace()
                gain = calc_information_gain(y, y_left, y_right)
                if gain > best_info_gain:
                    best_split = {
                        'predictor_i': f_idx,
                        'threshold': threshold,
                        'data_left': df_left,
                        'data_right': df_right,
                        'gain': gain
                    }
                    best_info_gain = gain
    return(best_split)

def grow_tree(X, y, depth=0, min_samples_split=2, max_depth=3):
    n_rows, n_cols = X.shape

    # Check to see if a node should be leaf node
    if n_rows >= min_samples_split and depth <= max_depth:
        # Get the best split
        best = get_best_split(X, y)
        #import ipdb;ipdb.set_trace()
        # If the split isn't pure
        if best['gain'] > 0:
            child1 = grow_tree(
                X=best['data_left'][:, :-1], 
                y=best['data_left'][:, -1], 
                depth=depth + 1
            )
            child2 = grow_tree(
                X=best['data_right'][:, :-1], 
                y=best['data_right'][:, -1], 
                depth=depth + 1
            )

            node = Node(
                feature=best['predictor_i'], 
                threshold=best['threshold'], 
                data_left=child1, 
                data_right=child2, 
                gain=best['gain']
            )

            return node
    #Leaf node - predict mode
    return Node(value=st.mode(y)[0][0])

def predict_value(x, tree):
    if tree.value != None:
        return(tree.value)
    feature_value = x[tree.feature]
    #Go down left child
    if feature_value <= tree.threshold:
        return predict_value(x=x, tree=tree.data_left)

    #Go down right child
    if feature_value > tree.threshold:
        return predict_value(x=x, tree=tree.data_right) 