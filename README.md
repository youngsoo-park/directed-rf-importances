# direct-rf-importances

## Overview
In general, random forest classifiers such as `sklearn.ensemble.RandomForestClassifier` will only yield undirected feature importances, making it difficult to determine the directionality of a feature. This script uses Monte Carlo sampling to estimate Prob(y=1|x) for a feature x across its entire range, while marginalizing over all other features. The output then can be used in various ways to represent the directionality of each feature in the model.

## Input
The code currently takes two pickle files as inputs, one containing the trained `sklearn.ensemble.RandomForestClassifier` object and another containing the data matrix. It can be run as 

`python directed_importances.py pickle_file_with_model.pkl picke_file_with_data_matrix.pkl`

Note that the code expects an entry with the name ***modelobj*** in the first pickle file containing the trained model object, an entry with the name ***feature_importances_names*** in the first pickle file containing the names of all features, and an entry with the name ***test_X*** in the second pickle file containing the data matrix. This will be generalized in later versions.

## Output
The code will return a numpy array of the feature names and their corresponding directed importance values. Currently, the directed importance values are calculated as the difference of Prob(y=1|x) between the maximum and minimum value of feature x within its range. Further developments will follow for alternative ways of quantifying the directed importances.