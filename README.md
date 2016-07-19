# direct-rf-importances

## Overview
In general, random forest classifiers such as `sklearn.ensemble.RandomForestClassifier` will only yield undirected feature importances, making it difficult to determine the directionality of a feature. This script uses Monte Carlo sampling to estimate Prob(y=1|x) for a feature x across its entire range, while marginalizing over all other features. The output then can be used in various ways to represent the directionality of each feature in the model.

## Input
The code currently takes two pickle files as inputs, one containing the trained `sklearn.ensemble.RandomForestClassifier` object and another containing the data matrix. Note that the code expects an entry with the name ***modelobj*** in the first pickle file containing the trained model object, an entry with the name ***feature_importances_names*** in the first pickle file containing the names of all features, and an entry with the name ***test_X*** in the second pickle file containing the data matrix. This will be generalized in later versions.

## Output
In the current notebook format, the code will return 1) the P(y=1|x) curves and detected transitions in them, and 2) the naive directed importance of P(y=1|x=x_max)-P(y|x=x_min).

## Caveats
Currently the code only works for numerical features with more than one feature value in the data. Implementation for categorical variables will happen in later versions.


## Attribution
If you found this code useful in your work, please cite it.