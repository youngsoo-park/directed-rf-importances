# monte-carlo-model-interpertation

## Overview
In general, random forest classifiers such as `sklearn.ensemble.RandomForestClassifier` will only yield undirected feature importances, making it difficult to determine the directionality of a feature. This script uses Monte Carlo sampling to estimate Prob(y=1|x) for a feature x across its entire range, while marginalizing over all other features. The output then can be used in various ways to represent the directionality of each feature in the model.

## Attribution
If you found this code useful in your work, please cite it.