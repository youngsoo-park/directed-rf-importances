# monte-carlo-model-interpertation

## Overview

Interpreting the behavior of a model, i.e. explaining a particular output of a model from the corresponding input, can become non-trivial for many commonly used algorithms. This is largely because these models draw high-dimensional probability surfaces with complex shapes, thereby making simple generalizations of model behavior (e.g. coefficients in linear models) impossible. As an example, even if we have a well-trained random forest classifier at hand, for a given classification result we cannot easily determine how each feature contributed to that result.

The goal of this simple module is to produce intuitive results about the marginalized effect of a given feature in a model. More specifically, Monte Carlo sampling is used to generate values of the conditional probabiltiy P(y=1|x_i) for a feature x_i in fine bins, while marginalizing over all other features by uniformly sampling from the entire feature space. While the resulting conditional probability curves are insensitive to feature interactions, they can be considered as a proxy for the global behavior of a feature in a model.


## Usage

To generate the conditional probability curves, you need three inputs: a trained `sklearn` model object, a representative dataset (e.g. the training data used), and a list of feature names. You can then run

```
import montecarlo as mc
interpreter = mc.MonteCarlo(model=model, X_in=X_in, X_names=X_names)
interpreter.generate(n_sample=n_sample, n_bins=n_bins)
print interpreter.curves
```
Here, `model` is the trained model object, `X_in` is the dataset, and `X_names` is the list of feature names. In the last line, you can specify how many Monte Carlo samples to generate with `n_sample` and how many bins you want the conditional probabilty curves to have by `n_bins`. Running the `generate` function will create a 3D numpy list, `curves`, where `curves[i,j,0]` is the conditional probability value for the i-th feature in the j-th bin and `curves[i,j,1]` is the measured standard deviation for that value.


## Utilities

To plot all of the generated curves, you can run
`interpreter.plot_curves()`.

To perform a basic edge detection in the generated curves, which can be useful for identifying potential sharp transitions corresponding to e.g. random forest decision criteria, you can run
`interpreter.edge_detect()`. This will generate a set of plots similar to those from `plot_curves` but with red arrows marking the detected sharp transitions. It will also generate a new attribute `transitions`, a 3D python list, where `transitions[i][j][0]` is the bin location of the j-th detected transition in the i-th feature, and `transitions[i][j][1]` is the magnitude of that transition.


## Limitations

The code may have potential issues dealing with non-numerical (binary, categorical, etc.) features, especially with the edge detection. 

## Attribution
If you found this code useful in your work, please cite it.
