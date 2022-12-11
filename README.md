# DOA estimation methods
```
CBF, MUSIC, MVDR, RVFL, RVFLv, ML-ELM, edRVFL, ME-SRVFLv, ME-SedRVFL [PYTHON & MATLAB]

We design a signal model-embedded loss function, which is a scenario-related solving function of the closed-form solution. 

Our method: ME-SedRVFL and ME-SRVFlv, a hybrid method, which embedded the signal model to  objective function for the self-supervised (SS) DOA estimation. 

We have reimplemented the parameters estimation tasks on our underwater acoustic datasets (simulations and experiments). 
```
## ME-SedRVFL and ME-SRVFlv
```
horizontal linear array (HLA) received signal model

custom loss funtion

beamforming
```


## Parameters used in randomization-based methods

```
n_types:
               The Total number of label types in the dataset.
seed:
                Random seeds.
N/n_nodes:
                The number of hidden neurons.
L:
                The number of hidden layers.
C/lam:
                The regularization parameter λ
nCV:
                The number of folds in cross validation.
scale：
                Linearly scale the random features before feedinto the nonlinear activation function. In this implementation, we consider the threshold which lead to 0.99 of the maximum/minimum value of the activation function as the saturating threshold. option.scale=0.9 means all the random features will be linearly scaledinto 0.9* [lower_saturating_threshold,upper_saturating_threshold].
w_random_vec_range: 
				A list, [min, max], the range of generating random weights.
b_random_vec_range: 
				A list, [min, max], the range of generating random bias.
random_weights:
				A Numpy array shape is [n_feature, n_nodes], weights of neuron.
random_bias: 
				A Numpy array shape is [n_nodes], bias of neuron.
beta: 
				A Numpy array shape is [n_feature + n_nodes, n_class], the 
task_type: 
				A string of ML task type, 'classification' or 'regression'.
```







