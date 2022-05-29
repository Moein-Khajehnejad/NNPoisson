# NNPoisson

NNPoisson is an implementation of our end-to-end model which exploits recent developments of flexible, but tractable, neural network point-process models to characterize dependencies between stimuli, actions, and neural data. 
The paper introducing the details of the model is in Proceedings of **the 39th International Conference on Machine Learning (ICML 2022)** .

## Dataset
1. Steinmetz dataset: We use a public dataset collected using Neuropixel probes in mice performing a visually-guided behavioural task. (Steinmetz, Nicholas A., et al. "Distributed coding of choice, action and engagement across the mouse brain." Nature 576.7786 (2019): 266-273.) 
2. Synthetic dataset: We also use a synthetic dataset produced from a hierarchical network model with reciprocally connected sensory and integration circuits intended to characterize animal behaviour in a fixed-duration motion discrimination task. (Wimmer, Klaus, et al. "Sensory integration dynamics in a hierarchical network explains choice probabilities in cortical area MT." Nature communications 6.1 (2015): 1-13.)

## Model Structure

<img width="889" alt="model_structure" src="https://user-images.githubusercontent.com/22978025/170870161-ce26589e-9df7-4fb5-898e-159a58b741ec.png">



## Usage

After cloning/downloading the repository:
- Add the following folders to your working path: 1)expr, 2)Steinmetz_Model, and 3)Synthetic_Model
- Make sure you have downloaded the data files for both Steinmetz and Synthetic datasets
- For each dataset, run 'main.py' stored in the corresponding folder of that model.
- The outputs will be stored in the '../nongit/' folder created by you in the base directory.


## Cite NNPoisson
If you find this model useful to your work, please also cite our paper at https://doi.org/10.1101/2020.07.13.201673 
