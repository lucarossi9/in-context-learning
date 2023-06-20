# In-context learning regression
![](setting.jpg)

## Table of contents
* [General info](#general-info)
* [Setup](#setup)
* [Code structure](#code structure)
* [Credits](#credits)

## General info
This repository contains the code for the "in-context learning regression" project. It allows you to evaluate our trained models, reproduce our results and train new models. As an example, you can replicate the plot investigating the closeness of the posterior mean-based estimator and Transformer.
![](OLS-random-mean.png)
	
## Setup

You can start by cloning our repository and following the steps below.

1. Install the dependencies for our code using Conda. You may need to adjust the environment YAML file depending on your setup.

    ```
    conda env create -f environment.yml
    conda activate in-context-learning
    ```

2. Download [model checkpoints](https://github.com/dtsip/in-context-learning/releases/download/initial/models.zip) and extract them in the current directory.

    ```
    wget https://github.com/lucarossi9/in-context-learning/releases/download/initial/models.zip
    unzip models.zip
    ```

3. [Optional] If you plan to train, populate `conf/wandb.yaml` with the information regarding your wandb account.

## Code structure
That's it! You can now explore our pre-trained models or train your own. The key entry points
are as follows (starting from `src`):
- The `eval.ipynb` notebook contains code to load our own pre-trained models, plot the pre-computed metrics, and evaluate them on new data.
- `train.py` takes as argument a configuration yaml from `conf` and trains the corresponding model. You can try `python train.py --config conf/toy.yaml` for a quick training run.



## Credits

Our codes are in part borrowed by the work
*What Can Transformers Learn In-Context? A Case Study of Simple Function Classes** <br>
*Shivam Garg\*, Dimitris Tsipras\*, Percy Liang, Gregory Valiant* <br>
Paper: http://arxiv.org/abs/2208.01066 <br><br>
```bibtex
    @misc{garg2023transformers,
      title={What Can Transformers Learn In-Context? A Case Study of Simple Function Classes}, 
      author={Shivam Garg and Dimitris Tsipras and Percy Liang and Gregory Valiant},
      year={2023},
      eprint={2208.01066},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
