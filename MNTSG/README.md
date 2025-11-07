# MNTSG

We propose MN-TSG, a framework that explores MOE (Mixture of Experts)-NCDE and integrates it with existing TSG models for irregular or continuous TSG tasks. The key designs of MOE-NCDE are the dynamic functions with mixture of experts and the decoupled design to better optimize the MOE dynamics. Further, we employ the existing TSG model to learn the joint distribution of the mixture of experts and the time series. In this way, the model can not only generate new samples but also produce suitable experts for them to enable MOE-NCDE for refined continuous TSG tasks. 

## Environment
Install the environment from the yaml file given here: environment_mntsg.yml

```bash
conda env create -f environment.yml --force --no-deps
```

## Data
Stocks and Energy data are located in /datasets. Sine, MuJoCo, polynomial datasets are generated and the scripts are included in datasets folder.
utils_data.py contains functions to load the data both in regular and irregular setups. Specifically, the irregular data is pre-processed by the TimeDataset_irregular class,
and it might take a while. Once the data pre-processing is done, it is saved in the /datasets folder.


## Reproducing the paper results
By setting the time series length and missing values within the script, the results in the paper can be reproduced:


**Step1:** The MOE NeuralCDE can be trained by `run_irregular_moencde.py`.

---

**Step2:** The training of the joint distribution of MOE expert weights and time series samples can be implemented through script `run_mntsg_diffsuion.py`.

---

**Step3:** By using the initially trained MOE-NeuralCDE, each newly generated sample, and the corresponding generated MOE weights, fine-grained and refined continuous time series generation can be achieved through script `run_irregular_moencde_continues.py`, thereby enhancing the accuracy of downstream tasks with richer temporal information.
