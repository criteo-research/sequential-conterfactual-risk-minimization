# Sequential Counterfactual Risk Minimization

This repository contains the code for reproducing experiments of the paper Sequential Counterfactual Risk Minimization submitted at ICML.

## Prerequisites

```
pip install -U -r requirements.txt
```

## Instructions

### Discrete Experiments

```
bash generate_figures_and_tables.sh
```

### Continuous Experiments

Go into the `continuous`sub folder and run the following.

Figure 1: run the Jupyter notebook 

```
gaussian_example.ipynb
```

Figure 2 SCRM vs CRM, continuous datasets 

```
python scrm_vs_crm.py
```

Table 1 SCRM vs CRM, example 3.1

```
python gaussian_example.py
```

Table 2 SCRM vs baselines, continuous datasets

```
python scrm_vs_baselines_counterfactual.py
python scrm_vs_baselines_sbpe.py
python scrm_vs_baselines_bkucb.py
python scrm_vs_rl.py

```

Figure 4: run the Jupyter notebook 

```
distance_scrm_gaussian_example.ipynb
```
### RL Baselines

`$ ...`


