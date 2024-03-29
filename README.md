# Sequential Counterfactual Risk Minimization

This repository contains the code for reproducing experiments of the paper Sequential Counterfactual Risk Minimization submitted at ICML.

## Prerequisites

Software:

```
pip install -U -r requirements.txt
```

Download discrete datasets from the LibSVM website:

```
BASEURL="https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel"
wget $BASEURL/yeast_test.svm.bz2
wget $BASEURL/yeast_train.svm.bz2
wget $BASEURL/scene_test.bz2
wget $BASEURL/scene_train.bz2
wget $BASEURL/tmc2007_test.svm.bz2
wget $BASEURL/tmc2007_train.svm.bz2

bunzip2 *bz2
```

## Instructions

### Discrete Experiments

```
bash generate_figures_and_tables.sh
```


### Continuous Experiments

Go into the `continuous`sub folder and run the following.

Figure 1: run the Jupyter notebook in continuous/

```
gaussian_example.ipynb
```

Figure 2 SCRM vs CRM, continuous datasets 

```
python continuous/scrm_vs_crm.py
```

Table 1 SCRM vs CRM, example 3.1 

```
python continuous/gaussian_example.py
```

Table 3 SCRM vs baselines, continuous datasets

```
python continuous/scrm_vs_baselines_counterfactual.py
python continuous/scrm_vs_baselines_sbpe.py
python continuous/scrm_vs_baselines_bkucb.py
python continuous/scrm_vs_rl.py

```

Figure 3,10: run the Jupyter notebook

```
compare_estimators.ipynb
```


Figure 4: run the Jupyter notebook in continuous/

```
distance_scrm_gaussian_example.ipynb
```


### Batch bandits Baselines
Discrete datasets

```
python batch_bandits_experiments.py
```
Continuous datasets

```
python continuous/scrm_vs_baselines_sbpe.py
python continuous/scrm_vs_baselines_bkucb.py
```


### RL Baselines
Discrete datasets

```
python compare_rl_scrm.py scene,yeast,tmc2007
```
Continuous datasets

```
python continuous/scrm_vs_rl.py
```


Figure 8: run the Jupyter notebook in continuous/

```
holderian_error_bound_assumption.ipynb
```