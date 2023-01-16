#!/bin/bash

########################################################################################################################
############################### Discrete Experiments ###################################################################
########################################################################################################################

####################
# Important parameters
####################
EPSILON=".1"
TEST_RATIO=".33"
LAMBDA_GRID="1e-5,1e-4,1e-3,1e-2,1e-1,1,1e1,1e2,1e3,1e4,1e5"
NB_ROLLOUTS=10
REPLAYS_PER_LINE=4
ROLLOUT_SCHEME=linear
#ROLLOUT_SCHEME=doubling
IPS_IX=""
#IPS_IX="--ips-ix"
VAR_ESTIMATION=""
#VAR_ESTIMATION="--truevar"
NB_RUNS_TO_ESTIMATE_VARIANCE=10 ## includes (train/test split + action) variance
PREFIX=''  # for naming files, e.g. "--prefix toto"

COMMON_PARAMS="
--epsilon $EPSILON --test-size $TEST_RATIO --n-rollouts $NB_ROLLOUTS --rollout-scheme $ROLLOUT_SCHEME
--n-replays $REPLAYS_PER_LINE --n-reruns $NB_RUNS_TO_ESTIMATE_VARIANCE $PREFIX $IPS_IX $VAR_ESTIMATION"
####################


####################
# EXP1: compare CRM vs SCRM vs base/skylines with a posteriori lambda selection
python compare_crm_scrm.py scene,yeast,tmc2007 --lambda-grid $LAMBDA_GRID $COMMON_PARAMS
####################


####################
# EXP2: loss evolution CRM vs SCRM with best known lambda for each
####################
BEST_LAMBDA_CRM=$(grep best_lambda_crm "compare_crm_scrm_discrete--scene.cfg" | cut -f2 -d ':')
BEST_LAMBDA_SCRM=$(grep best_lambda_scrm "compare_crm_scrm_discrete--scene.cfg" | cut -f2 -d ':')
echo "Loss evolution on scene with lambdas $BEST_LAMBDA_CRM / $BEST_LAMBDA_SCRM"
python loss_evolution_crm_scrm.py scene $BEST_LAMBDA_CRM $BEST_LAMBDA_SCRM $COMMON_PARAMS

BEST_LAMBDA_CRM=$(grep best_lambda_crm "compare_crm_scrm_discrete--yeast.cfg" | cut -f2 -d ':')
BEST_LAMBDA_SCRM=$(grep best_lambda_scrm "compare_crm_scrm_discrete--yeast.cfg" | cut -f2 -d ':')
echo "Loss evolution on yeast with lambdas $BEST_LAMBDA_CRM / $BEST_LAMBDA_SCRM"
python loss_evolution_crm_scrm.py yeast $BEST_LAMBDA_CRM $BEST_LAMBDA_SCRM $COMMON_PARAMS

BEST_LAMBDA_CRM=$(grep best_lambda_crm "compare_crm_scrm_discrete--tmc2007.cfg" | cut -f2 -d ':')
BEST_LAMBDA_SCRM=$(grep best_lambda_scrm "compare_crm_scrm_discrete--tmc2007.cfg" | cut -f2 -d ':')
echo "Loss evolution on tmc2007 with lambdas $BEST_LAMBDA_CRM / $BEST_LAMBDA_SCRM"
python loss_evolution_crm_scrm.py tmc2007 $BEST_LAMBDA_CRM $BEST_LAMBDA_SCRM $COMMON_PARAMS
####################


####################
# EXP3: autotuned lambda vs best lambda a posteriori
####################
python lambda_heuristic.py scene,yeast,tmc2007 $COMMON_PARAMS -j 4
####################
