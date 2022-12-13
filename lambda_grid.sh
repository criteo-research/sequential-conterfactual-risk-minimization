#!/bin/bash

GRID='1e-3,1e-1,.5,0,-.5,-1e-1,-1e-3'

python lambda_grid.py yeast -rol 5 -rep 4 -rer 10 -rs linear -lg $GRID &
python lambda_grid.py yeast -rol 5 -rep 4 -rer 10 -rs doubling -lg $GRID &

python lambda_grid.py scene -rol 5 -rep 4 -rer 10 -rs linear -lg $GRID &
python lambda_grid.py scene -rol 5 -rep 4 -rer 10 -rs doubling -lg $GRID &

python lambda_grid.py tmc2007 -rol 10 -rep 4 -rer 5 -rs linear -lg $GRID &
python lambda_grid.py tmc2007 -rol 10 -rep 4 -rer 5 -rs doubling -lg $GRID
