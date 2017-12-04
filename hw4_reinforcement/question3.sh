#!/bin/bash
python gridworld.py -a value -i 100 -g DiscountGrid --discount $1 --noise $2 --livingReward $3
