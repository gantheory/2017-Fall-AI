#!/bin/bash
python2 dataClassifier.py -c perceptron -d pacman -f -g ContestAgent -t 1000 -s 1000
python2 dataClassifier.py -c perceptron -d pacman -f -g FoodAgent -t 1000 -s 1000
python2 dataClassifier.py -c perceptron -d pacman -f -g StopAgent -t 1000 -s 1000
python2 dataClassifier.py -c perceptron -d pacman -f -g SuicideAgent -t 1000 -s 1000

