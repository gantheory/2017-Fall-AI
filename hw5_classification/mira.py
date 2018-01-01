# mira.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# Mira implementation
import util
PRINT = True

class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            Cgrid = [0.002, 0.004, 0.008]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.

        ## Something you might need to use:
        ## trainingData[i]: a feature vector, an util.Counter()
        ## trainingLabels[i]: label for each trainingData[i]
        ## self.weights[label]: weight vector for a label (class), an util.Counter()
        ## self.classify(data): this method might be needed in validation
        ## Cgrid: a list of constant C
        """
        "*** YOUR CODE HERE ***"
        classifiers = []
        guessRight = [len(validationData)] * len(Cgrid)
        for C in Cgrid:
            classifiers.append(self.weights.copy())

        for i in range(len(Cgrid)):
            C = Cgrid[i]
            classifier = classifiers[i]

            for iteration in range(self.max_iterations):
                for j in range(len(trainingData)):
                    nowData = trainingData[j]
                    nowLabel = trainingLabels[j]

                    nowScore = util.Counter()
                    for label, weight in classifier.iteritems():
                        nowScore[label] = weight * nowData
                    predictLabel = nowScore.argMax()

                    if predictLabel != nowLabel:
                        numerator = (self.weights[predictLabel] - self.weights[nowLabel]) * nowData + 1.0
                        denominator = 2.0 * (nowData * nowData)
                        tau = min([C, numerator / denominator])

                        for key in nowData:
                            nowData[key] *= tau
                        classifier[nowLabel] += nowData
                        classifier[predictLabel] -= nowData

            for j in range(len(validationData)):
                nowData = validationData[j]
                nowLabel = validationLabels[j]

                nowScore = util.Counter()
                for label, weight in classifier.iteritems():
                    nowScore[label] = weight * nowData
                predictLabel = nowScore.argMax()

                if predictLabel != nowLabel:
                    guessRight[i] -= 1

        best = max(guessRight)
        for i in range(len(guessRight)):
            if guessRight[i] == best:
                self.weights = classifiers[i]
                return

    def classify(self, data ):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses


