from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process.kernels import RBF
import sklearn.gaussian_process as gp
from domain import Domain
import numpy as np
import os.path
import sys
from joblib import dump, load

class PQL:
    """
    Initializes the learner.

    'model_type' is a string stating either "radial" or "network", the model that will be used
    for the estimation of the Q function

    'trajectory' is a list of tuple (x,u,x',r), where x is (p,s)

    """
    def __init__(self, model_type, trajectory):

        # parameters of the models

        self.model_type = model_type
        self.trajectory = trajectory
        self.domain = Domain()
        self.model = self.PQLearner()
        self.param = [] # param of the model

        if self.model_type == "radial":
            self.param = []


    def PQLearner(self):

        Q0 = self.getNewModel()

        xtrain = np.zeros((len(self.trajectory),3))
        ytrain = np.zeros(len(self.trajectory))
        j = 0

        #generate intial training set
        for (pt,st),action,(pnext,snext),reward in self.trajectory:
            xtrain[j][0] = pt
            xtrain[j][1] = st
            xtrain[j][2] = action
            ytrain[j] = reward
            j=j+1

        Q0.fit(xtrain,ytrain)
        return Q0

    # gives the max reward obtainable from a given state x for all actions possibles
    # u using the last SL model
    def maxPreviousQ(self,p,s):

        best_reward = float("-inf")

        for action in self.domain.ACTIONS:

            to_predict = [[p, s, action]] # to avoid dimensionality problems
            reward = self.model.predict((to_predict))

            if(reward > best_reward):
                best_reward = reward

        return best_reward

    # returns the reward estimated by the PQL model for a given state and action
    def rewardFromModel(self, x,u):

        currStateAction = np.zeros((1,3))
        p,s = x

        currStateAction[0][0]=p
        currStateAction[0][1]=s
        currStateAction[0][2]=u
        reward = self.model.predict(currStateAction)

        return reward

    def getNewModel(self):

        if(self.model_type == "radial"):

            kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(10.0, (1e-3, 1e3))
            model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True)
            return model


        elif (self.model_type == "network"):
            print("error, model not done yet in getNewModel")
            return 0

        else:
            print("error, model type not available")

            return 0
