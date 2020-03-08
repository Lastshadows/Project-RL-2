from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesClassifier
from domain import Domain
import numpy as np

class FittedQItLearner:
    """
    Initializes the learner.

    'model_type' is a string stating either "tree", "linear" or "network", the model that will be used
    for the estimation of the Q function

    'trajectory' is a list of tuple (x,u,r), where x is (p,s)

    'N' is the max number of iteration

    """
    def __init__(self, model_type, trajectory, N):

        self.model_type = model_type
        self.trajectory = trajectory
        self.domain = Domain()
        self.model = self.Q_iter(N)



    def Q_iter(self, N):

        X_U = []
        R = []

        for tuple in self.trajectory:

            # extraction of the elements of the trajectory into their respective lists
            x,u,r = tuple
            p,s = x

            X_U.append((p,s,u))
            R.append(r)

        X_U = np.array(X_U)
        R = np.array(R)

        # compute approximation of Q1
        if(self.model_type == "linear"):
            model = LinearRegression().fit(X_U, R)

        elif(self.model_type == "tree"):
            model = ExtraTreesClassifier(n_estimators=100, random_state=0)
            model.fit(X_U, R)

        elif (self.model_type == "network"):
            print("error, model not done yet")

        else: print("error, model type not available")

        self.model = model

        for i in(range(N)):

            R_i = []

            for tuple in self.trajectory:

                # building the training set for next iteration of Q
                x, u, r = tuple
                p, s = x

                # building the next state and the cumulated r
                p2, s2, t2 = self.domain.dynamics( p, s, u, 0) # we dont care about t
                cumulated_r = r + self.domain.DISCOUNT_FACTOR*self.maxPreviousQ(p2,s2)

                R_i = np.append(cumulated_r, R_i)

            R_i = np.array(R_i)

            # training the new model to approximate Qi
            model.fit(X_U, R_i)

        return model

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

    # returns the reward estimated by the Q_N model for a given state and action
    def rewardFromModel(self, x,u):

        p,s = x
        to_predict = [[p, s, u]]  # to avoid dimensionality problems
        reward = self.model.predict((to_predict))

        return reward