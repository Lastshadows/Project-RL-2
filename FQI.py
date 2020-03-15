from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from domain import Domain
import numpy as np
import os.path
from joblib import dump, load

class FittedQItLearner:
    """
    Initializes the learner.

    'model_type' is a string stating either "tree", "linear" or "network", the model that will be used
    for the estimation of the Q function

    'trajectory' is a list of tuple (x,u,r), where x is (p,s)

    'N' is the max number of iteration

    """
    def __init__(self, model_type, trajectory, N, nb_games):

        # parameters of the models
        self.trees_n_estimators = 50

        self.model_type = model_type
        self.trajectory = trajectory
        self.nb_games = nb_games
        self.domain = Domain()
        self.model = self.Q_iter(N)


    def Q_iter(self, N):

        X_U = []
        R = []
        win = 0
        lost = 0

        for tuple in self.trajectory:

            # extraction of the elements of the trajectory into their respective lists
            x,u,r = tuple
            p,s = x

            X_U.append((p,s,u))
            R.append(r)

            # count the amount of winned games in the training
            if r == 1:
                win += 1

            # count the amount of lost games in the training
            if r == -1:
                lost += 1

        print(" \nthere were " + str(win) + " winned games and " + str(lost) + " lost games")
        print(" \nstarting iterations of the FQI")

        X_U = np.array(X_U)
        R = np.array(R)
        name_model = ''

        # compute approximation of Q1
        if(self.model_type == "linear"):

            model = LinearRegression()

            # creating the name of the model
            name_model =   self.model_type +"_" + str(self.nb_games) + "_games_it_" + str(0)

            # if the exact same model has already been made (same name), just load it
            if os.path.isfile(name_model):
                model = load(name_model)

            # otherwise, create the model
            else: model = LinearRegression().fit(X_U, R)

        elif(self.model_type == "tree"):

            model = ExtraTreesRegressor(n_estimators= self.trees_n_estimators, random_state=0)

            # creating the name of the model
            name_model = self.model_type  + '_' + str(self.trees_n_estimators) + "_" + str(self.nb_games) + "_games_it_" + str(0)


            # if the exact same model has already been made (same name), just load it
            if os.path.isfile(name_model):
                model = load(name_model)

            # otherwise, create the model
            else : model.fit(X_U, R)

        elif (self.model_type == "network"):
            print("error, model not done yet")

        else:
            print("error, model type not available")

        # saving the model for potential later uses
        dump(model, name_model )

        self.model = model

        for i in(range(N)):

            print(" \nFQI iteration " + str(i+1) + " out of " + str(N) + " \n")
            R_i = []

            # writing down the name of the model we are about to build
            name_model = ''

            if self.model_type == "tree":
                name_model = self.model_type  + '_' + str(self.trees_n_estimators) + "_" + str(self.nb_games) + "_games_it_" + str(i)

            if self.model_type == 'linear':
                name_model =  self.model_type + "_" + str(self.nb_games) + "_games_it_" + str(i)

            if self.model_type == 'network':
                print(" no such models yet, need to create model names")

            # if we have already built a model for this scenario, skip the training set build
            if not (os.path.isfile(name_model)):
                for tuple in self.trajectory:

                    # building the training set for next iteration of Q
                    x, u, r = tuple
                    p, s = x

                    # building the next state and the cumulated r
                    p2, s2, t2 = self.domain.dynamics( p, s, u, 0) # we dont care about t
                    cumulated_r = r + self.domain.DISCOUNT_FACTOR*self.maxPreviousQ(p2,s2)

                    R_i = np.append(cumulated_r, R_i) # here maybe speed loss

            R_i = np.array(R_i)

            # if the exact same model has already been made (same name), just load it
            if os.path.isfile(name_model):
                self.model = load(name_model)
                print("existing model was found")

            else:
                # otherwise training the new model to approximate Qi
                self.model.fit(X_U, R_i)
                dump(self.model, name_model)

        return self.model

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