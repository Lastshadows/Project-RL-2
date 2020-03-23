from domain import Domain
import random
from FQI import FittedQItLearner

class Agent:

    # 'domain' is a variable of domain type
    # 'policy'  is a string giving the nature of the policy the agent will follow
    # if it is a SL model type (tree, linear or network), a FQI algo will be used
    # N is the number of iteration the FQI algo will use if this is the chosen policy
    # trajectory is the trajectory that the FQI would use to build its model
    # it must be a (x,u,r) tuple, where x is a (p,s) tuple
    def __init__(self, domain, policy_FQI, trajectory = [], N = 0, nb_games  = 0 ):
        # initialize dynamics and policy
        self.domain = domain
        self.policy_name = policy_FQI
        self.trajectoryFQI = trajectory
        self.N_FQI = N

        if N > 0:
            self.FQI = FittedQItLearner(policy_FQI, trajectory, N, nb_games)

    # selects an action (-4 or 4) to execute from the current state (position "p" and speed "s")
    def policy(self, p, s):

        if self.policy_name == "ACC":
            return self.domain.ACTIONS[1] # returns "acc"

        if self.policy_name == "RAND":
            rand = random.randint(0,1)
            return self.domain.ACTIONS[rand]

        if self.policy_name =="tree" :
            x = (p,s)
            return self.selectBestActionFromFQI(x)

        if self.policy_name =="linear" :
            x = (p,s)
            return self.selectBestActionFromFQI(x)

        if self.policy_name =="network" :
            x = (p,s)
            return self.selectBestActionFromFQI(x)


    # takes a state x (p,s) and gives back the best action to take according to
    # the FQI model
    def selectBestActionFromFQIOld(self, x):

        best_reward = float("-inf")
        best_action = 0

        for u in self.domain.ACTIONS:

            reward =  self.FQI.rewardFromModel(x, u)
            print("selecting best move : currently analysing move " + str(u)  )
            print("reward is : " + str(reward) + ", current best reward is "+ str(best_reward))
            if reward >= best_reward:
                best_reward = reward
                best_action = u

        print("best action is : " + str(best_action) )

        return best_action

    # takes a state x (p,s) and gives back the best action to take according to
    # the FQI model
    def selectBestActionFromFQI(self, x):

        best_reward = float("-inf")
        best_action = 0

        for u in self.domain.ACTIONS:

            reward =  self.FQI.rewardFromModel(x, u)
            # print("selecting best move : currently analysing move " + str(u)  )
            # print("reward is : " + str(reward) + ", current best reward is "+ str(best_reward))
            if reward >= best_reward:
                best_reward = reward
                best_action = u

        # print("best action is : " + str(best_action) )

        return best_action
