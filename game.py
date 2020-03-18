from domain import Domain
from agent import Agent

class Game:

    def __init__(self, p, s, policy, steps):

        # current state
        self.p = p
        self.s = s
        self.t = 0
        self.gamma = 0.95
        self.isWon = False

        # max number of steps played
        self.steps = steps

        self.reward = 0

        # import the dynamics
        self.domain = Domain()

        # create the agent
        self.agent =  Agent(self.domain, policy)

        # create a memory for the taken trajectory
        self.trajectory = []
        self.fullTrajectory = []

    def getReward(self):
        return self.reward

    def playGame(self):
        i = 0

        # we play as long as we are not in a temrinal state or havent played a given amount of steps
        while i < self.steps:
            # generate an action based on the policy of the agent
            action = self.agent.policy(self.p,self.s)
            # print(" the selected action is " + str(action) + "\n")

            # getting the resulting state from state and action
            next_state = self.domain.dynamics(self.p, self.s, action, self.t)
            p,s,t = next_state

            # fill the trajectory
            r = self.domain.rewardSignal(p, s)
            self.reward = self.reward + pow(self.gamma, i) * r
            self.trajectory.append(( (p,s), action, r ))   # after bardhyl
            #self.trajectory.append(((self.p, self.s), action, r))  # before bardhyl

            # update our current state
            self.p, self.s, self.t = next_state

            if self.domain.isTerminalState(self.p, self.s):
                # check if won
                if self.domain.rewardSignal(self.p, self.s) == 1:
                    self.isWon = True

                break

            i+= 1

        if self.domain.rewardSignal(self.p,self.s)==0:
            print(self.domain.rewardSignal(self.p, self.s))

    def playGameTillEnd(self):
        i = 0
        # we play as long as we are not in a terminal state
        while self.domain.isTerminalState(self.p, self.s) == False:

            # generate an action based on the policy of the agent
            action = self.agent.policy(self.p,self.s)

            # getting the resulting state from state and action
            next_state = self.domain.dynamics(self.p, self.s, action, self.t)
            p,s,t = next_state

            # fill the trajectory
            r = self.domain.rewardSignal(p, s)
            self.reward = self.reward + pow(self.gamma, i) * r
            self.fullTrajectory.append(((self.p, self.s), action,(p,s), r))

            # update our current state
            self.p, self.s, self.t = next_state
            i = i+1

        # check if won
        if self.domain.rewardSignal(self.p, self.s) == 1:
            self.isWon = True

    # sets the parameters for a FQI policy game
    # 'policy'  is a string giving the nature of the SL model type (tree, linear or network) that the FQI algo will use
    # N is the number of iteration the FQI algo will use if this is the chosen policy
    # trajectory is the trajectory that the FQI would use to build its model
    # it must be a (x,u,r) tuple, where x is a (p,s) tuple
    def setToFQI(self, policy_FQI, trajectory, N, nb_games):

        # change the policy name
        self.policy_name = policy_FQI
        # create the FQI agent and replace the original one
        self.agent = Agent(self.domain, policy_FQI, trajectory, N, nb_games)
