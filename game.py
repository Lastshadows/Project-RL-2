from domain import Domain
from agent import Agent
class Game:

    def __init__(self, p, s, policy, steps):

        # current state
        self.p = p
        self.s = s
        self.t = 0

        # max number of steps played
        self.steps = steps

        # import the dynamics
        self.domain = Domain()

        # create the agent
        self.agent =  Agent(self.domain, policy)

        # create a memory for the taken trajectory
        self.trajectory = []

    def playGame(self):
        i = 0

        # we play as long as we are not in a temrinal state or havent played a given amount of steps
        while( (not self.domain.isTerminalState(self.p, self.s)) and i < self.steps):

            # generate an action based on the policy of the agent
            action = self.agent.policy(self.p,self.s)

            # fill the trajectory
            r = self.domain.rewardSignal(self.p, self.s)
            self.trajectory.append(((self.p, self.s), action,r))

            # update our current state according to our dynamics and the action
            next_state = self.domain.dynamics(self.p, self.s, action, self.t)
            self.p, self.s, self.t = next_state


            i+= 1

