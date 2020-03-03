from domain import Domain
import random
class Agent:

    def __init__(self, domain, policy):
        # initialize dynamics and policy
        self.domain = domain
        self.policy_name = policy


    # selects an action (-4 or 4) to execute from the current state (position "p" and speed "s")
    def policy(self, p, s):

        if self.policy_name == "ACC":
            return self.domain.ACTIONS[1] # returns "acc"

        if self.policy_name == "RAND":
            rand = random.randint(0,1)
            return self.domain.ACTIONS[rand]


