from domain import Domain
class Agent:

    def __init__(self, domain, policy):
        # initialize dynamics and policy
        self.domain = domain
        self.pol = policy


    # selects an action to execute from the current state (position "p" and speed "s")
    def policy(self, p, s):
        if self.pol == "ACC":
            return self.domain.ACTIONS[1] # returns "acc"

