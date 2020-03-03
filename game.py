from domain import domain

class Game:

    def __init__(self, p, s):

        # initial state
        self.p = p
        self.s = s

        # import the dynamics
        self.domain = domain()

