from math import sqrt

class Domain:
    def __init__(self):

        # upper and lower bounds on p and s
        self.MAX_P = 1
        self.MAX_S = 3
        self.MIN_P = -1
        self.MIN_S = -3

        # actions
        self.ACTIONS = [-4, 4]
        self.ACTIONS_DICT = {"dec": -4, "acc": 4}
        self.ACTIONS_NAMES = ["dec","acc"]

        # physics constants
        self.M = 1
        self.G = 9.81

        # domain constants
        self.INTEGRATION_TIME_STEPS = 0.001
        self.TIME_DISCR = 0.1
        self.DISCOUNT_FACTOR =  0.95



    # returns true if we are in a terminal state
    def isTerminalState(self,p, s):
        if abs(p) > 1 or abs(s) > 3:
            return True
        return False

    # returns the instantenous reward
    def rewardSignal(self, p, s):
        reward = 0
        if (p < -1) or (abs(s) > 3 ):
            reward = -1
        if (p > 1) and (abs(s) <= 3 ):
            reward = 1
        return reward

    # returns the hill's height from the position p
    def hill(self, p):
        if p < 0:
            return p*p + p
        else:
            return p/(sqrt(1+5*p*p))

    # define the slope of the hill
    def hillPrime(self, p):

        if p < 0:
            return 2*p + 1
        return 1/((5*(p**2)+1)**(3/2))

    # derivative of the slope of the hill
    def hillDoublePrime(self,p):
        if p < 0:
            return 2
        return (-15*p)/((5*(p**2)+1)**(5/2))

    def pPrime(self, s):
        return s

    # returns the instant acceleration
    def sPrime(self, p,s,u):

        first_term = u/(self.M*(1 + self.hillPrime(p)**2))
        second_term =  - (self.G * self.hillPrime(p))/(1 + self.hillPrime(p)**2)
        third_term =  - (s**2 * self.hillPrime(p) * self.hillDoublePrime(p))/(1 + self.hillPrime(p)**2 )

        sum = first_term + second_term + third_term
        return sum


    def euler(self,  p_0, s_0, u):
        """
        Perform integration with simple euler method
        :param p_0: initial position
        :param s_0: initial speed
        :param u: acceleration value during two timesteps
        :return: (position, speed) after a timestep elapsed
        """

        n = int(self.TIME_DISCR/self.INTEGRATION_TIME_STEPS) + 1

        p = p_0
        s = s_0

        for i in range(n):
            s += self.INTEGRATION_TIME_STEPS * self.sPrime(p, s, u)
            p += self.INTEGRATION_TIME_STEPS * s
        return p, s

    def dynamics(self, p_0, s_0, u, t_0):
        """
        Describe the dynamics of the domain
        :param p_0: initial position
        :param s_0: initial speed
        :param u: action (acceleration value)
        :param t_0: initial time
        :return: nextp, next_s, next_t
        """

        next_p, next_s = self.euler( p_0, s_0, u)

        return next_p, next_s, t_0 + self.TIME_DISCR
