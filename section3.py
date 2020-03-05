from game import Game
import random 
import time
import sys
import numpy as np


if __name__ == '__main__':

    policy = "ACC"
    steps = 1000
    N = 100
    toolbar_width = 50

    print("\nGENERATING TRAJECTORIES:\n")
    # setup toolbar
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

    rewards = np.zeros(N)

    for i in range(N):
        p = random.uniform(-0.1, 0.1)
        s = 0
        game = Game(p,s, policy, steps)
        game.playGame()
        rewards[i]=game.get_reward()
        if (i+1)%(N/toolbar_width)==0:
            sys.stdout.write("-")
            sys.stdout.flush()

    sys.stdout.write("]\n") #

    print("The estimated expected reward for the initial state state is : "+str(sum(rewards)/N))

    #for tuple in monte_carlo(trajectories):
    #    print(tuple)