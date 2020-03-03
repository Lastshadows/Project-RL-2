from game import Game
import random 
import time
import sys

def monte_carlo(trajectories):
    stateActionPairs = []
    rewardHistory = []
    for trajectory in trajectories:
        for (pt,st),at,rt in trajectory:
            if [pt,st,at] not in stateActionPairs:
                stateActionPairs.append([pt,st,at])
                rewardHistory.append([(pt,st),at,1,rt])
            else:
                i = stateActionPairs.index([pt,st,at])
                stateActionPairs.pop(i)
                (pt,st),at,x,reward = rewardHistory.pop(i)
                reward = (x*reward + rt)/(x+1)
                x = x+1
                rewardHistory.append([(pt,st),at,x,reward])
                stateActionPairs.append([pt,st,at])

    return rewardHistory  


if __name__ == '__main__':

    policy = "ACC"
    steps = 20
    N = 10000
    trajectories = []

    toolbar_width = 50

    print("GENERATING TRAJECTORIES:")
    # setup toolbar
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

    for i in range(N):
        p = random.uniform(-0.1, 0.1)
        s = 0
        game = Game(p,s, policy, steps)
        game.playGame()
        trajectories.append(game.trajectory)

        if (i+1)%(N/toolbar_width)==0:
            sys.stdout.write("-")
            sys.stdout.flush()

    sys.stdout.write("]\n") #

    for tuple in monte_carlo(trajectories):
        print(tuple)