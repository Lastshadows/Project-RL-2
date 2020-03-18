from game import Game
import random
from sklearn.ensemble import ExtraTreesRegressor
import numpy as np
import sys
from domain import Domain
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='The default model is tree')
    parser.add_argument("--policy",type = str,
        help="chose a model between :tree (ExtraTreesRegressor) NN (Neural Network) line (linearRegressor) log (logisticRegressor)")
    args = parser.parse_args()

    if args.policy:
        if args.policy=="tree":
            model = 'TODO'
        elif args.policy=="NN":
            model = "TODO"
        elif args.policy == "line":
            model = "TODO"
        elif args.policy == "log":
            model = "TODO"
        else :
            print("UNKNOWN policy: "+args.policy)
            print("TRY : tree - NN - line - log")
            print("seek help by using --help")
            sys.exit(0)
    else :
        model = ExtraTreesRegressor(n_estimators= 50)

    #constants useful
    policy = "RAND" #Game policy
    steps = 1000 #number of steps not used here 
    gamma = 0.95 
    nbQ = 10 #Number of iteration of Q
    toolbar_width = 10
    j = 0 #number of winning games
    N = 100 #number of episodes should change to 1000 but takes to much time

    fourTuple = [] #(xt,ut,rt, xt+1)

    print("\nGENERATING TRAJECTORIES:\n")
    # setup toolbar
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

    for i in range(N):
        p = random.uniform(-0.1, 0.1)
        s = 0

        game = Game(p,s, policy, steps)
        game.playGameTillEnd() #Ends game when it reaches final state

        #number of game won
        if game.isWon == True:
            j = j+1

        #append the tuples
        for k in range(len(game.fullTrajectory)):
            fourTuple.append(game.fullTrajectory[k])

        if (i+1)%(N/toolbar_width)==0:
            sys.stdout.write("-")
            sys.stdout.flush()

    sys.stdout.write("]\n") #
    print("number of winning games : "+str(j))

    print("number of tuples : " + str(len(fourTuple)))

    Q0 = ExtraTreesRegressor(n_estimators= 50)

    xtrain = np.zeros((len(fourTuple),3))
    ytrain = np.zeros(len(fourTuple))
    j = 0

    #generate intial training set
    for (pt,st),action,(pnext,snext),reward in fourTuple:
        xtrain[j][0] = pt
        xtrain[j][1] = st
        xtrain[j][2] = action
        ytrain[j] = reward
        j=j+1

    Q0.fit(xtrain,ytrain)

    #Some tests to see if everything OK 
    a = np.ones((1,3))
    print(Q0.predict(a))   

    #Useful variables
    Qprev = Q0
    currStateAction1 = np.zeros((1,3)) #State and chosen action is Left
    currStateAction2 = np.zeros((1,3)) #State and chosen action is Right

    #Progress bar
    print("\nFitted Q learning:\n")
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

    #Fitted Q algorithm
    for i in range(nbQ):
        print(" iteration " + str(i))
        j = 0
        
        #Rebuild the training set
        for (pt,st),action,(pnext,snext),reward in fourTuple:

            currStateAction1[0][0]=pnext
            currStateAction1[0][1]=snext
            currStateAction1[0][2]=-4

            currStateAction2[0][0]=pnext
            currStateAction2[0][1]=snext
            currStateAction2[0][2]=4

            ytrain[j] = reward + gamma*max(Qprev.predict(currStateAction1),Qprev.predict(currStateAction2))
            j = j+1

        Qcurr = ExtraTreesRegressor(n_estimators= 50)
        Qcurr.fit(xtrain,ytrain)
        Qprev = Qcurr

        if (i+1)%(nbQ/toolbar_width)==0:
            sys.stdout.write("-")
            sys.stdout.flush()

    sys.stdout.write("]\n") #

    print("done iterating")

    #Chose a initial state (p,s)
    p = random.uniform(-0.1, 0.1)
    s = 0
    actionAll = []
    D = Domain()
    t = 0

    #Determine the path backwards
    while abs(p) < 1 and abs(s) < 3:
        currStateAction1[0][0]=p
        currStateAction1[0][1]=s
        currStateAction1[0][2]=-4

        currStateAction2[0][0]=p
        currStateAction2[0][1]=s
        currStateAction2[0][2]=4

        #print(p)

        if Qcurr.predict(currStateAction1) > Qcurr.predict(currStateAction2):
            action = -4
            actionAll.append("LEFT")
        else:
            action = 4
            actionAll.append("RIGHT")

        p,s,t = D.dynamics(p, s, action, t)

    print("done playing the final game")

    #Check the final position, the actions taken and if it is a winning state
    print("position: "+str(p) + " "+str(s))
    print("Is it a winning state ? " + str(D.rewardSignal(p, s)))
    print("Actions chosen : "+str(actionAll))