from game import Game
import random
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import sys
from domain import Domain
import argparse
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense

def build_dataset1(N):
    policy = "RAND"
    steps = 1000
    gamma = 0.95
    j = 0
    fourTuple = []
    print("BUILDING DATASET 1")
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

    print("DATASET BUILD")
    print("number of winning games : "+str(j))
    print("number of tuples : " + str(len(fourTuple)))
    return fourTuple

def build_dataset2(N):
    policy = "RAND"
    steps = 1000
    gamma = 0.95
    print("BUILDING DATASET 2")
    fourTuple = []
    nb_win = 0
    nb_loss = 0
    while nb_win< N/2 or nb_loss < N/2:
        p = random.uniform(-0.1, 0.1)
        s = 0

        game = Game(p,s, policy, steps)
        game.playGameTillEnd() #Ends game when it reaches final state

        #number of game won

        if game.isWon == True and nb_win<N/2:
            #append the tuples
            for k in range(len(game.fullTrajectory)):
                fourTuple.append(game.fullTrajectory[k])
            nb_win = nb_win +1

        if game.isWon == False and nb_loss<N/2:
            #append the tuples
            for k in range(len(game.fullTrajectory)):
                fourTuple.append(game.fullTrajectory[k])
            nb_loss = nb_loss +1

    print("DATASET BUILD")
    print("number of winning games : "+str(N/2))
    print("number of tuples : " + str(len(fourTuple)))
    return fourTuple

def get_model(name):
    if name=="tree":
        model = ExtraTreesRegressor(n_estimators= 50)
    elif name=="line":
        model = LinearRegression()
    elif name =="NN":
        model = Sequential()
        model.add(Dense(200, input_dim=3))
        model.add(Dense(1))
        model.compile(optimizer="SGD", loss='mean_squared_error')
    return model

def FQI(name,N,Qprev,fourTuple,xtrain,ytrain,convergence = None):
    print("\nFIXED Fitted Q learning:\n")

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

        Qcurr = get_model("NN")
        Qcurr.fit(xtrain,ytrain,epochs=10, batch_size=5, shuffle=True, verbose=1)
        Qprev = Qcurr
        if convergence != None:
            for i in range(10):
                p = random.uniform(-0.7, 0.7)
                game = Game(p,s, policy, steps)
                game.playGameGivenQ(Qprev) #Ends game when it reaches final state
                print(game.reward)
                sumR = sumR + game.reward
                length = length + len(game.fullTrajectory)


    print("FQI done iterating")
    return Qcurr

def fit_model(name,model,xtrain,ytrain):
    if name == "NN":
        model.fit(xtrain,ytrain,epochs=10, batch_size=5, shuffle=True, verbose=1)
    else
        model.fit(xtrain,ytrain)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='The default model is tree,default database is rand and the default stopping condition is fixed')
    parser.add_argument("--policy",type = str,
        help="chose a model between :tree (ExtraTreesRegressor) NN (Neural Network) line (linearRegressor) log (logisticRegressor)")
    parser.add_argument("--data",type = str,
        help="chose a dataset between : rand (random number of winning episodes) even (50% winning/losing episodes)")
    parser.add_argument("--stop",type = str,
        help="chose a stopping condition for FQN between : convergence(the difference Qn and Qn-1 below a certain threshold) or fixed (10 iterations)")
    args = parser.parse_args()

    if args.policy:
        name = args.policy
    else :
        name = "tree"

    if args.data:
        if args.data == "rand":
            fourTuple = build_dataset1(N)
        elif args.data =="even":
            fourTuple = build_dataset2(N)
        else
            print("UNKNOWN database: "+str(args.data))
            sys.exit(0)
    else
        fourTuple = build_dataset1(N)

    #constants useful
    policy = "RAND" #Game policy
    steps = 1000 #number of steps not used here
    gamma = 0.95
    nbQ = 10 #Number of iteration of Q
    toolbar_width = 10
    j = 0 #number of winning games
    N = 1000 #number of episodes should change to 1000 but takes to much time

    Q0 = get_model(name)

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

    fit_model(name,Q0,xtrain,ytrain)
    #Some tests to see if everything OK
    a = np.ones((1,3))
    print(Q0.predict(a))

    #Useful variables
    Qprev = Q0
    currStateAction1 = np.zeros((1,3)) #State and chosen action is Left
    currStateAction2 = np.zeros((1,3)) #State and chosen action is Right

    #Progress bar

    Qcurr = fixed_FQI()
    #Chose a initial state (p,s)
    sumR = 0
    length = 0
    for i in range(10):
        p = random.uniform(-0.1, 0.1)
        game = Game(p,s, policy, steps)
        game.playGameGivenQ(Qprev) #Ends game when it reaches final state
        print(game.reward)
        sumR = sumR + game.reward
        length = length + len(game.fullTrajectory)

    print("the sum of all: "+str(sumR))
    print("length of all :" + str(length))

    pvect = np.zeros(200)
    t = np.arange(-1, 1.0, 0.01)
    print(t.shape)

    for i in range(200):
        p = -1 + i/100
        sumreward = 0
        for j in range(10):
            game = Game(p,s, policy, steps)
            game.playGameGivenQ(Qprev) #Ends game when it reaches final state
            sumreward = sumreward + game.reward
        #print(game.reward)
        sumR = sumR + game.reward
        length = length + len(game.fullTrajectory)
        print(sumreward/10)
        pvect[i]=sumreward/10
        #print("pvect "+ str(pvect))

    print(pvect)
    print("the sum of all: "+str(sumR))
    print("length of all :" + str(length))

    plt.figure(figsize=(20,10))
    plt.plot(t, pvect, lw=2)
    plt.ylabel("Reward")
    plt.xlabel("Initial position p")
    plt.savefig('myfig.png')

    """
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
    """
