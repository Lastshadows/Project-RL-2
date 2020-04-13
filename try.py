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

def get_actions(fourTuple):
    actions = []
    for (pt,st),action,(pnext,snext),reward in fourTuple:
        if action == 4:
            actions.append("Right")
        elif action == -4:
            actions.append("Left")
    return actions

def build_dataset1(N):
    #Generates episodes with a Rand policy each of them ending in a terminal state
    #INPUT N : a number of episodes
    #OUTPUT fourTuple: full trajectories of the N episodes
    policy = "RAND"
    steps = 1000

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
    #Generates episodes with a Rand policy each of them ending in a terminal state 
    #with the same number of winning and losing episodes
    #INPUT N : a number of episodes
    #OUTPUT fourTuple: full trajectories of the N episodes
    policy = "RAND"
    steps = 1000

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
    #INPUT name: name of the wanted model
    #OUPUT model: wanted model
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

def FQI(name,N,Qprev,fourTuple,xtrain,ytrain,convergence = False):
    print("\nFIXED Fitted Q learning:\n")
    if convergence == False:
        nbQ=10
    else:
        nbQ = 100
    currStateAction1 = np.zeros((1,3)) #State and chosen action is Left
    currStateAction2 = np.zeros((1,3)) #State and chosen action is Right
    a = np.zeros((1,3)) 
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
        Qcurr = get_model(name)
        Qcurr = fit_model(name,Qcurr,xtrain,ytrain)
        
        if convergence != False and i >=3:
            error = 0
            s = 0
            action = [-4, 4]
            
            for k in range(1000):
                p = random.uniform(-0.7, 0.7)
                rand = random.randint(0,1)
                a[0][0] = p 
                a[0][1] = s
                a[0][2] = action[rand]
                previous = Qprev.predict(a)
                current = Qcurr.predict(a)
                error = error + (previous[0]-current[0])**2
            error = error/1000
            if error <= convergence: 
                print("current error : "+str(error))
                break
        Qprev = Qcurr

    print("FQI done iterating")
    return Qcurr

def fit_model(name,model,xtrain,ytrain):
    #INPUT name: name of the model wanted
    #      model: given model
    #      xtrain: X training set
    #      ytrain: Y training set
    #OUTPUT model : trained model    
    if name == "NN":
        model.fit(xtrain,ytrain,epochs=10, batch_size=5, shuffle=True, verbose=1)
    else :
        model.fit(xtrain,ytrain)
    return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='The default model is tree,default database is rand and the default stopping condition is fixed')
    parser.add_argument("--policy",type = str,
        help="chose a model between :tree (ExtraTreesRegressor) NN (Neural Network) line (linearRegressor) log (logisticRegressor)")
    parser.add_argument("--data",type = str,
        help="chose a dataset between : rand (random number of winning episodes) even (50% winning/losing episodes)")
    parser.add_argument("--convergence",type = float,
        help="chose the second stopping condition for FQN : threshold (the difference Qn and Qn-1 below a certain threshold)")
    
    parser.add_argument("--savegraph",type = str,
        help="save graph of final Fitted Q algo for every position p in namefile: namefile")
    args = parser.parse_args()

    N = 1000 #number of episodes

    if args.policy:
        name = args.policy
    else :
        name = "tree"

    if args.data:
        if args.data == "rand":
            fourTuple = build_dataset1(N)
        elif args.data =="even":
            fourTuple = build_dataset2(N)
        else:
            print("UNKNOWN database: "+str(args.data))
            sys.exit(0)
    else:
        fourTuple = build_dataset1(N)

    if args.convergence:
        convergence = args.convergence
    else:
        convergence = False

    #constants useful
    policy = "RAND" #Game policy
    steps = 1000 #number of steps not used here 
    gamma = 0.95 
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

    Qprev=fit_model(name,Q0,xtrain,ytrain)
    
    #PERFORM FQI    
    Qprev = FQI(name,N,Qprev,fourTuple,xtrain,ytrain,convergence) 
    
    #AVERAGE REWARD IN STARTING POSITION [-0.1,0.1]
    sumR = 0
    length = 0
    s=0
    maxlength = float("inf")
    sequence = []

    for i in range(10):
        p = random.uniform(-0.1, 0.1)
        game = Game(p,s, policy, steps)
        game.playGameGivenQ(Qprev) #Ends game when it reaches final state
        #print(game.reward)
        sumR = sumR + game.reward
        length = length + len(game.fullTrajectory)
        if len(game.fullTrajectory)<maxlength:
            sequence = game.fullTrajectory

    print("Average reward with p in [-0.1,0.1]: "+str(sumR/10))
    print("Average length :" + str(length/10))
    print("Best solution : "+ str(get_actions(sequence)))

    #PLOT BEST MOVE GIVEN p and s
    accelerate = np.zeros((1,3)) 
    decelerate = np.zeros((1,3))
    accelerate[0][2]=4
    decelerate[0][2]=-4
    plt.figure(figsize=(20,10))
        
    for i in range(0,21):
        p = i/10-1
        accelerate[0][0]=p
        decelerate[0][0]=p
        for j in range(0,61):
            s = j/10-3
            accelerate[0][1]=s
            decelerate[0][1]=s
            print("initial position: "+str(p)+" "+ str(s))
            acc = Qprev.predict(accelerate)
            dec = Qprev.predict(decelerate)
            print("acceleration : "+ str(acc[0])+" deceleration : "+str(dec[0]))

            if acc > dec:
                print("Best decision : accelerate\n")
                plt.scatter(p, s,c='green')
            elif acc == dec:
                print("Best decision : same\n")
                plt.scatter(p, s,c='black')
            else :
                print("Best decision : decelerate\n")
                plt.scatter(p, s,c='red')
    
    plt.ylabel("s")
    plt.xlabel("p")
    plt.savefig("policy"+str(name)+".png")
    

    # ESTIMATED reward by using learnt Q for every position
    print("The expected return value for this policy starting in :")
    pvect = np.zeros(200)
    t = np.arange(-1, 1.0, 0.01)
    s = 0
    for i in range(200):
        p = -1 + i/100
        sumreward = 0
        for j in range(10):
            game = Game(p,s, policy, steps)
            game.playGameGivenQ(Qprev) #Ends game when it reaches final state
            sumreward = sumreward + game.reward
        
        sumreward = sumreward/10
        print("p="+str(p)+" and s=0 : "+str(sumreward))
        sumR = sumR + sumreward
        pvect[i]=sumreward

    print(pvect)
    print("\nThe expected return value for this policy for any position : "+str(np.mean(pvect)))

    if args.savegraph:
        plt.figure(figsize=(20,10))
        plt.plot(t, pvect, lw=2)
        plt.ylabel("Reward")
        plt.xlabel("Initial position p")
        plt.savefig(args.savegraph)