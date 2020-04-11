from game import Game
import random
from sklearn.ensemble import ExtraTreesRegressor
import numpy as np
import sys
from domain import Domain
import argparse
from section4 import GIFMaker
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pytorchTry import PQL
import torch


def generateWinningGame(p,s,policy, steps):

    game = Game(p, s, policy, steps)
    while not game.isWon:
        game = Game(p, s, policy, steps)
        game.playGameTillEnd()

    return game

def generateLosingGame(p,s,policy, steps):

    game = Game(p, s, policy, steps)
    game.playGameTillEnd()
    while game.isWon:
        game = Game(p, s, policy, steps)
        game.playGameTillEnd()

    return game


def generateTraj(toolbar_width, nb_of_games, all_win, mixed_games, save, policy, steps):

        fourTuple = []
        win = 0  # number of winning games

        print("\nGENERATING TRAJECTORIES:\n")
        # setup toolbar
        sys.stdout.write("[%s]" % (" " * toolbar_width))
        sys.stdout.flush()
        sys.stdout.write(
            "\b" * (toolbar_width + 1))  # return to start of line, after '['

        if not mixed_games:

            filename = str(nb_of_games) + '_games_all_win_' + str(all_win)

        else : filename = str(nb_of_games) + '_games_mixed_' + str(mixed_games)

        # if the desired number of won games has already been generated, no need to generate them anymore
        if os.path.isfile(filename):

            with open(filename, 'rb') as f:
                fourTuple = pickle.load(f)
        else:

            for i in range(nb_of_games):
                #print(" game " + str(i))
                p = random.uniform(-0.1, 0.1)
                s = 0

                if all_win :
                    game = generateWinningGame(p,s,policy, steps)

                else:
                    if mixed_games and i%2 :
                        game = generateWinningGame(p,s,policy, steps)
                    else:
                        game = Game(p, s, policy, steps)
                        game.playGameTillEnd()  # Ends game when it reaches final state

                # number of game won
                if game.isWon == True:
                    win = win + 1

                # append the tuples TODO  why not just do fourtuple = game.fullTrajectory ?
                for k in range(len(game.fullTrajectory)):
                    fourTuple.append(game.fullTrajectory[k])

                if (i + 1) % (nb_of_games / toolbar_width) == 0:
                    sys.stdout.write("-")
                    sys.stdout.flush()

            # saving the generated games for further use
            if save:
                with open(filename , 'wb') as fp:
                    pickle.dump(fourTuple, fp)

        sys.stdout.write("]\n")  #
        print("number of winning games : " + str(win))

        print("number of tuples : " + str(len(fourTuple)))

        return fourTuple


def setAndPlayGame(policy, steps, policy_PQL, fourTuple, nb_of_games, all_win, PATH, model):
    # creating the FQI game
    print("building the PQL model")
    p = random.uniform(-0.1, 0.1)
    game = Game(p,0,policy, steps)
    game.setToPQL(policy_PQL, fourTuple, PATH)

    print("playing last game")
    game.playGame()

    i = 0
    for tuple in game.trajectory:
        # print(tuple) VERBOSE
        i+=1

    print("game of " + str(i) + " moves with policy " + str(policy_PQL) +  " based on " + str(nb_of_games) )
    print("training  games were all won : " + str(all_win) )
    print(" game was won : " +  str(game.isWon))

    GIFMaker(game, policy_PQL, "PQL")
    return game



if __name__ == '__main__':


    # constants useful
    policy = "RAND"  # Game policy
    policy_PQL = "network" # SL algo used for building PQL
    steps = 500 # max number of steps of a game (will be ignored in this configuration)
    toolbar_width = 10
    nb_of_games = 150# working nb was 1000 but takes time
    all_win = True
    mixed_games =  not all_win
    save = True
    Train = False # train another network ?

    gamma = 0.95
    alpha = 0.0000001 # working value was 0.0000001
    model = torch.nn.Sequential()

    PATH = "modeltensor(999.5970, grad_fn=<MseLossBackward>)" # to modify if we want to load a model

    fourTuple = []  # (xt,ut,rt, xt+1)
    fourTuple = generateTraj(toolbar_width, nb_of_games, all_win, mixed_games,save, policy, steps)

    if Train:
        model, PATH = PQL(fourTuple, alpha, gamma, PATH)
    else:
        model = torch.load(PATH)
        model.eval()
    ###################################### TEST

    x = []
    y = []

    for tuple in fourTuple:
        ((p, s), action,(p2,s2), r) =  tuple
        x.append([p,s,action])
        y.append([r])


    x = torch.tensor(x)
    y = torch.tensor(y)

    for i in range(len(fourTuple)):
        #print(i)

        Q = x[i]
        r = model(torch.tensor(Q))

        print(" pred = " + str(r) + " , true val = " + str(y[i]))

    setAndPlayGame(policy, steps, policy_PQL, fourTuple, nb_of_games, all_win, PATH, model)
