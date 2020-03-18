from game import Game
import random
from sklearn.ensemble import ExtraTreesRegressor
import numpy as np
import sys
from domain import Domain
import argparse

def generateWinningGame(p,s,policy, steps):

    game = Game(p, s, policy, steps)
    while not game.isWon:
        game = Game(p, s, policy, steps)
        game.playGameTillEnd()

    return game

if __name__ == '__main__':


    # constants useful
    policy = "RAND"  # Game policy
    policy_FQI = "tree" # SL algo used for building FQI
    steps = 1000  # max number of steps of a game (will be ignored in this configuration)
    N_FQI = 10  # Number of iteration of Q
    toolbar_width = 10
    win = 0  # number of winning games
    nb_of_games = 100 # number of episodes should change to 1000 but takes to much time

    fourTuple = []  # (xt,ut,rt, xt+1)

    print("\nGENERATING TRAJECTORIES:\n")
    # setup toolbar
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write(
        "\b" * (toolbar_width + 1))  # return to start of line, after '['

    for i in range(nb_of_games):
        p = random.uniform(-0.1, 0.1)
        s = 0

        #game = Game(p, s, policy, steps)
        #game.playGameTillEnd()  # Ends game when it reaches final state

        game = generateWinningGame(p,s,policy, steps)

        # number of game won
        if game.isWon == True:
            win = win + 1

        # append the tuples TODO  why not just do fourtuple = game.fullTrajectory ?
        for k in range(len(game.fullTrajectory)):
            fourTuple.append(game.fullTrajectory[k])

        if (i + 1) % (nb_of_games / toolbar_width) == 0:
            sys.stdout.write("-")
            sys.stdout.flush()

    sys.stdout.write("]\n")  #
    print("number of winning games : " + str(win))

    print("number of tuples : " + str(len(fourTuple)))

    # creating the FQI game

    print("building the FQI model")
    p = random.uniform(-0.1, 0.1)
    game = Game(p,0,policy, steps)
    game.setToFQI(policy_FQI, fourTuple, N_FQI, nb_of_games)

    print("playing last game")
    game.playGame()

    i = 0
    for tuple in game.trajectory:
        print(tuple)
        i+=1

    print("game of " + str(i) + " moves")
