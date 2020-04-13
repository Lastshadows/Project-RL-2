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

def generateWinningGame(p,s,policy, steps):

    game = Game(p, s, policy, steps)
    while not game.isWon:
        game = Game(p, s, policy, steps)
        game.playGameTillEnd()

    return game


def generateTraj(toolbar_width, nb_of_games, all_win, save, policy, steps):

        fourTuple = []
        win = 0  # number of winning games

        print("\nGENERATING TRAJECTORIES:\n")
        # setup toolbar
        sys.stdout.write("[%s]" % (" " * toolbar_width))
        sys.stdout.flush()
        sys.stdout.write(
            "\b" * (toolbar_width + 1))  # return to start of line, after '['


        filename = str(nb_of_games) + '_games_all_win_' + str(all_win)

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


def setAndPlayGame(policy, steps, policy_FQI, fourTuple, N_FQI, nb_of_games, all_win):
    # creating the FQI game
    print("building the FQI model")
    p = random.uniform(-0.1, 0.1)
    game = Game(p,0,policy, steps)
    game.setToFQI(policy_FQI, fourTuple, N_FQI, nb_of_games)

    print("playing last game")
    game.playGame()

    i = 0
    for tuple in game.trajectory:
        # print(tuple) VERBOSE
        i+=1

    print("game of " + str(i) + " moves with policy " + str(policy_FQI) +  " based on " + str(nb_of_games) + " and " + str(N_FQI) + " steps")
    print("training  games were all won : " + str(all_win) )
    print("game won : " + str(game.isWon))

    GIFMaker(game, policy_FQI)
    return game



if __name__ == '__main__':


    # constants useful
    policy = "RAND"  # Game policy
    policy_FQI = "tree" # SL algo used for building FQI
    steps = 500 # max number of steps of a game (will be ignored in this configuration)
    N_FQI = 25 # Number of iteration of Q
    toolbar_width = 10
    nb_of_games = 300# number of episodes should change to 1000 but takes to much time
    all_win = False
    save = False

    fourTuple = []  # (xt,ut,rt, xt+1)
    fourTuple = generateTraj(toolbar_width, nb_of_games, all_win, save, policy, steps)
    setAndPlayGame(policy, steps, policy_FQI, fourTuple, N_FQI, nb_of_games, all_win)

    moves = []

    """
    # iterate over the trajectory
    for i in range(nb_of_games):

        # all 10 games
        if ((i % 10) == 0 and i != 0):
            print( "i = " + str(i))
            small_traj_size =  int(i/nb_of_games*len(fourTuple)) # we take a sub trajectory of a reducted size (i/nb_of_game % of the full trajectory)

            truncated_fourtTuple =  fourTuple[:small_traj_size]
            game = setAndPlayGame(policy, steps, policy_FQI, truncated_fourtTuple, N_FQI, nb_of_games, all_win)
            nb_of_moves = len(game.trajectory)
            moves.append(nb_of_moves)
            print("game won : " + str(game.isWon))

    y_pos = np.arange(len(moves))

    # Create bars
    plt.bar(y_pos, moves)

    # Create names on the x-axis
    #plt.xticks(y_pos, bars)

    # Show graphic
    plt.show()
    """
