from game import Game
from FQI import FittedQItLearner
import random
import pickle
import os.path

if __name__ == "__main__":


    policy = "RAND"
    policy_FQI = "tree"
    steps = 100000000
    FQI_steps = 2000
    nb_of_games = 100
    concatenated_traj = []
    N_FQI = 40

    filename = str(nb_of_games) + 'games_half_win'

    # if the desired number of won games has already been generated, no need to generate them anymore
    if os.path.isfile(filename):

        with open(filename, 'rb') as f:
            ...
            concatenated_traj = pickle.load(f)

    # if the desired number of winned games hasn't been generated yet, generate them and store them
    else:
        print("playing the games")
        i = 0
        losing_allowed = False

        while i < nb_of_games:

            # play a game
            p = random.uniform(-0.1, 0.1)
            game = Game(p, 0, policy, steps)
            game.playGame()

            if game.isWon and not losing_allowed:
                i+=1
                print(str(i))

                # build the trajectory
                for tuple in game.trajectory:
                    concatenated_traj.append(tuple)
                    x,u,r = tuple

                # allow the addition of a lost game next time
                losing_allowed = True

            # if we just added a winning game, we can add another losing case
            elif losing_allowed and not game.isWon:
                i += 1
                print(str(i))

                for tuple in game.trajectory:
                    concatenated_traj.append(tuple)
                    x, u, r = tuple

                # next added game need to be winned
                losing_allowed = False

            # saving the generated games for further use
            with open(filename , 'wb') as fp:
                pickle.dump(concatenated_traj, fp)



    print("building the FQI model")
    p = random.uniform(-0.1, 0.1)
    game = Game(p,0,policy, FQI_steps)
    game.setToFQI(policy_FQI, concatenated_traj, N_FQI, nb_of_games)

    print("playing last game")
    game.playGame()

    i = 0
    for tuple in game.trajectory:
        print(tuple)
        i+=1

    print("game of " + str(i) + " moves")
