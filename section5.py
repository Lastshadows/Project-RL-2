from displayer import save_caronthehill_image
import imageio
import os
from game import Game
from FQI import FittedQItLearner

if __name__ == "__main__":

    policy = "RAND"
    policy_FQI = "linear"
    steps = 10
    nb_of_games = 100
    concatenated_traj = []
    N_iterations = 10
    N_FQI = 10
    model_type = 'linear'

    for i in range(nb_of_games):

        # play a game
        game = Game(0, 0, policy, steps)
        game.playGame()

        # build the trajectory
        for tuple in game.trajectory:
            concatenated_traj.append(tuple)

    game = Game(0,0,policy, steps)
    game.setToFQI(policy_FQI, concatenated_traj, N_FQI)
    game.playGame()

    i = 0
    images = []

    # creation of the temporary folder where we will put the images
    newpath = r'GIF'
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    # build the gif from the images generated
    for tuple in game.trajectory:
        state, u, r = tuple
        p, s = state
        filename = "GIF/carTraj" + str(i) + ".png"

        # save the image of a given state
        save_caronthehill_image(p, s, out_file=filename)

        # add the image to the gif
        images.append(imageio.imread(filename))

        # immediately delete the image
        os.remove(filename)

        i += 1

    # save the last image
    filename = "GIF/carTraj" + str(i) + ".png"
    a = save_caronthehill_image(p, s, out_file=filename, close=True)

    # add the image to the gif
    images.append(imageio.imread(filename))

    # immediately delete the image
    os.remove(filename)

    # delete the folder
    os.rmdir(newpath)

    # save the gif
    imageio.mimsave('trajectory' + policy_FQI + '.gif', images)