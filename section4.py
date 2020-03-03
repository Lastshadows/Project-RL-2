from displayer import save_caronthehill_image
from game import Game
import imageio

# Execution example
if __name__ == "__main__":

    policy = "ACC"
    steps = 100

    # play a game
    game = Game(0, 0, policy, steps)
    game.playGame()

    p,s = (0,0)
    i = 0
    images = []

    # build the gif from the images generated
    for tuple in game.trajectory:
        state, u, r = tuple
        p,s = state
        filename = "GIF/carTraj" + str(i) + ".png"
        # save the image of a given state
        save_caronthehill_image(p, s, out_file=filename)
        i+=1
        # add the image to the gif
        images.append(imageio.imread(filename))

    filename = "GIF/carTraj" + str(i) + ".png"
    a = save_caronthehill_image(p, s, out_file=filename, close=True)
    images.append(imageio.imread(filename))

    # save the gif 
    imageio.mimsave('trajectory.gif', images)
