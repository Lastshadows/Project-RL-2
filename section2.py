from game import Game
import random


if __name__ == '__main__':

    policy = "ACC"
    steps = 1000

    p = random.uniform(-0.1, 0.1)
    s = 0

    game = Game(p,s, policy, steps)
    game.playGame()

    print(" Section 2 : \n")
    print(" Printing the trajectory of the game (" + str(steps) + " steps) under the following format : \n")
    print("(position, speed), action, reward")

    for tuple in game.trajectory:
        print(tuple)