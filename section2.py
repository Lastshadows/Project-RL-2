from game import Game

if __name__ == '__main__':

    policy = "ACC"
    steps = 20

    game = Game(0,0, policy, steps)
    game.playGame()

    print(" Section 2 : \n")
    print(" Printing the trajectory of the game (" + str(steps) + " steps) under the following format : \n")
    print("(position, speed), action, reward")

    for tuple in game.trajectory:
        print(tuple)