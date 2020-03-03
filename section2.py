from game import Game

if __name__ == '__main__':

    policy = "ACC"
    steps = 20

    game = Game(0,0, policy, steps)
    game.playGame()

    for tuple in game.trajectory:
        print(tuple)