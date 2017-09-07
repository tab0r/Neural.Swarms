# utilities
import numpy as np
import pylab as pl
import pickle, os, sys, pdb
from tqdm import *
from random import randint
sys.path.append(os.path.abspath("../reinforcement/"))
sys.path.append(os.path.abspath("../../../Python.Swarms/"))

# game imports
from navi_game import NaviStrategy
from ReinforcementNaviGame import ReinforcementNaviGame, ReinforcementStrategy
from game_display_helper import make_gif

# imports for neural net
from keras.models import Sequential, load_model
from keras.layers.core import Dense
from keras.optimizers import sgd, RMSprop, Adagrad, Adadelta, Adam
import theano
from reinforcement_training import build_model, train_model, plot_learning_info

class HybridNaviGame(ReinforcementNaviGame):
    def setup(self):
        ReinforcementNaviGame.setup(self)
        self.strategy = HybridStrategy(
                            goal = self.goal,
                            model = self.model,
                            tolerance = self.tolerance,
                            idle_t = self.goal_idle)
        self.Navigator.bindStrategy(self.strategy)

# Hybrid Learning strategy - experimental
# works great! Can train a model that handles basic obstacles in 30k steps
class HybridStrategy(ReinforcementStrategy):
    def plan_movement(self, e = 0.01, position = None):
        d = np.random.random()
        # explore/learn e% of the time
        if d < e:
            if d < e/2:
                choice = randint(0, 4)
            else:
                choice = NaviStrategy.plan_movement(self)
        # exploit current Q-function
        else:
            _, quality = self.get_quality()
            choice = np.argmax(quality)
            # this would be an actual hybrid strategy
            # re-implement later
            # if choice == 5:
            #     choice = NaviStrategy.plan_movement(self)
        return choice

# constructs an MLP with inputs & outputs for different game modes, and whatever hidden layers you pass in with a dictionary
def baseline_model(optimizer = Adam(lr = 0.00001),
                    layers = [{"size":20,"activation":"relu"}]):
    return build_model(optimizer, layers, inputs = 4, outputs = 5)

# these methods construct new training games, and pass them to train_model with the model you pass in.
# lots of improvements to be made here... including some I made and accidentally deleted.... so that's a thing.
# this one places random blocks into the game
def train_with_blocks(model, episodes, steps, gamecount, blockcount):
    training_game_size_x = 40
    training_game_size_y = 30
    episodes_per_game = int(episodes/gamecount)
    outputs = []
    for i in range(gamecount):
        game = HybridNaviGame(training_game_size_y,
                                training_game_size_x,
                                model,
                                tolerance = 3)
        game.setup()
        game.Navigator.strategy.mode = 3
        blocks = []
        for _ in range(blockcount):
            blocks.append(game.add_block())
        outputs.append(train_model(game = game,
                model = model,
                episodes = episodes_per_game,
                steps = steps,
                e_start = .9,
                e_stop = .1))
        draw_game(game, save = True, filename = "training_game" + str(i) + ".png")
        del game
        model.save("block_training_backup.h5")
    return outputs

# the next training function I write will need to randomly select from a variety of different obstacle situations, and train with them all.

if __name__=='__main__':
    # lets train a hybrid DQN model!
    # make the model
    neurons = 20
    hiddens = [{"size":neurons,"activation":"relu"},
                {"size":neurons,"activation":"relu"}]
    # the baseline_model function takes a dictionary of hidden layers,
    # and sets up your input/output layers for the game
    # [{"size":100,"activation":"relu"}, {"size":100,"activation":"relu"}]
    # make an optimizer
    from keras.optimizers import sgd, RMSprop, Adagrad, Adadelta, Adam
    # note: DON'T CHANGE THIS UNTIL YOU KNOW YOUR MODEL LEARNS SOMETHING
    # optimizer = sgd(lr = 0.0001)
    # optimizer_str = "SGD"
    # optimizer = Adagrad()
    # optimizer_str = "Adagrad"
    # optimizer = RMSprop()
    # optimizer_str = "RMSprop"
    # optimizer = Adadelta()
    # optimizer_str = "Adadelta"
    # seriously, Adam is magical, I don't really understand it but just use it
    optimizer = Adam()
    optimizer_str = "Adam"
    model = baseline_model()
    # this probably won't work
    # model = load_model("guided_rl_model_wide.h5")
    # this probably will work
    # model.load_weights("your model")

    # set up the training game
    training_game_size_x = 19
    training_game_size_y = 13

    training_game = HybridNaviGame(training_game_size_y,
                                    training_game_size_x,
                                    model,
                                    tolerance = 2)
    training_game.setup()
    # Strategy mode 1 is coordinates only
    # Mode 2 is pixel input
    # Mode 3 is pix + coords
    training_game.Navigator.strategy.mode = 1

    training_episodes = int(input("How many episodes?\n"))
    steps = 15 #int(input("How many steps per episode?\n"))
    print("Ready to beging training")
    _ = input("Press enter to begin")
    # train the model
    output = train_model(game = training_game,
                    model = model,
                    episodes = training_episodes,
                    steps = steps,
                    e_start = .9,
                    e_stop = .1)

    file_str = str(training_game_size_y) + "x" + str(training_game_size_x) + "_"
    file_str += str(training_episodes) + "_" + str(steps) + "_" + str(neurons)
    #  str(len(hiddens))
    file_str += "_" + optimizer_str
    # pl.legend()
    # pl.plot()
    it_str = str(0)
    print("Saving trained model...")
    model.save("guided_rl_model_" + file_str + it_str + ".h5")
    # print("Saving training plot")
    # pl.savefig("hybrid_plots" + file_str + it_str + ".png")
    make_gif(training_game, 100)
