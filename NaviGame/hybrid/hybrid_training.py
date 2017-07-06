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
            if choice == 5:
                choice = NaviStrategy.plan_movement(self)
        return choice

# constructs an MLP with inputs & outputs for different game modes, and whatever hidden layers you pass in with a dictionary
def baseline_model(optimizer = Adam(lr = 0.00001),
                    layers = [{"size":20,"activation":"relu"}]):
    return build_model(optimizer, layers, inputs = 1204, outputs = 6)

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

# this one places two walls on either side of the target, with a random length and width
def train_with_channel(model, episodes, steps, gamecount):
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
        length = randint(8, 14) * 2
        width = randint(1, 5) * 2
        step_1 = (1, 0)
        start_1 = (15 - int(0.5*length), 20 - int(0.5*width))
        start_2 = (15 - int(0.5*length), 20 + int(0.5*width))
        # ish
        game.add_wall(length = length, start = start_1, step = step_1)
        game.add_wall(length = length, start = start_2, step = step_1)

        outputs.append(train_model(game = game,
                model = model,
                episodes = episodes_per_game,
                steps = steps,
                e_start = .9,
                e_stop = .1))
        draw_game(game, save = True, filename = "training_game" + str(i) + ".png")
        del game
        model.save("wall_training_backup.h5")
    return outputs

# the next training function I write will need to randomly select from a variety of different obstacle situations, and train with them all.

if __name__=='__main__':
    # lets train a hybrid DQN model!
    # make the model
    print("If you are running this on a machine with GPU, and didn't use flags, abort now and restart with: \n")
    print("THEANO_FLAGS=device=gpu,floatX=float32 python this_file.py\n")
    print("But that's kinda a lie, cuz this code is a lil buggy and every time I try to do that on AWS it explodes. I don't own a machine with a GPU, so I've been running it on compute-optimized AWS nodes for long runs. That said, my best models were trained in under 2 hours on a 2016 MacBook.")
    print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ")
    # neurons = int(input("How many hidden layer neurons?\n"))
    neurons = 20
    hiddens = [{"size":neurons,"activation":"relu"}]
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
    training_game_size_x = 40
    training_game_size_y = 30

    training_game = HybridNaviGame(training_game_size_y,
                                    training_game_size_x,
                                    model,
                                    tolerance = 5)
    training_game.setup()
    training_game.Navigator.strategy.mode = 3

    training_episodes = int(input("How many episodes?\n"))
    steps = int(input("How many steps per episode?\n"))
    print("Ready to beging training")
    _ = input("Press enter to begin")
    # train the model
    output = train_model(game = training_game,
                    model = model,
                    episodes = training_episodes,
                    steps = steps,
                    e_start = .9,
                    e_stop = .1)

    # plot learning info
    title_str = str(training_game_size_y) + "x" + str(training_game_size_x) + " with "
    title_str += str(training_episodes) + " episodes, " + str(steps) + " steps per episode\n"
    # title_str += str(len(hiddens)) + " hidden layers, optimized with " +
    title_str += str(neurons) + " neurons in hidden layer, optimized with " + optimizer_str + "\n"
    f, axarr = pl.subplots(3, 1, figsize = (8, 10.5), dpi = 600)

    base = int(training_episodes/1000)
    for _, k in enumerate([base, 5*base, 25*base]):
        mean_step = k
        mean_rewards = []
        mean_dists = []
        mean_loss = []
        num_means = int(len(output['distances'])/mean_step/steps)
        steps_per_mean = steps*mean_step
        x = np.linspace(0, training_episodes, num_means)
        for i in range(num_means):
            mean_r = 0
            mean_d = 0
            mean_l = 0
            for j in range(steps_per_mean):
                mean_r += output['rewards'][j + i * steps_per_mean]
                mean_d += output['distances'][j + i * steps_per_mean]
                mean_l += output['loss'][j + i * steps_per_mean]
            mean_r = mean_r / steps_per_mean
            mean_d = mean_d / steps_per_mean
            mean_l = mean_l / steps_per_mean
            mean_rewards.append(mean_r)
            mean_dists.append(mean_d)
            mean_loss.append(mean_l)
        label = str(mean_step) + " Episodes"
        axarr[0].plot(x, mean_loss, label = label)
        axarr[1].plot(x, mean_dists, label = label)
        axarr[2].plot(x, mean_rewards, label = label)

    axarr[0].grid(True)
    axarr[0].set_title(title_str + 'Mean Loss')
    axarr[1].grid(True)
    axarr[1].set_title('Mean Distances from Goal')
    axarr[2].grid(True)
    axarr[2].set_title('Mean Rewards')
    f.subplots_adjust(hspace=0.2)

    file_str = str(training_game_size_y) + "x" + str(training_game_size_x) + "_"
    file_str += str(training_episodes) + "_" + str(steps) + "_" + str(neurons)
    #  str(len(hiddens))
    file_str += "_" + optimizer_str
    pl.legend()
    pl.plot()
    it_str = str(0)
    print("Saving trained model...")
    model.save("guided_rl_model_" + file_str + it_str + ".h5")
    print("Saving training plot")
    pl.savefig("hybrid_plots" + file_str + it_str + ".png")
