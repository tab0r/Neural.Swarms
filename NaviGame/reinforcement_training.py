# utilities
import pandas as pd
import numpy as np
import pylab as pl
import pickle, os, sys, pdb
from collections import Counter
from tqdm import *
from notebook_game_helper import draw_game, animate_game
sys.path.append(os.path.abspath("../../Python.Swarms/"))

# game imports
from random import random, sample, randint
from game import BoardGame
from board import Board
from figure import Figure, FigureStrategy
from logger import log
from navi_game import *

# imports for neural net
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers.core import Dense
from keras.optimizers import sgd, RMSprop, Adagrad, Adadelta, Adam
import theano

# Navigator game main class
class ReinforcementNaviGame(NaviGame):
    def __init__(self, height, width, model, tolerance = 2, goal_idle = 1):
        NaviGame.__init__(self, height, width,
                            goal = (int(height/2), int(width/2)),
                            moving_target = False,
                            tolerance = tolerance,
                            goal_idle = goal_idle)
        self.model = model
        self.display_str = "Game start"

    def setup(self):
        NaviGame.setup(self)
        self.strategy = ReinforcementStrategy(
                            self.goal,
                            self.model,
                            tolerance = self.tolerance,
                            idle_t = self.goal_idle)
        self.Navigator.bindStrategy(self.strategy)

    def step(self):
        NaviGame.step(self)
        self.display_str = ""
        self.display_str += "Last Reward: "
        reward = self.Navigator.strategy.last_reward
        self.display_str += "{0:.2f}".format(reward)
        distance = self.Navigator.strategy.get_distance(self.Navigator.position(), self.goal)
        self.display_str += " Distance: " + "{0:.2f}".format(distance)

# Reinforcement Learning strategy - sketchy af but kinda works
class ReinforcementStrategy(NaviStrategy):
    def __init__(self, goal, model, tolerance, idle_t):
        # Deep-Q network
        self.model = model
        self.last_reward = 0
        self.idle_t = idle_t
        self.dynamic_reward = True
        self.mode = 0
        NaviStrategy.__init__(self, goal, tolerance)

    def plan_movement(self, e = 0.05, position = None):
        d = np.random.random()
        # explore 5% of the time
        if d < e:
            choice = randint(0, 4)
        # exploit current Q-function
        else:
            _, quality = self.get_quality(mode = 2)
            choice = np.argmax(quality)
        return choice

    def get_quality(self, mode = None):
        if mode == None:
            mode = self.mode
        # fixed goal mode
        if mode == 0:
            ipt = list(self.figure.position())
            n = 2
        # moving goal mode
        elif mode == 1:
            ipt, _ = self.get_input()
            n = 4
        # pixel input mode
        elif mode == 2:
            s = np.array(self.board.cells)
            color = lambda i: 0 if i == None else i.color
            colormask = np.vectorize(color)
            ipt = colormask(s)
            n = self.board.width * self.board.height
        # combined input mode
        elif mode == 3:
            ipt, _ = self.get_input()
            cells = self.board.cells
            flatten = lambda l: [item for sublist in l for item in sublist]
            cells = flatten(cells)
            color = lambda i: 0 if i == None else i.color
            colormask = [color(i) for i in cells]
            ipt.extend(colormask)
            n = 4 + (self.board.width * self.board.height)
        if mode == 2:
            ipt = np.array(ipt)
        else:
            ipt = np.array(ipt).reshape(1, n)
        quality = self.model.predict(ipt)
        return ipt, quality

    def get_reward(self, step = -1, goal = 1):
        if self.dynamic_reward == True:
            distance = self.get_distance(self.figure.position(), self.goal)
            distance = np.min([distance, 2*self.tolerance])
            reward = goal * (1 - ((distance - 1)/self.tolerance))
            # now the reward is in the range (-goal, goal)
        else:
            if self.at_goal > self.idle_t:
                reward = goal
                self.wins += 1
            else:
                reward = step
        return reward

    def step(self, choice = None):
        self.last_reward = self.get_reward()
        NaviStrategy.step(self, choice)

    def shift(self, pos=None):
        # just to reset position for each game
        if pos == None:
            pos = (randint(0, self.board.height-1), randint(0, self.board.width-1))
        try:
            self.figure.move(y=pos[0], x=pos[1], relative=False)
        except:
            self.shift()

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

def baseline_model(optimizer = Adam(lr = 0.00001),
                    layers = [{"size":20,"activation":"relu"}],
                    ipt_mode = 0, opt_mode = 0):
    # four inputs - each coordinate when we move the goal
    # two inputs for now, until it's doing reallly good at fixed goals
    # and now 81 inputs, for a pixel value test!
    # now we get 1204 - 1200 pixels for a 40x30 board, plus 4 inputs
    if ipt_mode == 0:
        inputs = 2
    elif ipt_mode == 1:
        inputs = 4
    elif ipt_mode == 2:
        inputs = (40, 30)
    elif ipt_mode == 3:
        inputs = 1204
    # inputs = 1204
    # six outputs - one for each action, and one to use det strategy
    if opt_mode == 1:
        num_outputs = 6
    else:
        num_outputs = 5
    # prepare the navigator model
    model = Sequential()
    # initial inputs
    l = list(layers)
    l0 = l[0]
    del l[0]
    model.add(Dense(l0['size'],
                input_dim = inputs,
                activation = l0['activation']))
    # add convolutions if mode == 2
    if ipt_mode == 2:
        model.add(Convolution2D(64, 3, 3, subsample=(1,1),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
        model.add(Activation('relu'))
    # the hidden layers
    for layer in l:
        model.add(Dense(layer['size'], activation=layer['activation']))
    # the output layer
    model.add(Dense(num_outputs, activation='tanh'))
    model.compile(optimizer = optimizer,
                    loss = "mean_squared_error")
    return model

def train_model(game, model, episodes = 10, steps = 2,
                            e_start = 1, e_stop = 0):
    initial_e, final_e = e_start, e_stop
    # use 1 until we add stochasticity to the model
    gamma = 1
    e_delta = initial_e - final_e / episodes
    distances = []
    loss_log, replay_log, inputs, targets, rewards = [],[],[],[],[]
    desc = "Network training starting"
    # set up tqdm progress bar to display loss dynamically
    t = trange(episodes, desc=desc, leave=True)
    for j in t:
        # set up variables for episode
        e = initial_e - j * e_delta
        r_total_ep = 0
        # play through episode
        for i in range(steps):
            # feedforward pass which defines the next state
            # note that the e is an epsilon-greedy value,
            # which decreases as training carries on
            # choice = game.Navigator.strategy.plan_movement(e)
            input_i, quality_pred = game.Navigator.strategy.get_quality()
            # save network input data from above, which is our <s>
            _, dist = game.Navigator.strategy.get_input()
            distances.append(dist)

            # move to s'
            game.step()

            # update our Q[s,a] using the reward we get and
            # the quality prediction for our new state
            reward = game.Navigator.strategy.get_reward()
            choice = game.Navigator.strategy.last_choice
            quality = reward
            if dist > game.tolerance:
                quality += gamma * quality_pred.flatten()[choice]
            target_i = quality_pred
            n = len(quality_pred.flatten())
            target_i[0][choice] = quality
            target_i = target_i.reshape(1, n)
            # online learning
            loss = model.train_on_batch(input_i, target_i)

            # store data for experience replay learning
            inputs.append(input_i)
            targets.append(target_i)
            rewards.append(reward)
            loss_log.append(loss)

        # experience replays on a set random experience
        # they are chosen randomly
        for i in range(np.min([len(inputs), steps])):
            if len(inputs) == 1:
                experience = 0
            else:
                # lower_limit = int(0.75 * len(inputs))
                # experience = randint(lower_limit, max(1, len(inputs)-1))
                experience = randint(0, len(inputs)-1)
            replay_i = inputs[experience]
            replay_t = targets[experience]
            loss_replay = model.train_on_batch(replay_i, replay_t).flatten()[0]
            # replay_log.append(loss_replay)
        loss_str = '{0:.3g}'.format(loss_replay)
        desc = "Episode " + str(j) + ", Replay Loss: " + loss_str
        t.set_description(desc)
        t.refresh() # to update progress bar
        # shift goal and set last_reward to 0 so next episode is a "fresh game"
        # game.shift_goal()
        game.Navigator.strategy.shift()
        game.Navigator.strategy.last_reward = 0
    # housekeeping to return everything nicely
    output = dict()
    output['loss'] = loss_log
    # output['replays'] = replay_log
    output['experiences'] = [inputs, targets]
    output['distances'] = distances
    output['rewards'] = rewards
    return output

def grid_search_sketch():
    training_game_size = 9
    training_episodes = 1000
    steps = 10
    final_mean_loss, final_mean_dists, final_mean_reward = [], [], []
    for neurons in range(60, 120, 10):
        for layers in [2, 3, 4, 5, 10]:
            # lets train a DQN model!
            # make the model
            layer = {"size":neurons,"activation":"relu"}
            hiddens = [layer for i in range(layers)]    # make an optimizer
            from keras.optimizers import sgd, RMSprop, Adagrad, Adadelta, Adam
            # note to self: DON'T CHANGE THIS UNTIL YOU KNOW WE'RE LEARNING SOMETHING
            # optimizer = sgd(lr = 0.0001)
            # optimizer_str = "SGD"
            # optimizer = Adagrad()
            # optimizer_str = "Adagrad"
            # optimizer = RMSprop()
            # optimizer_str = "RMSprop"
            optimizer = Adadelta()
            optimizer_str = "Adadelta"
            # optimizer = Adam()
            # optimizer_str = "Adam"
            model = baseline_model(optimizer, hiddens)
            # prepare the game for training model
            training_game = ReinforcementNaviGame(training_game_size,
                                            training_game_size,
                                            model)
            training_game.setup()

            output = train_model(game = training_game,
                            model = model,
                            episodes = training_episodes,
                            steps = steps)

            # plot learning info
            title_str = str(training_game_size) + "x" + str(training_game_size) + " with "
            title_str += str(training_episodes) + " episodes, " + str(steps) + " steps per episode, & "
            title_str += str(len(hiddens)) + " hidden layers, optimized with " + optimizer_str + "\n"
            f, axarr = pl.subplots(3, 1, figsize = (10, 15), dpi = 300)
            # f.canvas.set_window_title("RL Loss, 100 eps w/ 50 steps, Look: 20")
            mean_step = 10
            num_means = int(len(output['distances'])/mean_step/steps)

            for _, k in enumerate([5, 10, 100]):
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

            # axarr[1].plot(x, output['replays'])
            # axarr[1].set_title('Replay Loss')
            # axarr[2].plot(x2, output['reward_totals'])
            # axarr[2].set_title('Total Reward')
            # axarr[2].plot(x2, output['distances'])
            # axarr[2].set_title('Distance from Goal')

            file_str = str(training_game_size) + "x" + str(training_game_size) + "_"
            file_str += str(training_episodes) + "_" + str(steps) + "_" + str(len(hiddens))
            file_str += "_" + str(neurons) + "_neurons_"+ optimizer_str
            pl.legend()
            pl.plot()
            pl.savefig("rl_plots" + file_str + ".png")
            final_mean_reward.append(mean_rewards.pop())
            final_mean_dists.append(mean_dists.pop())
            final_mean_loss.append(mean_loss.pop())
        # pl.show()
    f, axarr = pl.subplots(1, 1, figsize = (10, 5), dpi = 300)
    x = np.linspace(0, layers, layers)
    axarr.plot(x, final_mean_reward, label = "Final mean rewards")
    axarr.plot(x, final_mean_dists, label = "Final mean distances")
    axarr.plot(x, final_mean_loss, label = "Final mean loss")
    axarr.set_title("Network Grid Test")
    pl.legend()
    pl.savefig("rl_plots_network_test.png")
    pl.close()

if __name__=='__main__':
    # lets train a DQN model!
    # make the model
    print("If you are running this on a machine with GPU, and didn't use flags, abort now and restart with: \n")
    print("THEANO_FLAGS=device=gpu,floatX=float32 python this_file.py")
    
    hiddens = [{"size":100,"activation":"relu"},
               {"size":20,"activation":"relu"}]
    #            {"size":100,"activation":"relu"},
    #           {"size":100,"activation":"relu"}]
    # make an optimizer
    from keras.optimizers import sgd, RMSprop, Adagrad, Adadelta, Adam
    # note to self: DON'T CHANGE THIS UNTIL YOU KNOW WE'RE LEARNING SOMETHING
    # optimizer = sgd(lr = 0.0001)
    # optimizer_str = "SGD"
    # optimizer = Adagrad()
    # optimizer_str = "Adagrad"
    # optimizer = RMSprop()
    # optimizer_str = "RMSprop"
    # optimizer = Adadelta()
    # optimizer_str = "Adadelta"
    optimizer = Adam()
    optimizer_str = "Adam"
    # model = baseline_model(optimizer, hiddens, ipt_mode = 3, opt_mode = 1)
    model = load_model("guided_rl_model_wide.h5")

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

    print("Saving trained model...")
    model.save("guided_rl_model_wide.h5")
    # plot learning info
    title_str = str(training_game_size_y) + "x" + str(training_game_size_x) + " with "
    title_str += str(training_episodes) + " episodes, " + str(steps) + " steps per episode, & "
    title_str += str(len(hiddens)) + " hidden layers, optimized with " + optimizer_str + "\n"
    f, axarr = pl.subplots(3, 1, figsize = (10, 15), dpi = 300)

    base = int(episodes/100.0)
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
    file_str += str(training_episodes) + "_" + str(steps) + "_" + str(len(hiddens))
    file_str += "_" + optimizer_str
    pl.legend()
    pl.plot()
    pl.savefig("hybrid_plots" + file_str + "_2.png")
