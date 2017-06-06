# utilities
import pandas as pd
import numpy as np
import pylab as pl
import pickle
import os
import pdb
from collections import Counter
from tqdm import *
from notebook_game_helper import draw_game, animate_game

# import the game
from neural_navi_game import *

# imports for neural net
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd, RMSprop, Adagrad
import theano

def baseline_model(optimizer = sgd(lr = 0.0001),
                    layers = [{"size":20,"activation":"relu"}]):
    # five inputs - one for each coordinate, and last reward
    # one output - returns the predicted reward for a next state
    num_outputs = 1
    # prepare the navigator model
    model = Sequential()
    # initial inputs
    l = list(layers)
    l0 = l[0]
    del l[0]
    model.add(Dense(l0['size'],
                    input_dim = 5,
                    activation = l0['activation']))
    # the hidden layers
    for layer in l:
        model.add(Dense(layer['size'], activation=layer['activation']))
    # the output layer
    model.add(Dense(num_outputs, activation='relu'))
    model.compile(optimizer = optimizer,
                    loss = "mean_squared_error")
    return model

def train_model(game, model, episodes = 10, steps = 50):
    log, replay_log = [], []
    wins = 0
    desc = "Network training starting"
    # set up tqdm progress bar to display loss dynamically
    replay_inputs = []
    replay_targets = []
    t = trange(episodes, desc=desc, leave=True)
    for j in t:
        inputs, targets = [], []
        loss = 0
        for i in range(steps):
            position = game.Navigator.position()
            choice = game.Navigator.strategy.plan_movement()
            action = game.Navigator.strategy.actions[choice]
            next_pos = (position[0]+action[0], position[1]+action[1])
            input_i = game.Navigator.strategy.get_input()
            target_i = game.Navigator.strategy.last_reward \
                + game.Navigator.strategy.predict_quality(position = next_pos)
            # online learning
            loss += model.train_on_batch(np.array(input_i).reshape(1, 5),
            np.array([[target_i]]))
            # experience replay learning
            inputs.append(input_i)
            targets.append(target_i)
            if game.Navigator.strategy.at_goal == True:
                wins += 1
            game.step()
        replay_inputs.append(inputs)
        replay_targets.append(targets)
        # pdb.set_trace()
        # experience replay on a random episode
        epi = randint(0, len(replay_inputs) - 1)
        loss_replay = model.train_on_batch(replay_inputs[epi],
                                    replay_targets[epi]).flatten()[0]
        log.append(loss)
        replay_log.append(loss_replay)
        desc = "Episode " + str(j) + ", Wins: " + str(wins) \
                + ", Replay Loss: " + str(loss_replay)
        t.set_description(desc)
        t.refresh() # to update progress bar
        # shift goal and set last_reward to 0 so next episode is a "fresh game"
        game.shift_goal()
        game.Navigator.strategy.last_reward = 0
    return log, replay_log

if __name__=='__main__':
    training_game_size = 10
    training_episodes = 10
    steps = 100
    # make the model
    hiddens = [{"size":20,"activation":"relu"},
                {"size":20,"activation":"relu"},
                {"size":20,"activation":"relu"}]
    model = baseline_model(layers = hiddens)

    # prepare the game for training model
    training_game = NeuralNaviGame(training_game_size,
                                    training_game_size,
                                    model,
                                    model_type = "reinforcement")
    training_game.setup()

    logs = train_model(game = training_game,
                    model = model,
                    episodes = training_episodes,
                    steps = steps)
