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
from keras.optimizers import sgd, RMSprop, Adagrad, Adadelta, Adam
import theano

def baseline_model(optimizer = sgd(lr = 0.001),
                    layers = [{"size":20,"activation":"relu"}]):
    # five inputs - each coordinate, and action selection
    inputs = 5
    # one output - returns the predicted reward for a next state
    num_outputs = 1
    # prepare the navigator model
    model = Sequential()
    # initial inputs
    l = list(layers)
    l0 = l[0]
    del l[0]
    model.add(Dense(l0['size'],
                    input_dim = inputs,
                    activation = l0['activation']))
    # the hidden layers
    for layer in l:
        model.add(Dense(layer['size'], activation=layer['activation']))
    # the output layer
    model.add(Dense(num_outputs, activation='relu'))
    model.compile(optimizer = optimizer,
                    loss = "mean_squared_error")
    return model

def train_model(game, model, episodes = 10, steps = 2):
    initial_e, final_e = 1, .1
    e_delta = initial_e - final_e /10
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
        e = initial_e - j * e_delta
        for i in range(steps):
            # feedforward pass which defines the next state
            choice = game.Navigator.strategy.plan_movement(e)
            # save network input data from above, which is our [s, a]
            input_i = game.Navigator.strategy.get_input(choice)
            # move to s'
            game.step()
            # update our Q[s,a] using the reward we get and
            # the quality prediction for our new state
            target_i = game.Navigator.strategy.last_reward \
                + game.Navigator.strategy.get_quality()
            # online learning
            loss += model.train_on_batch(np.array(input_i).reshape(1, 5),
                                        np.array(target_i))
            # experience replay learning
            inputs.append(input_i)
            targets.append(target_i)
            if game.Navigator.strategy.at_goal == True: wins += 1
        replay_inputs.append(inputs)
        replay_targets.append(targets)
        # pdb.set_trace()
        # experience replay on a random episode
        epi = randint(0, len(replay_inputs) - 1)
        replay_i = np.array(replay_inputs[epi])
        replay_t = np.array(replay_targets[epi]).flatten()
        loss_replay = model.train_on_batch(replay_i, replay_t)
        replay_loss_flay = loss_replay.flatten()[0]
        log.append(loss)
        replay_log.append(replay_loss_flay)
        desc = "Episode " + str(j) + ", Wins: " + str(wins) \
                + ", Replay Loss: " + str(loss_replay)
        t.set_description(desc)
        t.refresh() # to update progress bar
        # shift goal and set last_reward to 0 so next episode is a "fresh game"
        game.shift_goal()
        game.Navigator.strategy.last_reward = 0
    # housekeeping to return everything nicely
    output = dict()
    output['log'] = log
    output['replays'] = replay_log
    output['episodes'] = [replay_inputs, replay_targets]
    return output

if __name__=='__main__':
    training_game_size = 10
    training_episodes = 3
    steps = 10
    # make the model
    model = baseline_model()

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

    print("Training data: ")
    for i in range(training_episodes):
        print("Episode ", i)
        for j in range(steps):
            inputs = logs['episodes'][0][i][j]
            targets = logs['episodes'][1][i][j]
            print(inputs, targets)
