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
    # four inputs - each coordinate
    inputs = 4
    # five outputs - one for each action
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
    # the hidden layers
    for layer in l:
        model.add(Dense(layer['size'], activation=layer['activation']))
    # the output layer
    model.add(Dense(num_outputs, activation='linear'))
    model.compile(optimizer = optimizer,
                    loss = "mean_squared_error")
    return model

def train_model(game, model, episodes = 10, steps = 2):
    initial_e, final_e = 1, 0
    # use 1 until we add stochasticity to the model
    gamma = 1
    e_delta = initial_e - final_e / episodes
    reward_total = 0
    r_totals = []
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
            choice = game.Navigator.strategy.plan_movement(e)
            quality_prediction = game.Navigator.strategy.get_quality()
            # save network input data from above, which is our <s>
            ipt, dist = game.Navigator.strategy.get_input()
            input_i = np.array(ipt).reshape(1, 4)

            # move to s'
            game.step()

            # update our Q[s,a] using the reward we get and
            # the quality prediction for our new state
            reward = game.Navigator.strategy.get_reward()
            quality = reward
            if dist > game.tolerance:
                quality += gamma * quality_prediction.flatten()[choice]
            r_total_ep += reward
            reward_total += reward
            r_totals.append(reward_total)

            target_i = quality_prediction
            target_i[0][choice] = quality
            target_i = target_i.reshape(1, 5)
            # online learning
            loss = model.train_on_batch(input_i, target_i)

            # store data for experience replay learning
            inputs.append(input_i)
            targets.append(target_i)
            rewards.append(reward)
        loss_log.append(loss)

        # experience replay on a random experience
        # expand to a full "episode" of random experiences
        if len(inputs) == 1:
            experience = 0
        else:
            lower_limit = int(0.45 * len(inputs))
            experience = randint(0, max(1, len(inputs)-1))
        replay_i = inputs[experience]
        replay_t = targets[experience]
        loss_replay = model.train_on_batch(replay_i, replay_t).flatten()[0]
        replay_log.append(loss_replay)
        rt_str = '{0:.3g}'.format(r_total_ep)
        loss_str = '{0:.3g}'.format(loss_replay)
        desc = "Episode " + str(j) + ", Rewards: " + rt_str \
                + ", Wins: " + str(game.Navigator.strategy.wins)
                #+ ", Replay Loss: " + loss_str
        t.set_description(desc)
        t.refresh() # to update progress bar
        # shift goal and set last_reward to 0 so next episode is a "fresh game"
        game.shift_goal()
        game.Navigator.strategy.last_reward = 0
    # housekeeping to return everything nicely
    output = dict()
    output['log'] = loss_log
    output['replays'] = replay_log
    output['experiences'] = [inputs, targets]
    output['rewards'] = rewards
    output['reward_totals'] = r_totals
    return output

if __name__=='__main__':
    debug = int(input("Debug? (0/1): "))
    if debug == 1:
        pdb.set_trace()
    training_game_size = int(input("Training game size: "))
    training_episodes = int(input("Training episodes: "))
    steps = int(input("Steps per episode: "))
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

    # print("Training data: ")
    # for i in range(training_episodes):
    #     print("Episode ", i)
    #     for j in range(steps):
    #         inputs = logs['experiences'][0][i*steps + j]
    #         target = ['{0:.3g}'.format(num) for num in
    #                         logs['experiences'][1][i*steps + j][0]]
    #         reward = logs['rewards'][i*steps + j]
    #         print(inputs, target, reward)
