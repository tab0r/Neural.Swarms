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

def model_benchmark(figure, goal):
    print('''Prediction Test: If network does not provide differentiated
     responses to these inputs, keep training or modify network. The piece
    is on opposite sides of the goal here so the predicted rewards should
    be opposites.''')
    print(" Note that these outputs correspond to these actions:")
    actions = figure.strategy.actions
    model = figure.strategy.model
    print(" (y,x): ",actions)
    ipt0 = [goal[0], goal[1]+2]
    ipt0.extend(list(goal))
    ipt0.append(-0.1)
    predict0 = model.predict(np.array(ipt0).reshape(1,4))
    choice0 = np.argmax(predict0)
    # print(predict0, choice0)
    print('''   Position +(0, 1),  Goal {}: {}'''.format(goal, choice0))
    ipt1 = [goal[0], goal[1]-2]
    ipt1.extend(list(goal))
    ipt1.append(-0.1)
    predict1 = model.predict(np.array(np.array(ipt1)).reshape(1,4))
    choice1 = np.argmax(predict1)
    # print(predict1, choice1)
    print('''   Position -(0, 2), Goal {}: {}'''.format(goal, choice1))
    ipt2 = [goal[0]+2, goal[1]]
    ipt2.extend(list(goal))
    ipt2.append(-0.1)
    predict2 = model.predict(np.array(ipt2).reshape(1,4))
    choice2 = np.argmax(predict2)
    # print(predict2, choice2)
    print('''   Position +(2, 0),  Goal {}: {}'''.format(goal, choice2))
    ipt3 = [goal[0]-2, goal[1]]
    ipt3.extend(list(goal))
    ipt3.append(-0.1)
    predict3 = model.predict(np.array(ipt3).reshape(1,4))
    choice3 = np.argmax(predict3)
    # print(predict3, choice3)
    print('''   Position -(2, 0), Goal {}: {}'''.format(goal, choice3))
    # print(''' Are they equal? If so this is extra bad... ''')
    # print(model.predict(np.array([1,7]).reshape(1,2)) == model.predict(np.array([14,7]).reshape(1,2)))

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
    model.add(Dense(num_outputs, activation='sigmoid'))
    model.compile(optimizer = optimizer,
                    loss = "mean_squared_error")
    return model

def train_model(game, model, episodes = 0, steps = 50,
                epochs_per_episode = 5, batch_size = 10):
    logs = []
    for j in range(episodes):
        inputs, targets = [], []
        for i in range(steps):
            position = game.Navigator.position()
            goal = game.Navigator.strategy.goal
            inputs.append(game.Navigator.strategy.get_input(position))
            targets.append(get_reward(game.Navigator, position))
            game.step()
        # pdb.set_trace()
        loss = model.train_on_batch(inputs, targets)
        logs.append(loss)
        print("Episode " + str(j) + " loss: " + str(loss))
    return logs

def get_reward(figure, position, look_forward = 10, depth = 0):
    goal = figure.strategy.goal
    dist = get_distance(position, goal)
    reward = -0.1
    if (dist == 1.0):
        reward += 1
    elif depth < look_forward:
        choice = figure.strategy.plan_movement()
        action = figure.strategy.actions[choice]
        next_pos = (position[0]+action[0], position[1]+action[1])
        reward += get_reward(figure, next_pos, look_forward = look_forward, depth = depth +1)
    return reward

def get_distance(position, goal):
    #return np.abs(position - np.array(self.goal)).sum()
    return np.linalg.norm(position - np.array(goal))

def main(training_game_size = 10, training_episodes = 10):
    # make the model
    model = baseline_model()

    # prepare the game for training model
    training_game = NeuralNaviGame(training_game_size, training_game_size,
                                    model = model,
                                    # model_type = "reinforcement",
                                    moving_target = True)
    training_game.setup()

    print("Training model")
    logs = train_model(game = training_game,
                    model = model,
                    episodes = training_episodes)
    return model, logs

if __name__=='__main__':
    model, log = main(training_episodes = 50)
