# utilities
import numpy as np
import pandas as pd
import pickle, os, sys
from tqdm import *

# game imports
sys.path.append(os.path.abspath("../../../Python.Swarms/Source"))
from navi_game import NaviGame, NaviStrategy

# Reinforcement game main class
# Differences from NaviGame:
# - a bunch of options removed (goal idle, moving target, goal itself)
# - feeds the Reinforcement Strategy a reward (diff between last two frames scores)
class ReinforcementNaviGame(NaviGame):
    def __init__(self):
        NaviGame.__init__(self)
        self.display_str = "Game start"
        self.last_score = 0

    def setup(self, model):
        if model == None:
            strategy = Q_Learner(self.goal, self.model)
        else:
            strategy = DeepQ(self.goal, model)
        NaviGame.setup(self, strategy)
        self.shift_goal()

    def step(self):
        NaviGame.step(self)
        reward = self.score - self.last_score
        self.last_score = self.score
        self.Navigator.strategy.get_reward(reward)
        self.display_str = ""
        self.display_str += "Score: "
        self.display_str += "{0:.2f}".format(self.score)
        self.display_str += " Last Reward: "
        self.display_str += "{0:.2f}".format(reward)

    def train_strategy(self, episodes = 336000, steps = 10, e_start = 1.0, e_stop = 0.1):
        e_delta = e_start - e_stop / episodes
        desc = "Strategy training starting"
        # turn on training
        self.Navigator.strategy.online_learning = True
        # set up tqdm progress bar to display loss dynamically
        t = trange(episodes, desc=desc, leave=True)
        for j in t:
            # reset game - score, last_reward, set to zero
            # figure moves to random position, memory cleared
            # implement self.reset() function later
            self.reset()
            self.last_score = 0
            self.Navigator.strategy.reset_memory()
            # calculate epsilon greedy value for episode
            e = e_start - j * (e_start-e_stop)/episodes
            self.Navigator.strategy.epsilon = e
            # play through episode
            for i in range(steps):
                self.step()
            desc = "Episode " + str(j)
            t.set_description(desc)
            t.refresh() # to update progress bar

        # reset values - now ready to play
        self.Navigator.strategy.online_learning = False
        self.Navigator.strategy.reset_memory()
        self.score = 0
        self.last_score = 0

class Q_Learner(NaviStrategy):
    def __init__(self, goal):
        # Q learner
        self.q_table = dict()
        self.online_learning = False
        self.pixel_input_mode = False
        self.reset_memory()
        NaviStrategy.__init__(self, goal)

    def reset_memory(self):
        self.memory = {"inputs": [], "choices": [], "rewards": []}
        self.epsilon = 0.01

    def get_reward(self, reward):
        self.memory['rewards'].append(reward)

    def step(self, choice = None):
        # make predictions
        # get this frames input
        s_prime, dist = self.get_input()
        # lookup the quality values
        if str(s_prime) in self.q_table.keys(): # check if we've seen this state
            quality = self.q_table[str(s_prime)] # grab the values
        else: # if not, generate some random numbers with mean zero
            quality = 2*np.random.random((1,5)) - 1

        # perform training step
        if (self.online_learning == True) and (len(self.memory['inputs']) > 0):
            s = self.memory['inputs'][-1]
            a = self.memory['choices'][-1] # these are -1 because the game gives rewards
            r = self.memory['rewards'][-1] # after the figure.step() has occured
            if str(s) in self.q_table.keys(): # check if we've seen this state
                target = self.q_table[str(s)] # grab the values
            else: # if not, generate some random numbers with mean zero
                target = np.random.random((1,5)) - 1
            gamma = 0.1
            target[0][a] = r + gamma * np.max(quality[0])
            self.q_table[str(s)] = target
        # store this frames' input
        self.memory['inputs'].append(s_prime)

        if choice == None:
            d = np.random.random()
            # explore some of the time
            if d < self.epsilon:
                choice = np.random.randint(0, 4)
            # exploit current Q-function
            else:
                choice = np.argmax(quality)

        e_choice = NaviStrategy.step(self, choice) # since the model may attempt
        # to wander off the board, or into the flag, e_choice may be different
        # from choice. maybe do something if so, but so far handle it well
        self.memory['choices'].append(choice) # store a'

# WIP LSTM-DQN strategy
class DeepQ(Q_Learner):
    def __init__(self, goal, model):
        # Deep-Q network
        self.model = model
        self.input_sequences = []
        self.sequence_length = 5
        self.experience_replay = False
        Q_Learner.__init__(self, goal)

    def reset_memory(self):
        Q_Learner.reset_memory(self)
        self.memory['input_sequences'] = []

    def prep_sequences(self):
        # set up initial sequence values - just the agent sitting in
        # its starting position, accumulating rewards
        ipt, dist = self.get_input(pixel_ipt = self.pixel_input_mode)
        for _ in range(self.sequence_length):
            self.memory['inputs'].append(ipt)
        self.memory['choices'].append(4)
        if dist == 1:
            self.memory['rewards'].append(1)
        else:
            self.memory['rewards'].append(-0.1)
        self.memory['input_sequences'].append(np.array([self.memory['inputs'][-self.sequence_length:]],))

    def placeIt(self):
        NaviStrategy.placeIt(self)
        self.prep_sequences()

    def step(self, choice = None):
        # get this frames input
        ipt, dist = self.get_input(pixel_ipt = self.pixel_input_mode)
        if len(self.memory['inputs']) == 0:
            self.prep_sequences()
        # store this frames' input
        self.memory['inputs'].append(ipt)
        s_prime = np.array([self.memory['inputs'][-self.sequence_length:]],)
        # make the quality prediction
        quality = self.model.predict(s_prime)

        # perform training step
        if (self.online_learning == True):# and (len(self.input_sequences) > 1):
            s = self.memory['input_sequences'][-1]
            a = self.memory['choices'][-1] # these are -1 because the game gives rewards
            r = self.memory['rewards'][-1] # after the figure.step() has occured
            target = self.model.predict(s)
            gamma = 0.1 # hard coding discount at 0.5 because games are short, there's some stochasticity around boundaries and only the reward itself matters
            target[0][a] = r + gamma * np.max(quality[0])
            self.model.train_on_batch(s, target)
        self.input_sequences.append(s_prime) # make s' = self.input_sequences[-1]

        if choice == None:
            d = np.random.random()
            # explore some of the time
            if d < self.epsilon:
                choice = np.random.randint(0, 4)
            # exploit current Q-function
            else:
                choice = np.argmax(quality)

        e_choice = NaviStrategy.step(self, choice) # since the model may attempt to wander off the board
        self.memory['choices'].append(e_choice) # store a'

        # experience replay
        if self.experience_replay == True:
            # experience replay
            print("experience_replay - Not yet implemented")
            pass
