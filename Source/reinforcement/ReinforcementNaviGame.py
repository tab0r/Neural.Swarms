# utilities
import numpy as np
import pandas as pd
import pickle, os, sys
from tqdm import *

# game imports
from Q_strategies import Q_Learner, DeepQ, DeepQxr, LSTMDeepQ
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
            strategy = DeepQxr(self.goal, model)
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

    def train_strategy(self, episodes = 336000, steps = 5, e_start = 1.0, e_stop = 0.25):
        e_delta = e_start - e_stop / episodes
        desc = "Strategy training starting"
        # turn on training
        self.Navigator.strategy.online_learning = True
        # set up tqdm progress bar to display loss dynamically
        t = trange(episodes, desc=desc, leave=True)
        for j in t:
            # reset game - score, last_reward, set to zero
            # figure moves to random position, memory cleared
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

    def reset(self):
        self.Navigator.strategy.reset_memory()
        NaviGame.reset(self)
