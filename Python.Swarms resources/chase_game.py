#! /usr/bin/env python

# game imports
from random import random, sample, randint
from game import BoardGame
from board import Board
from figure import Figure, FigureStrategy
from logger import log
from navi_game import NaviGame, NaviStrategy

# utilities
import numpy as np
import copy

# Predator game main class
class ChaseGame(NaviGame):
    def __init__(self, height, width, amount = 1):
        self._turn_count = 0
        self._amount = amount
        NaviGame.__init__(self, height, width, moving_target = True)

    def setup(self):
        NaviGame.setup(self)
        Followers = []
        for i in range(self._amount):
            p = Figure(self.board)
            p.bindStrategy(FollowStrategy(self.Navigator))
            p.strategy.placeIt()
            p.color = 3
            Followers.append(p)

# Prey for the food
class FollowStrategy(NaviStrategy):
    def __init__(self, figure):
        self.symbol = "*"
        self.leader = figure
        NaviStrategy.__init__(self, goal = self.leader.position())

    def __str__(NaviStrategy):
        return "FollowStrategy"

    def step(self, choice = None):
        self.goal = self.leader.position()
        NaviStrategy.step(self, choice)

if __name__=='__main__':
    np.random.seed(2)
    test_game = ChaseGame(16, 16, amount = 3)
    test_game.setup()
