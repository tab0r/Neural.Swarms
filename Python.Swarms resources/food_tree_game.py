#! /usr/bin/env python

# game imports
from random import random, sample, randint
from game import BoardGame
from board import Board
from figure import Figure, FigureStrategy
from logger import log
from hunt_game import HuntGame, PreyStrategy, PredStrategy

# utilities
import numpy as np
import copy

# Predator game main class
class FoodTree(BoardGame):
    def __init__(self, height = 32, width = 32, apex_amount = 1,
                pred_model=None, prey_model=None):
        self.board = Board(height, width)
        self.pred_model = pred_model
        self.prey_model = prey_model
        self._turn_count = 0
        self.pred_amount = apex_amount
        self.prey_amount = apex_amount * 5
        self.grass_amount = apex_amount * 25

    def setup(self):
        self.Prey = []
        for i in range(self.grass_amount):
            p = Figure(self.board)
            p.bindStrategy(GrassStrategy())
            p.strategy.placeIt()
            p.color = 1
            self.Prey.append(p)
        for i in range(self.prey_amount):
            p = Figure(self.board)
            p.bindStrategy(PreyStrategy(self.prey_model))
            p.strategy.placeIt()
            p.color = 2
            self.Prey.append(p)
        self.Predators = []
        for i in range(self.apex_amount):
            self.p = Figure(self.board)
            self.p.bindStrategy(PredStrategy(self.pred_model))
            self.p.strategy.placeIt()
            self.p.color = 3
            self.Predators.append(p)

    def step(self):
        HuntGame.step(self)
        # hurt existing figures (should make them hungry)
        for figure in self.board.figures:
            figure.strategy.health -= 1
        # add new figures

if __name__=='__main__':
    np.random.seed(2)
    test_game = FoodTree(8, 8)
    test_game.setup()
