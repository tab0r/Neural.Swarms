#! /usr/bin/env python

# game imports
from random import random, sample, randint
from game import BoardGame
from board import Board
from figure import Figure, FigureStrategy
from logger import log

# utilities
import numpy as np

# Predator game main class
class ChaseGame(BoardGame):
    def __init__(self, height, width, pred_model=None, prey_model=None):
        self.board = Board(height, width)
        self.pred_model = pred_model
        self.prey_model = prey_model

    def setup(self):
        self.Prey = Figure(self.board)
        self.Prey.bindStrategy(PreyStrategy(self.prey_model))
        self.Prey.strategy.placeIt()
        self.Prey.color = 2
        self.Predator = Figure(self.board)
        self.Predator.bindStrategy(PredStrategy(self.pred_model))
        self.Predator.strategy.placeIt()
        self.Predator.color = 1

# Prey for the target location
class PreyStrategy(FigureStrategy):
    def __init__(self, model):
        self.symbol = "."
        self.model = model
        # stay, right, left, up, down (y, x)
        self.actions = [(0, 0), (0, 2), (0, -2), (2, 0), (-2, 0)]

    def placeIt(self, x=None, y=None):
        if x == None and y == None:
            x = sample(range(0, self.board.width), 1)[0]
            y = sample(range(0, self.board.height), 1)[0]
        try:
            self.figure.add(y, x)
        except:
            self.placeIt(x=x+randint(0, 1), y=y+randint(0, 1))

    def get_input(self):
        ipt = list(self.figure.position())
        ipt.extend(self.predator().position())
        return ipt

    def plan_movement(self):
        ipt = self.get_input()
        dist_y = ipt[0] - ipt[2]
        dist_x = ipt[1] - ipt[3]
        # this agent never chills
        # do something
        if abs(dist_y) < abs(dist_x):
            # x distance is greater
            # so change y
            if dist_y < 0:
                # less than 0, so move down
                choice = 4
            elif dist_y > 0:
                choice = 3
            else:
                choice = np.random.choice([3, 4])
        elif abs(dist_y) > abs(dist_x):
            # y distance is greater
            if dist_x < 0:
                # less than 0, so move right
                choice = 2
            elif dist_x > 0:
                choice = 1
            else:
                choice = np.random.choice([1, 2])
        else:
            # the predator is on a 45 degree line, so make a random choice
            if (dist_x < 0) and (dist_y < 0):
                choice = np.random.choice([2, 4])
            elif (dist_x > 0) and (dist_y < 0):
                choice = np.random.choice([1, 4])
            elif (dist_x > 0) and (dist_y > 0):
                choice = np.random.choice([1, 3])
            elif (dist_x < 0) and (dist_y > 0):
                choice = np.random.choice([2, 3])
            else:
                choice = randint(1, 4)
        return choice

    def step(self):
        ipt = self.get_input()
        if self.model != None:
            choice = self.model.predict(np.array(ipt).reshape(1, 4))
            action = self.actions[np.argmax(choice)]
        else:
            choice = self.plan_movement()
            action = self.actions[choice]
        try:
            self.figure.move(action[0], action[1], relative = True)
        except self.board.AboveWidthException:
            action = self.actions[2]
        except self.board.AboveHeightException:
            action = self.actions[4]
        except self.board.BelowWidthException:
            action = self.actions[1]
        except self.board.BelowHeightException:
            action = self.actions[3]
        except self.board.TakenException:
            action = self.actions[randint(1,4)]

    def predator(self):
        predators = []
        y, x = self.figure.position()
        radius = 0
        while len(predators) <= 1:
            radius += 1
            for height in range(y - radius, y + radius + 1):
                for width in range(x - radius, x + radius + 1):
                    try:
                        figure = self.board.figure(height, width)
                        if figure: predators.append(figure)
                    except:
                        pass
        predators.remove(self.figure)
        return predators[0]

# Predator to get to target
class PredStrategy(FigureStrategy):
    def __init__(self, model):
        self.symbol = "O"
        self.model = model
        # stay, right, left, up, down (y, x)
        self.actions = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]

    def placeIt(self):
        x = sample(range(0, self.board.width), 1)[0]
        y = sample(range(0, self.board.height), 1)[0]
        try:
            self.figure.add(y, x)
        except:
            self.placeIt()

    def get_input(self):
        ipt = list(self.figure.position())
        ipt.extend(list(self.prey().position()))
        return ipt

    def plan_movement(self):
        ipt = self.get_input()
        dist_y = ipt[0] - ipt[2]
        dist_x = ipt[1] - ipt[3]
        if (dist_y == 1 and dist_x ==0) or (dist_y == 0 and dist_x == 1):
            # chill
            choice = 0
        else:
            # do something
            if abs(dist_y) < abs(dist_x):
                # x distance is greater
                if dist_x < 0:
                    # less than 0, so move right
                    choice = 1
                else:
                    choice = 2
            else:
                # y distance is greater or equal
                if dist_y < 0:
                    # less than 0, so move up
                    choice = 3
                else:
                    choice = 4
        return choice

    def step1(self):
        pass

    def step(self):
        ipt = self.get_input()
        if self.model != None:
            choice = self.model.predict(np.array(ipt).reshape(1, 4))
            action = self.actions[np.argmax(choice)]
        else:
            choice = self.plan_movement()
            action = self.actions[choice]
        try:
            self.figure.move(action[0], action[1], relative = True)
        except self.board.AboveWidthException:
            action = self.actions[0]
            #self.figure.move(action[0], action[1], relative = True)
        except self.board.AboveHeightException:
            action = self.actions[0]
            #self.figure.move(action[0], action[1], relative = True)
        except self.board.BelowWidthException:
            action = self.actions[0]
            #self.figure.move(action[0], action[1], relative = True)
        except self.board.BelowHeightException:
            action = self.actions[0]
            ##self.figure.move(action[0], action[1], relative = True)
        except self.board.TakenException:
            action = self.actions[0]

    def prey(self):
        prey = None
        y, x = self.figure.position()
        radius = 0
        while prey == None:
            radius += 1
            for height in range(y - radius, y + radius + 1):
                for width in range(x - radius, x + radius + 1):
                    try:
                        figure = self.board.figure(height, width)
                        if figure and (figure != self.figure):
                            prey = figure
                    except:
                        pass
        return prey

if __name__=='__main__':
    np.random.seed(2)
    test_game = ChaseGame(16, 16)
    test_game.setup()
