#! /usr/bin/env python

# game imports
from random import random, sample, randint
from game import BoardGame
from board import Board
from figure import Figure, FigureStrategy
from logger import log

# utilities
import numpy as np
import copy

# Predator game main class
class HuntGame(BoardGame):
    def __init__(self, height, width, amount = 1,
                pred_model=None, prey_model=None):
        self.board = Board(height, width)
        self.pred_model = pred_model
        self.prey_model = prey_model
        self._turn_count = 0
        self._amount = amount

    def setup(self):
        Prey = []
        for i in range(self._amount):
            p = Figure(self.board)
            p.bindStrategy(PreyStrategy(self.prey_model))
            p.strategy.placeIt()
            p.color = 2
            Prey.append(p)
        self.Predator = Figure(self.board)
        self.Predator.bindStrategy(PredStrategy(self.pred_model))
        self.Predator.strategy.placeIt()
        self.Predator.color = 1

    def step(self):
        dead_figures = []
        for figure in self.board.figures:
            if figure.strategy.health == 0:
                dead_figures.append(figure)
                #del figure
            else:
                figure.strategy.step()
        for figure in dead_figures:
            #print("A figure is dead")
            pos = figure.position()
            del self.board.positions[figure]
            self.board.cells[pos[0]][pos[1]] = None

# Prey for the food
class PreyStrategy(FigureStrategy):
    def __init__(self, model, name = "Bunny",
            predator_types = set(["Wolf", "Hawk"]),
            health = 5, vision_radius = 5, speed = 2):
        self.symbol = "."
        self.model = model
        # stay, right, left,
        #          up, down (y, x)
        self.actions = [(0, 0), (0, speed), (0, -speed),
                                (speed, 0), (-speed, 0)]
        self._last_seen_predator = None
        self.name = name
        self.predator_types = predator_types
        self.vision_radius = 5
        self.max_health = copy.copy(health)
        self.health = health
        self.states = {"healthy": 5, "injured": 3, "dead": 0}

    def __str__(PreyStrategy):
        return PreyStrategy.name

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
        ipt.extend(self.predator())
        return ipt

    def plan_movement(self):
        ipt = self.get_input()
        if ((ipt[2] == -1) and (ipt[3] == -1)):
            choice = 0
        else:
            dist_y = ipt[0] - ipt[2]
            dist_x = ipt[1] - ipt[3]
            # health check
            if (abs(dist_y) == 1) or (abs(dist_x) == 1):
                self.health -= 1
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
        if self.health <= self.states["injured"]:
            choice = 0
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
        predator = None
        y, x = self.figure.position()
        radius = 0
        for i in range(self.vision_radius):
            for height in range(y - radius, y + radius + 1):
                for width in range(x - radius, x + radius + 1):
                    try:
                        figure = self.board.figure(height, width)
                        if figure:
                            # print("Found a figure", figure.strategy.name)
                            if str(figure.strategy) in self.predator_types:
                                # print("Found a predator")
                                predator = figure.position()
                                self._last_seen_predator = predator
                    except:
                        pass
            radius += 1
        if predator == None:
            predator = (-1, -1)
        return predator

# Predator to get to target
class PredStrategy(PreyStrategy):
    def __init__(self, model,
                name = "Wolf",
                prey_types = {"Bunny"},
                predator_types = {},
                vision_radius = 7,
                health = 9,
                speed = 1):
        PreyStrategy.__init__(self,
                model = model,
                name = name,
                predator_types = predator_types,
                vision_radius = vision_radius,
                health = health,
                speed = speed)
        self.symbol = "O"
        self.eating = False
        self.prey_types = prey_types
        self.states = {"healthy": 9, "hungry": 6, "injured": 2,
                         "dying": 1, "dead": 0}

    def get_input(self):
        ipt = PreyStrategy.get_input(self)
        ipt.extend(list(self.prey()))
        return ipt

    def plan_movement(self):
        prey_choice = PreyStrategy.plan_movement(self)
        if prey_choice != 0:
            choice = prey_choice
        elif self.health <= self.states["injured"]:
            choice = 0
        else:
            ipt = self.get_input()
            dist_y = ipt[0] - ipt[4]
            dist_x = ipt[1] - ipt[5]
            if (ipt[4] == -1) and (ipt[5] == -1):
                choice = randint(1, 4)
            else:
                if (dist_y == 1 and dist_x ==0) or (dist_y == 0 and dist_x == 1):
                    # chill
                    self.eating = True
                    if self.health < self.max_health:
                        self.health += 1
                    choice = 0
                else:
                    # do something
                    self.eating = False
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

    def step_pass(self):
        pass

    def prey(self):
        prey = None
        y, x = self.figure.position()
        radius = 0
        for i in range(self.vision_radius):
            radius += 1
            for height in range(y - radius, y + radius + 1):
                for width in range(x - radius, x + radius + 1):
                    try:
                        figure = self.board.figure(height, width)
                        if str(figure.strategy) in self.prey_types:
                            prey = figure.position()
                    except:
                        pass
        if prey == None:
            prey = (-1, -1)
        return prey

if __name__=='__main__':
    np.random.seed(2)
    test_game = HuntGame(8, 8)
    test_game.setup()
