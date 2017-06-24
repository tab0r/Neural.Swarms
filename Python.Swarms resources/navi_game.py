#! /usr/bin/env python

# game imports
from random import random, sample, randint
from game import BoardGame
from board import Board
from figure import Figure, FigureStrategy
from logger import log

# utilities
import numpy as np
import pdb

# Navigator game main class
class NaviGame(BoardGame):
    def __init__(self,
            height,
            width,
            goal = None,
            moving_target = False,
            tolerance = 1.1,
            goal_idle = 2,
            display_str = "NaviGame"):
        self.board = Board(height, width)
        # if goal == None:
        #     self.goal = (randint(0,width-1), randint(0,height-1))
        # else:
        self.goal = goal
        self.moving_target = moving_target
        self.tolerance = tolerance
        self.goal_idle = goal_idle
        self.display_str_base = display_str
        self.wins = 0

    def setup(self):
        self.Flag = Figure(self.board)
        self.Flag.bindStrategy(FlagStrategy())
        if self.goal == None:
            self.Flag.strategy.placeIt()
            self.goal = self.Flag.position()
        else:
            # pdb.set_trace()
            self.Flag.strategy.placeIt(y = self.goal[0], x = self.goal[1])
        self.Flag.color = 2
        self.Navigator = Figure(self.board)
        self.Navigator.bindStrategy(NaviStrategy(
                                    goal = (self.goal[1], self.goal[0]),
                                    tolerance = self.tolerance))
        self.Navigator.strategy.placeIt()
        self.Navigator.color = 3

    def shift_goal(self, goal = None, figure = None):
        if goal == None:
            goal = (randint(0, self.board.width-1), randint(0, self.board.height-1))
        try:
            self.goal = goal
            self.Flag.move(y = goal[0], x = goal[1], relative=False)
            if figure == None:
                self.Navigator.strategy.goal = goal
            else:
                figure.strategy.goal = goal
        except:
            self.shift_goal()

    def step(self):
        for figure in self.board.figures:
            figure.strategy.step()
        if (self.Navigator.strategy.at_goal > self.goal_idle) \
                        and (self.moving_target == True):
            self.shift_goal()
            self.wins += 1
            self.Navigator.strategy.at_goal = 0
        self.display_str = self.display_str_base + ", Wins: " + str(self.wins)

    def add_block(self, position = None):
        block = Figure(self.board)
        block.bindStrategy(FlagStrategy())
        if position != None:
                block.strategy.placeIt(y = position[0], x = position[1], soft = True)
        else:
                block.strategy.placeIt(soft = True)
        block.color = 1
        return block

    def add_wall(self, start = None, length = 5, step = None):
        Wall = []
        if step == None:
            step = (0, 0)
            while step == (0, 0):
                step = (randint(-2, 2),randint(-2, 2))
        if start == None:
            xstart = randint(1, self.board.width)
            ystart = randint(1, self.board.height)
            start = (ystart, xstart)
        for i in range(length):
            pos = (start[0] + i * step[0], start[1] + i * step[1])
            if (pos[0] < self.board.height) and (pos[1] < self.board.width):
                # there is just one error here - board taken exception
                # we can skip that segment
                try:
                    Wall.append(self.add_block(pos))
                except:
                    pass
        return Wall

# flag for the target location
class FlagStrategy(FigureStrategy):
    symbol = "~"

    def placeIt(self, x=None, y=None, soft = False):
        if x == None and y == None:
            y = sample(range(0, self.board.height), 1)[0]
            x = sample(range(0, self.board.width), 1)[0]
        try:
            self.figure.add(y=y, x=x)
        except:
            if soft == False:
                self.placeIt(x=x+randint(-1, 1), y=y+randint(-1, 1))
            else:
                print("Figure position not available, figure not placed")
                pass

    def step(self):
        pass

# navigator to get to target
class NaviStrategy(FigureStrategy):
    def __init__(self, goal, tolerance = 1.4):
        self.symbol = "."
        self.goal = goal
        # right, left, up, down (y, x), stay
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]
        self.at_goal = 0
        self.tolerance = tolerance
        self.last_choice = 4

    def __str__(FigureStrategy):
        return "NaviStrategy"

    def placeIt(self, x=None, y=None, soft = False):
        if x == None and y == None:
            y = sample(range(0, self.board.height), 1)[0]
            x = sample(range(0, self.board.width), 1)[0]
        try:
            self.figure.add(y=y, x=x)
        except:
            if soft == False:
                self.placeIt(x=x+randint(-1, 1), y=y+randint(-1, 1))
            else:
                print("Figure position not available, figure not placed")
                pass

    def get_input(self, position = None):
        if position == None:
            position = list(self.figure.position())
        goal = self.goal
        dist = self.get_distance(position, goal)
        ipt = list(position)
        ipt.extend(list(goal))
        return ipt, dist

    def get_distance(self, position, goal):
        #return np.abs(position - np.array(self.goal)).sum()
        return np.linalg.norm(np.array(position) - np.array(goal))

    def plan_movement(self, debug = False):
        # use the deterministic strategy
        ipt, dist = self.get_input()
        dist_y = ipt[0] - ipt[2]
        dist_x = ipt[1] - ipt[3]
        if debug == True: pdb.set_trace()
        if dist <= self.tolerance:
            # chill
            choice = 4
        else:
            # do something
            self.at_goal = 0
            if abs(dist_y) < abs(dist_x):
                # x distance is greater
                if dist_x < 0:
                    # less than 0, so move right
                    choice = 0
                else:
                    choice = 1
            else:
                # y distance is greater or equal
                if dist_y < 0:
                    # less than 0, so move up
                    choice = 2
                else:
                    choice = 3
        # return self.actions[choice]
        return choice

    def step(self, choice = None):
        position = self.figure.position()
        goal = self.goal
        dist = self.get_distance(position, goal)
        if choice == None:
            choice = self.plan_movement()
        self.last_choice = choice
        if (dist < self.tolerance):# and (choice == 0):
            self.at_goal += 1
        else:
            self.at_goal = 0
        if choice != 4:
            action = self.actions[choice]
            try:
                self.figure.move(y = action[0], x = action[1], relative = True)
            except:
                choice = randint(0,4)
                self.step(choice = choice)

if __name__=='__main__':
    test_game = NaviGame(8, 8, moving_target = True, tolerance = 1.4, goal_idle = 4)
    test_game.setup()
