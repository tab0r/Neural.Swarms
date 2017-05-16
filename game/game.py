#! /usr/bin/env python

from logger import log
from board import Board
from figure import Figure

class Game():
    pass

class BoardGame(Game):

    def __init__(self, height, width):
        self.board = Board(height, width)

    def step(self):
        for figure in self.board.figures:
            figure.strategy.step()
