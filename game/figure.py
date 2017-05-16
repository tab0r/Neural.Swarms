#! /usr/bin/env python

from logger import log

class Figure():
    color = 0

    @property
    def symbol(self):
        try: return self.strategy.symbol
        except: return "*"

    def bindStrategy(self, strategy):
        strategy.figure = self
        self.strategy = strategy

    def __init__(self, board):
        self.board = board

    def add(self, y, x):
        self.board.add(self, y, x)

    def position(self):
        return self.board.position(self)

    def move(self, y, x, relative = False):
        self.board.move(self, y, x, relative = relative)


class FigureStrategy():
    @property
    def board(self):
        return self.figure.board
