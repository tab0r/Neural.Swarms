#! /usr/bin/env python

from logger import log

class Board():
    class TakenException(Exception): pass
    class AboveWidthException(Exception): pass
    class BelowWidthException(Exception): pass
    class AboveHeightException(Exception): pass
    class BelowHeightException(Exception): pass

    @property
    def figures(self):
        return self.positions.keys()

    def __init__(self, height, width):
        self.width = width
        self.height = height
        self.cells = []
        self.positions = dict()
        for y in range(height):
            row = []
            for x in range(width): row.append(None)
            self.cells.append(row)

    def checkPosition(self, y, x):
        if y < 0:  raise Board.BelowHeightException(str(y))
        if x < 0:  raise Board.BelowWidthException(str(x))
        if y >= self.height:  raise Board.AboveHeightException(str(y))
        if x >= self.width:  raise Board.AboveWidthException(str(x))
        if not self.empty(y, x): raise Board.TakenException

    def color(self, y, x):
        figure = self.figure(y, x)
        if figure: return figure.color
        else: return 0

    def add(self, figure, y, x):
        self.checkPosition(y, x)
        self.cells[y][x] = figure
        self.positions[figure] = (y, x)

    def symbol(self, y, x):
        figure = self.figure(y, x)
        if figure: return figure.symbol
        else: return " "

    def empty(self, y, x):
        return self.figure(y, x) == None

    def figure(self, y, x):
        return self.cells[y][x]

    def move(self, figure, y, x, relative = False):
        oldY, oldX = self.positions[figure]
        if relative:
            y = oldY + y
            x = oldX + x
        self.checkPosition(y, x)
        self.cells[oldY][oldX] = None
        self.cells[y][x] = figure
        self.positions[figure] = (y, x)

    def position(self, figure):
        return self.positions[figure]
