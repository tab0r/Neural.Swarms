#! /usr/bin/env python

from random import random, sample, randint
from moskito_game import MoskitoGame, MoskitoStrategy
from figure import Figure
from logger import log
from time import sleep

class BirdGame(MoskitoGame):
    def figureStrategyFactory(self):
        strategy = BirdStrategy()
        strategy.amountOfNeighbours = self.amountOfNeighbours
        return strategy

class BirdStrategy(MoskitoStrategy):
    symbol = "-"
    deltaX, deltaY = (1, 1)

    def planMovement(self):
        mnm = self.meanNeighborMovement(self.amountOfNeighbours)
        self.deltaY, self.deltaX = mnm
        self.wallAvoiding()
        # if self.deltaY == 0 or self.deltaX == 0: self.modify()

    def mean(self, l):
        if float(sum(l))/float(len(l)) < 0: return -1
        else: return 1

    def meanNeighborMovement(self, amount):
        neighbours = self.neighbours(amount)
        deltaX, deltaY = [], []
        for neighbour in neighbours:
            deltaY.append(neighbour.strategy.deltaY)
            deltaX.append(neighbour.strategy.deltaX)
        return (self.mean(deltaY), self.mean(deltaX))

    def modify(self):
        self.deltaX = sample((-1, 1), 1)[0]
        self.deltaY = sample((-1, 1), 1)[0]

    def neighbours(self, amount):
        neighbours = []
        y, x = self.figure.position()
        radius = 0
        while len(neighbours) <= amount:
            radius += 1
            for height in range(y - radius, y + radius + 1):
                for width in range(x - radius, x + radius + 1):
                    try:
                        figure = self.board.figure(height, width)
                        if figure: neighbours.append(figure)
                    except:
                        pass
        neighbours.remove(self.figure)
        return neighbours

    def wallAvoiding(self):
        y, x = self.figure.position()
        width = self.board.width
        height = self.board.height
        if random() < 1.0/(y + 1): self.deltaY = abs(self.deltaY)
        if random() < 1.0/(x + 1): self.deltaX = abs(self.deltaX)
        if random() < 1.0/(height - y): self.deltaY = -abs(self.deltaY)
        if random() < 1.0/(width - x): self.deltaX = -abs(self.deltaX)





