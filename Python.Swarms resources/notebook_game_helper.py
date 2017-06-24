import matplotlib as plt
import pylab as pl
import numpy as np
from IPython import display

def draw_game(game, mpl=True):
    s = np.array(game.board.cells)
    color = lambda i: 0 if i == None else i.color
    colormask = np.vectorize(color)
    s_colors = colormask(s)
    if mpl==True:
        pl.close('all')
        ax = pl.gca()
        ax.set_aspect('equal')
        pl.pcolor(s_colors)
        pl.colorbar()
        pl.show();
    else:
        print(s_colors)

def step_game(game, n=1):
    for i in range(n):
        game.step()

def step_test(game, n=10):
    for i in range(n):
        s0 = np.array(game.board.cells)
        game.step()
        s1 = np.array(game.board.cells)
        if s0.all() == s1.all():
            pass
        else:
            draw_game(game)

def step_and_draw_game(game, mpl=True):
    step_game(game)
    draw_game(game, mpl)

def animate_game(game, n, mpl=True):
    for i in range(n):
        display.clear_output(wait=True);
        display.display(step_and_draw_game(game, mpl));

def get_strategies(game):
    for figure in game.board.figures:
        print(figure.strategy.deltaY, figure.strategy.deltaX)
