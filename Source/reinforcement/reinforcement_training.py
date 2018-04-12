# utilities
import numpy as np
import pickle, os, sys, pdb, time
from tqdm import *
# from keras.models import load_model
# from lstm import build_model as build_lstm

# game imports
from ReinforcementNaviGame import ReinforcementNaviGame
sys.path.append(os.path.abspath("../../../Python.Swarms/Source/"))
from game_display_helper import make_gif

if __name__=='__main__':
    global_start_time = time.time()
    # lets train a DQN model!
    # for i in range(1):
    it_str = "q_table_random_target"# + str(i)
    training_game = ReinforcementNaviGame()
    # model = build_lstm([4, 5, 10, 5])
    # load_file_str = "reinforcement_model_" + it_str + ".h5"
    # print("Loading model: ", load_file_str)
    # model = load_model(load_file_str)
    training_game.setup(model = None)
    # make_gif(training_game, 20, "untrained_" + it_str)
    training_game.train_model(episodes = 600000)
    print('Training duration (s) : ', time.time() - global_start_time)
    # it_str = "15_2000_x" + str(1+i)
    # save_file_str = "reinforcement_model_" + it_str + ".h5"
    # print("Saving model: ", save_file_str)
    # model.save(save_file_str)
    training_game.shift_figure(training_game.Navigator)
    training_game.shift_goal()
    training_game.Navigator.strategy.reset_memory()
    make_gif(training_game, 20, "trained_" + it_str + "_a")
    training_game.shift_figure(training_game.Navigator)
    training_game.shift_goal()
    training_game.Navigator.strategy.reset_memory()
    make_gif(training_game, 20, "trained_" + it_str + "_b")
    training_game.shift_figure(training_game.Navigator)
    training_game.shift_goal()
    training_game.Navigator.strategy.reset_memory()
    make_gif(training_game, 20, "trained_" + it_str + "_c")
    training_game.shift_figure(training_game.Navigator)
    training_game.shift_goal()
    training_game.Navigator.strategy.reset_memory()
    make_gif(training_game, 20, "trained_" + it_str + "_d")
    print('Total duration (s) : ', time.time() - global_start_time)
