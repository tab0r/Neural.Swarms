# utilities
import numpy as np
import pickle, os, sys, pdb, time
from tqdm import *
from keras.models import load_model, Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam
from lstm import build_model as build_lstm

# game imports
from ReinforcementNaviGame import ReinforcementNaviGame
sys.path.append(os.path.abspath("../../../Python.Swarms/Source/"))
from game_display_helper import make_gif

def test_q_table(it_str, load = False, train = False):
    global_start_time = time.time()
    training_game = ReinforcementNaviGame()
    training_game.setup(model = None)
    if load == True:
        q_table_load = pickle.load(open("q_table.pkl", 'rb'))
        training_game.Navigator.strategy.q_table = q_table_load
    if train == True:
        training_game.train_strategy()
        q_table_dump = training_game.Navigator.strategy.q_table
        pickle.dump(q_table_dump, open("q_table.pkl", 'wb'))
    test_strategy(it_str, training_game)

def test_dqn(it_str, load = False, train = False):
    global_start_time = time.time()

    if load == True:
        load_file_str = "reinforcement_model_" + it_str + ".h5"
        print("Loading model: ", load_file_str)
        model = load_model(load_file_str)
    else:
        # for LSTMDeepQ
        # model = build_lstm([2, 5, 4, 5])
        # for MLP DeepQ
        model = Sequential()
        model.add(Dense(5, input_dim = 2, activation = 'relu'))
        # model.add(Dense(5, activation = 'relu'))
        model.add(Dense(5, activation = 'sigmoid'))
        model.compile(optimizer = Adam(), loss = "mean_squared_error")

    training_game = ReinforcementNaviGame()
    training_game.setup(model)
    if train == True:
        training_game.train_strategy()
    save_file_str = "reinforcement_model_" + it_str + ".h5"
    print("Saving model: ", save_file_str)
    model.save(save_file_str)
    test_strategy("MLP_", training_game)
    print('Total duration (s) : ', time.time() - global_start_time)

def test_strategy(it_str, game, qty = 100):
    for i in range(qty):
        game.reset()
        make_gif(game, 10, "test_" + it_str + str(i))

if __name__=='__main__':
    global_start_time = time.time()
    # lets train a DQN model!
    it_str = "test"
    # train_q_table(it_str, True, False)
    test_dqn(it_str, False, True)
