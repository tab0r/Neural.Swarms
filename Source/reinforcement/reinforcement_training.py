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

def train_q_table(it_str, load = True, train = True):
    training_game = ReinforcementNaviGame()
    training_game.setup(model = None)
    if load == True:
        q_table_load = pickle.load(open("q_table.pkl", 'rb'))
        training_game.Navigator.strategy.q_table = q_table_load
    if train == True:
        training_game.train_model()
        print('Training duration (s) : ', time.time() - global_start_time)
        q_table_dump = training_game.Navigator.strategy.q_table
        pickle.dump(q_table_dump, open("q_table.pkl", 'wb'))
    test_strategy(it_str, training_game)

def test_strategy(it_str, game, qty = 100):
    for i in range(qty):
        game.reset()
        make_gif(game, 10, "test_" + it_str + str(i))

if __name__=='__main__':
    global_start_time = time.time()
    # lets train a DQN model!
    # for i in range(1):
    it_str = "4x4_10steps_"# + str(i)
    train_q_table(it_str, True, False)

    # model = build_lstm([4, 5, 10, 5])
    # load_file_str = "reinforcement_model_" + it_str + ".h5"
    # print("Loading model: ", load_file_str)
    # model = load_model(load_file_str)
    # it_str = "15_2000_x" + str(1+i)
    # save_file_str = "reinforcement_model_" + it_str + ".h5"
    # print("Saving model: ", save_file_str)
    # model.save(save_file_str)

    print('Total duration (s) : ', time.time() - global_start_time)
