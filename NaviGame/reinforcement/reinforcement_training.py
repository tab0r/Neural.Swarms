# utilities
import numpy as np
import pylab as pl
import pickle, os, sys, pdb
from tqdm import *
from random import random, sample, randint
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.layers.core import Dense

# game imports
from ReinforcementNaviGame import ReinforcementNaviGame, ReinforcementStrategy
sys.path.append(os.path.abspath("../../../Python.Swarms/"))
from game_display_helper import make_gif
# constructs an MLP with inputs & outputs for different game modes, and whatever hidden layers you pass in with a dictionary
def build_model(optimizer = Adam(lr = 0.00001),
                    layers = [{"size":20,"activation":"relu"}],
                    inputs = 2, outputs = 5):
    # prepare the navigator model
    model = Sequential()
    # initial inputs
    l = list(layers)
    l0 = l[0]
    del l[0]
    model.add(Dense(l0['size'],
                input_dim = inputs,
                activation = l0['activation']))
    # the hidden layers
    for layer in l:
        model.add(Dense(layer['size'], activation=layer['activation']))
    # the output layer
    model.add(Dense(outputs, activation='tanh'))
    model.compile(optimizer = optimizer,
                    loss = "mean_squared_error")
    return model

# takes a game, a model, number of episodes and steps, and perhaps most importantly, the range for our epsilon-greedy training.
def train_model(game, model, episodes = 10, steps = 2,
                            e_start = 1, e_stop = 0):
    initial_e, final_e = e_start, e_stop
    # use 1 until we add stochasticity to the model
    gamma = 1
    e_delta = initial_e - final_e / episodes
    distances = []
    loss_log, replay_log, inputs, targets, rewards = [],[],[],[],[]
    desc = "Network training starting"
    # set up tqdm progress bar to display loss dynamically
    t = trange(episodes, desc=desc, leave=True)
    for j in t:
        # set up variables for episode
        e = initial_e - j * e_delta
        r_total_ep = 0
        # play through episode
        for i in range(steps):
            # feedforward pass which defines the next state
            # note that the e is an epsilon-greedy value,
            # which decreases as training carries on
            # choice = game.Navigator.strategy.plan_movement(e)
            input_i, quality_pred = game.Navigator.strategy.get_quality()
            # save network input data from above, which is our <s>
            _, dist = game.Navigator.strategy.get_input()
            distances.append(dist)

            # move to s'
            game.step()

            # update our Q[s,a] using the reward we get and
            # the quality prediction for our new state
            reward = game.Navigator.strategy.get_reward()
            choice = game.Navigator.strategy.last_choice
            quality = reward
            if dist > game.tolerance:
                quality += gamma * quality_pred.flatten()[choice]
            target_i = quality_pred
            n = len(quality_pred.flatten())
            target_i[0][choice] = quality
            target_i = target_i.reshape(1, n)
            # online learning
            loss = model.train_on_batch(input_i, target_i)

            # store data for experience replay learning
            inputs.append(input_i)
            targets.append(target_i)
            rewards.append(reward)
            loss_log.append(loss)

        # experience replays on a set random experience
        # they are chosen randomly
        for i in range(np.min([len(inputs), steps])):
            if len(inputs) == 1:
                experience = 0
            else:
                # lower_limit = int(0.75 * len(inputs))
                # experience = randint(lower_limit, max(1, len(inputs)-1))
                experience = randint(0, len(inputs)-1)
            replay_i = inputs[experience]
            replay_t = targets[experience]
            loss_replay = model.train_on_batch(replay_i, replay_t).flatten()[0]
            # replay_log.append(loss_replay)
        loss_str = '{0:.3g}'.format(loss_replay)
        desc = "Episode " + str(j) + ", Replay Loss: " + loss_str
        t.set_description(desc)
        t.refresh() # to update progress bar
        # shift goal and set last_reward to 0 so next episode is a "fresh game"
        # game.shift_goal()
        game.Navigator.strategy.shift()
        game.Navigator.strategy.last_reward = 0
    # housekeeping to return everything nicely
    output = dict()
    output['loss'] = loss_log
    # output['replays'] = replay_log
    output['experiences'] = [inputs, targets]
    output['distances'] = distances
    output['rewards'] = rewards
    return output

# plots loss, distance from goal, and rewards, meaned over three different intervals of episodes. the mean is necessary as the training is so noisy. i use three different intervals, factors of five apart, as it seemed to yield readable charts. lots of potential for improvement and customization here.
# currently doesn't work with output from the special training functions.
def plot_learning_info(output, game, training_episodes = 10000, steps = 10, title_str = None, file_str = None):
    # plot learning info
    training_game_size_y = game.board.height
    training_game_size_x = game.board.width
    steps = len(output['distances'])
    if title_str == None:
        title_str = "Learning Plots\n"
    f, axarr = pl.subplots(3, 1, figsize = (8, 10.5), dpi = 600)

    for n in [1000, 5000, 25000]:
        mean_step = n
        mean_rewards = []
        mean_dists = []
        mean_loss = []
        num_means = int(len(output['distances'])/mean_step/steps)
        steps_per_mean = steps*mean_step
        x = np.linspace(0, training_episodes, num_means)
        for i in range(num_means):
            mean_r = 0
            mean_d = 0
            mean_l = 0
            for j in range(steps_per_mean):
                mean_r += output['rewards'][j + i * steps_per_mean]
                mean_d += output['distances'][j + i * steps_per_mean]
                mean_l += output['loss'][j + i * steps_per_mean]
            mean_r = mean_r / steps_per_mean
            mean_d = mean_d / steps_per_mean
            mean_l = mean_l / steps_per_mean
            mean_rewards.append(mean_r)
            mean_dists.append(mean_d)
            mean_loss.append(mean_l)
        label = str(mean_step) + " Episodes"
        axarr[0].plot(x, mean_loss, label = label)
        axarr[1].plot(x, mean_dists, label = label)
        axarr[2].plot(x, mean_rewards, label = label)

    axarr[0].grid(True)
    axarr[0].set_title(title_str + 'Mean Loss')
    axarr[1].grid(True)
    axarr[1].set_title('Mean Distances from Goal')
    axarr[2].grid(True)
    axarr[2].set_title('Mean Rewards')
    f.subplots_adjust(hspace=0.2)

    if file_str == None:
        file_str = str(training_game_size_y) + "x" + str(training_game_size_x)
        file_str += "_" + str(training_episodes) + ".png"
    pl.legend()
    pl.plot()
    pl.savefig(file_str)
    pl.show()

# the next training function I write will need to randomly select from a variety of different obstacle situations, and train with them all.

if __name__=='__main__':
    # lets train a DQN model!
    # make the model
    layers = 2
    #int(input("How many hidden layers?\n"))
    hiddens = []
    for i in range(layers):
        neurons = 5
        #int(input("How many layer neurons?\n"))
        hiddens.append({"size":neurons,"activation":"relu"})
    # the baseline_model function takes a dictionary of hidden layers,
    # and sets up your input/output layers for the game
    # [{"size":100,"activation":"relu"}, {"size":100,"activation":"relu"}]
    # make an optimizer
    from keras.optimizers import sgd, Adam
    optimizer = Adam()
    optimizer_str = "Adam"
    # ipt_mode 3 gets the game screen as input, opt_mode 1 has a deterministic strategy as a valid choice
    model = build_model(optimizer, hiddens)
    # model.load_weights("your model")

    # set up the training game
    training_game_size_x = 19
    training_game_size_y = 13
    tol = 1.4
    #float(input("How much tolerance from goal?\n"))
    training_game = ReinforcementNaviGame(training_game_size_y,
                                    training_game_size_x,
                                    model,
                                    tolerance = tol,
                                    goal_idle = 5)

    # finish game setup with model in hand
    training_game.setup()
    training_game.Navigator.strategy.mode = 1

    training_episodes = int(input("How many episodes?\n"))
    steps = int(input("How many steps per episode?\n"))
    print("Ready to beging training")
    _ = input("Press enter to begin")
    # train the model
    output = train_model(game = training_game,
                    model = model,
                    episodes = training_episodes,
                    steps = steps,
                    e_start = 1,
                    e_stop = .5)

    # plot learning info
    title_str = str(training_game_size_y) + "x" + str(training_game_size_x) + " with "
    title_str += str(training_episodes) + " episodes, " + str(steps) + " steps per episode\n"
    # title_str += str(len(hiddens)) + " hidden layers, optimized with " +
    title_str += str(neurons) + " neurons in hidden layer, optimized with " + optimizer_str + "\n"

    it_str = str(0)
    print("Saving trained model...")
    model.save("reinforcement_model_" + file_str + it_str + ".h5")
    print("Saving training plot")
    make_gif(training_game, 100, file_str)
