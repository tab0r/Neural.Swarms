# utilities
import pandas as pd
import numpy as np
import pylab as pl
import pickle, os, sys, pdb
from collections import Counter
from tqdm import *

# game imports

# imports for neural net
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers.core import Dense
from keras.optimizers import sgd, RMSprop, Adagrad, Adadelta, Adam
import theano

# constructs an MLP with inputs & outputs for different game modes, and whatever hidden layers you pass in with a dictionary
def baseline_model(optimizer = Adam(lr = 0.00001),
                    layers = [{"size":20,"activation":"relu"}],
                    ipt_mode = 0, opt_mode = 0):
    # four inputs - each coordinate when we move the goal
    # two inputs for now, until it's doing reallly good at fixed goals
    # and now 81 inputs, for a pixel value test!
    # now we get 1204 - 1200 pixels for a 40x30 board, plus 4 inputs
    if ipt_mode == 0:
        inputs = 2
    elif ipt_mode == 1:
        inputs = 4
    elif ipt_mode == 2:
        inputs = (40, 30)
    elif ipt_mode == 3:
        inputs = 1204
    # inputs = 1204
    # six outputs - one for each action, and one to use det strategy
    if opt_mode == 1:
        num_outputs = 6
    else:
        num_outputs = 5
    # prepare the navigator model
    model = Sequential()
    # initial inputs
    l = list(layers)
    l0 = l[0]
    del l[0]
    model.add(Dense(l0['size'],
                input_dim = inputs,
                activation = l0['activation']))
    # add convolutions if mode == 2
    if ipt_mode == 2:
        model.add(Convolution2D(64, 3, 3, subsample=(1,1),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
        model.add(Activation('relu'))
    # the hidden layers
    for layer in l:
        model.add(Dense(layer['size'], activation=layer['activation']))
    # the output layer
    model.add(Dense(num_outputs, activation='tanh'))
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

# doesn't work, might not fix. grid search is nonsense in this context. assume you don't need a big network.
def grid_search_sketch():
    training_game_size = 9
    training_episodes = 1000
    steps = 10
    final_mean_loss, final_mean_dists, final_mean_reward = [], [], []
    for neurons in range(60, 120, 10):
        for layers in [2, 3, 4, 5, 10]:
            # lets train a DQN model!
            # make the model
            layer = {"size":neurons,"activation":"relu"}
            hiddens = [layer for i in range(layers)]    # make an optimizer
            from keras.optimizers import sgd, RMSprop, Adagrad, Adadelta, Adam
            # note to self: DON'T CHANGE THIS UNTIL YOU KNOW WE'RE LEARNING SOMETHING
            # optimizer = sgd(lr = 0.0001)
            # optimizer_str = "SGD"
            # optimizer = Adagrad()
            # optimizer_str = "Adagrad"
            # optimizer = RMSprop()
            # optimizer_str = "RMSprop"
            optimizer = Adadelta()
            optimizer_str = "Adadelta"
            # optimizer = Adam()
            # optimizer_str = "Adam"
            model = baseline_model(optimizer, hiddens)
            # prepare the game for training model
            training_game = ReinforcementNaviGame(training_game_size,
                                            training_game_size,
                                            model)
            training_game.setup()

            output = train_model(game = training_game,
                            model = model,
                            episodes = training_episodes,
                            steps = steps)

            # plot learning info
            title_str = str(training_game_size) + "x" + str(training_game_size) + " with "
            title_str += str(training_episodes) + " episodes, " + str(steps) + " steps per episode, & "
            title_str += str(len(hiddens)) + " hidden layers, optimized with " + optimizer_str + "\n"
            f, axarr = pl.subplots(3, 1, figsize = (10, 15), dpi = 300)
            # f.canvas.set_window_title("RL Loss, 100 eps w/ 50 steps, Look: 20")
            mean_step = 10
            num_means = int(len(output['distances'])/mean_step/steps)

            for _, k in enumerate([5, 10, 100]):
                mean_step = k
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

            # axarr[1].plot(x, output['replays'])
            # axarr[1].set_title('Replay Loss')
            # axarr[2].plot(x2, output['reward_totals'])
            # axarr[2].set_title('Total Reward')
            # axarr[2].plot(x2, output['distances'])
            # axarr[2].set_title('Distance from Goal')

            file_str = str(training_game_size) + "x" + str(training_game_size) + "_"
            file_str += str(training_episodes) + "_" + str(steps) + "_" + str(len(hiddens))
            file_str += "_" + str(neurons) + "_neurons_"+ optimizer_str
            pl.legend()
            pl.plot()
            pl.savefig("rl_plots" + file_str + ".png")
            final_mean_reward.append(mean_rewards.pop())
            final_mean_dists.append(mean_dists.pop())
            final_mean_loss.append(mean_loss.pop())
        # pl.show()
    f, axarr = pl.subplots(1, 1, figsize = (10, 5), dpi = 300)
    x = np.linspace(0, layers, layers)
    axarr.plot(x, final_mean_reward, label = "Final mean rewards")
    axarr.plot(x, final_mean_dists, label = "Final mean distances")
    axarr.plot(x, final_mean_loss, label = "Final mean loss")
    axarr.set_title("Network Grid Test")
    pl.legend()
    pl.savefig("rl_plots_network_test.png")
    pl.close()

# plots loss, distance from goal, and rewards, meaned over three different intervals of episodes. the mean is necessary as the training is so noisy. i use three different intervals, factors of five apart, as it seemed to yield readable charts. lots of potential for improvement and customization here.
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

# these methods construct new training games, and pass them to train_model with the model you pass in.
# lots of improvements to be made here... including some I made and accidentally deleted.... so that's a thing.
# this one places random blocks into the game
def train_with_blocks(model, episodes, steps, gamecount, blockcount):
    training_game_size_x = 40
    training_game_size_y = 30
    episodes_per_game = int(episodes/gamecount)
    outputs = []
    for i in range(gamecount):
        game = HybridNaviGame(training_game_size_y,
                                training_game_size_x,
                                model,
                                tolerance = 3)
        game.setup()
        game.Navigator.strategy.mode = 3
        blocks = []
        for _ in range(blockcount):
            blocks.append(game.add_block())
        outputs.append(train_model(game = game,
                model = model,
                episodes = episodes_per_game,
                steps = steps,
                e_start = .9,
                e_stop = .1))
        draw_game(game, save = True, filename = "training_game" + str(i) + ".png")
        del game
        model.save("block_training_backup.h5")
    return outputs

# this one places two walls on either side of the target, with a random length and width
def train_with_channel(model, episodes, steps, gamecount):
    training_game_size_x = 40
    training_game_size_y = 30
    episodes_per_game = int(episodes/gamecount)
    outputs = []
    for i in range(gamecount):
        game = HybridNaviGame(training_game_size_y,
                                training_game_size_x,
                                model,
                                tolerance = 3)
        game.setup()
        game.Navigator.strategy.mode = 3
        length = randint(8, 14) * 2
        width = randint(1, 5) * 2
        step_1 = (1, 0)
        start_1 = (15 - int(0.5*length), 20 - int(0.5*width))
        start_2 = (15 - int(0.5*length), 20 + int(0.5*width))
        # ish
        game.add_wall(length = length, start = start_1, step = step_1)
        game.add_wall(length = length, start = start_2, step = step_1)

        outputs.append(train_model(game = game,
                model = model,
                episodes = episodes_per_game,
                steps = steps,
                e_start = .9,
                e_stop = .1))
        draw_game(game, save = True, filename = "training_game" + str(i) + ".png")
        del game
        model.save("wall_training_backup.h5")
    return outputs

# the next training function I write will need to randomly select from a variety of different obstacle situations, and train with them all.

if __name__=='__main__':
    # lets train a DQN model!
    # make the model
    print("If you are running this on a machine with GPU, and didn't use flags, abort now and restart with: \n")
    print("THEANO_FLAGS=device=gpu,floatX=float32 python this_file.py\n")
    print("But that's kinda a lie, cuz this code is a lil buggy and every time I try to do that on AWS it explodes. I don't own a machine with a GPU, so I've been running it on compute-optimized AWS nodes for long runs. That said, my best models were trained in under 2 hours on a 2016 MacBook.")
    print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ")
    neurons = int(input("How many hidden layer neurons?\n"))
    hiddens = [{"size":neurons,"activation":"relu"}]
    # the baseline_model function takes a dictionary of hidden layers,
    # and sets up your input/output layers for the game
    # [{"size":100,"activation":"relu"}, {"size":100,"activation":"relu"}]
    # make an optimizer
    from keras.optimizers import sgd, RMSprop, Adagrad, Adadelta, Adam
    # note: DON'T CHANGE THIS UNTIL YOU KNOW YOUR MODEL LEARNS SOMETHING
    # optimizer = sgd(lr = 0.0001)
    # optimizer_str = "SGD"
    # optimizer = Adagrad()
    # optimizer_str = "Adagrad"
    # optimizer = RMSprop()
    # optimizer_str = "RMSprop"
    # optimizer = Adadelta()
    # optimizer_str = "Adadelta"
    # seriously, Adam is magical, I don't really understand it but just use it
    optimizer = Adam()
    optimizer_str = "Adam"
    # ipt_mode 3 gets the game screen as input, opt_mode 1 has a deterministic strategy as a valid choice
    model = baseline_model(optimizer, hiddens, ipt_mode = 3, opt_mode = 1)
    # this probably won't work
    # model = load_model("guided_rl_model_wide.h5")
    # this probably will work
    # model.load_weights("your model")

    # set up the training game
    training_game_size_x = 40
    training_game_size_y = 30

    training_game = HybridNaviGame(training_game_size_y,
                                    training_game_size_x,
                                    model,
                                    tolerance = 5)
    training_game.setup()
    training_game.Navigator.strategy.mode = 3

    training_episodes = int(input("How many episodes?\n"))
    steps = int(input("How many steps per episode?\n"))
    print("Ready to beging training")
    _ = input("Press enter to begin")
    # train the model
    output = train_model(game = training_game,
                    model = model,
                    episodes = training_episodes,
                    steps = steps,
                    e_start = .9,
                    e_stop = .1)

    # plot learning info
    title_str = str(training_game_size_y) + "x" + str(training_game_size_x) + " with "
    title_str += str(training_episodes) + " episodes, " + str(steps) + " steps per episode\n"
    # title_str += str(len(hiddens)) + " hidden layers, optimized with " +
    title_str += str(neurons) + " neurons in hidden layer, optimized with " + optimizer_str + "\n"
    f, axarr = pl.subplots(3, 1, figsize = (8, 10.5), dpi = 600)

    base = int(training_episodes/1000)
    for _, k in enumerate([base, 5*base, 25*base]):
        mean_step = k
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

    file_str = str(training_game_size_y) + "x" + str(training_game_size_x) + "_"
    file_str += str(training_episodes) + "_" + str(steps) + "_" + str(neurons)
    #  str(len(hiddens))
    file_str += "_" + optimizer_str
    pl.legend()
    pl.plot()
    it_str = str(0)
    print("Saving trained model...")
    model.save("guided_rl_model_" + file_str + it_str + ".h5")
    print("Saving training plot")
    pl.savefig("hybrid_plots" + file_str + it_str + ".png")
