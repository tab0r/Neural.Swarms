# Neural.Swarms
Python swarm simulation with neural net based agents. Mainly focused on the NaviGame in my fork of [Python.Swarms](https://github.com/thetabor/Python.Swarms/), but the way I extend the game to allow a neural net is applicable to all of the games. You will need that repo, so clone it down before continuing. I am exploring both supervised and reinforcement learning for the agent. For a nice visual display, see:

```
/NaviGame/Reinforcement Model Training.ipynb
/NaviGame/Supervised Model Training.ipynb
```

If you just want to train models or want to display somehow other than with a jupyter notebook, use these files:

```
/NaviGame/reinforcement_training.py
/NaviGame/supervised_training.py
```

# tl;dr

- To train a model, use on of the two .py files above
- Customize your neural net by passing ```baseline_model()``` a list of layer dictionaries in the format ```{"size": 100, "activation": 'ReLU'}```
- Use the ```train_model()``` method for training
- Supervised training is still fairly basic, the reinforcement training has a variety of training methods with walls, blocks, rooms and dividers
- With the right choice of methods in the right sequence, you should be able to train a model that can get through basic mazes.

# Known issues

- Figure placement can create recursion crashes
- Not enough comments anywhere

# Initial Goals

The goals are behavioral in nature, rather than statistic.
- **Navigation** : demonstrate reinforcement learning for simple navigation
- **Cooperation** : multiple agents move a target
- **Model Extension** : find a way to build a model which can be trained to arbitrary performance levels, then still train further on new environments.

# Agents, Environments, and Reinforcement Learning

- **Agents**, in the context of machine learning, are a class of algorithms which make choices once deployed. Agents may be anything from humble vacuum cleaners to stock-picking algorithms. We generally say the agent exists in an environment, be it virtual or physical.
- **Supervised Learning** is the standard method of many statistical models and neural networks. It requires an (X, y) style training set, with inputs and desired outputs. The training set becomes a limitation, as the agent will only perform as well as the data it learns from. For the simple task of reaching a goal position in a deterministic environment, it performs very well after a short training period. We obtain data for this training from a deterministic strategy, so the neural network is limited to that performance level on the game. Simply adding a small barrier is enough to completely halt the network's strategy.
- **Reinforcement Learning** allows us to train our algorithms with rewards. Rather than learning from an (X, y) training set, it learns from experience. Each experience comes with certain rewards, and each time a reward is received, the algorithm can learn.
- **Deep-Q Networks** are a way to deploy reinforcement learning to neural networks. The network predicts Q-values for each action the network is allowed. A Q-value is a **quality** of a state, or the expected sum of rewards as we play the game from that state. We (almost) always select the max Q-value we predict.
- **RL Data:** Initially, the agent has absolutely no knowledge of the environment, so Q-values are effectively random. At each step, it updates the Q-value using the actual reward, plus the Q-value of the next step it plans on taking. So, our model trains on (X, y) data, but each y is actually self-generated and often very inaccurate. But since a part of it is ground truth, the model eventually learns something close enough to real Q-values to function.

# Deterministics and Supervised Examples
Here we see how quickly the supervised learner can perform well on the simple task. In contrast, the reinforcement learner struggles to perform well, but it is does show potential. Here are some examples of simulation performance:

| Deterministic Strategy | Almost trained supervised model | Trained supervised | Typical supervised learning curve |
| --- | --- | --- | --- |
| ![Deterministic](https://github.com/thetabor/Neural.Swarms/blob/master/notes/gifs/deterministic_strategy_test.gif) | ![Almost trained](https://github.com/thetabor/Neural.Swarms/blob/master/notes/gifs/supervised/slight_undertrained_supervised.gif) | ![Fully trained](https://github.com/thetabor/Neural.Swarms/blob/master/notes/gifs/supervised/trained_supervised.gif) | ![Supervised learning curve](https://github.com/thetabor/Neural.Swarms/blob/master/notes/images/supervised_curve_0.png)|

The supervised network learns from the deterministic strategy on the left, and eventually learns to mimic it perfectly.

This also means that the supervised learner is limited by the strategy it learns from. So, enter *reinforcement learning*!

# Reinforcement Examples

Reinforcement learning allows the agent to explore strategies on its own, and by receiving rewards from its environment, learns which are better.

When the DQN agent is initialized, it's output values are effectively random numbers, and training is very susceptible to local minima. So, we train using an explore/exploit ratio that decreases throughout the training session. Typically, it starts at 0.9, and ends at 0.1. Additionally, I can make some of the choices come from our deterministic strategy, to focus training on the "correct" routes. Third, we know that our deterministic strategy works, so why not use it? And, finally, a tolerance function can make the game easier or harder, to let's start with an easier game, then make it harder once the agent is doing well.

With all this in mind, I built a new model. This model takes inputs as usual, the whole game screen. As outputs, it has the five usual outputs; up, down, left, right and stay, plus a new addition: use the deterministic strategy. So, for the simple games, all our DQN agent has to do is learn to always use the deterministic strategy. Once it learns this, then we can start exploring more complex problems. Meet Larry, the simple bundle of neurons:

| Break In | More Training | Trained with harder game | Non-optimal paths |
| --- | --- | --- | --- |
| ![Larry1](https://github.com/thetabor/Neural.Swarms/blob/master/NaviGame/larry/20000_x_15/larry_gif_00_20000x15.gif) | ![Larry2](https://github.com/thetabor/Neural.Swarms/blob/master/NaviGame/larry/20000_x_15/larry_gif_01_plus60000x5_a.gif) | ![img3](https://github.com/thetabor/Neural.Swarms/blob/master/NaviGame/larry/20000_x_15/larry_gif_07_plus_2x_60000x5_d.gif) | ![img4](https://github.com/thetabor/Neural.Swarms/blob/master/NaviGame/larry/20000_x_15/larry_gif_09_plus_3x_60000x5_b.gif) |

All this progress made me wonder if Larry could handle a challenge...

| And he ran away screaming... | Training hasn't helped yet | Still works on simple games |
| --- | --- | --- |
| ![Larry1](https://github.com/thetabor/Neural.Swarms/blob/master/NaviGame/larry/20000_x_15/larry_gif_11_plus_3x_60000x5_d.gif) | ![Larry2](https://github.com/thetabor/Neural.Swarms/blob/master/NaviGame/larry/20000_x_15/larry_gif_12_plus_3x_60000x5_e.gif) | ![Larry3](https://github.com/thetabor/Neural.Swarms/blob/master/NaviGame/larry/20000_x_15/larry_gif_14_larry_maze_d.gif)

To conclude, reinforcement learning clearly works, and leaves flexibility to function of new challenges. I've built new training systems for training with a variety of obstacles.

# References

- [Python.Swarms](https://github.com/elmar-hinz/Python.Swarms/)
- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- Excellent materials in Georgia Tech's [Reinforcement Learning](https://www.udacity.com/course/reinforcement-learning--ud600) course on Udacity.
- [Keras Plays Catch](https://edersantana.github.io/articles/keras_rl/)
- [Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/)
- Nervanasys blog post (linked in Karpathy):
    - [Deep Reinforcement Learning](https://www.nervanasys.com/demystifying-deep-reinforcement-learning/)
