# Neural.Swarms
Python swarm simulation with neural net based agents. Mainly focused on the NaviGame in my fork of [Python.Swarms](https://github.com/thetabor/Python.Swarms/), but the way I extend the game to allow a neural net is applicable to all of the games. I am exploring both supervised and reinforcement learning for the agent. For a nice visual display, see:

```
/NaviGame/Reinforcement Model Training.ipynb
/NaviGame/Supervised Model Training.ipynb
```

If you just want to train models or want to display somehow other than with a jupyter notebook, use these files:

```
/NaviGame/reinforcement_training.py
/NaviGame/supervised_training.py
```

The program uses python, jupyter, and keras with a theano back end. It will run on most Unix (Linux, Mac) but probably not directly on Windows.

It's coded with Python 3.6 on Mac OS X.

# Agents, Environments, and Reinforcement Learning

- **Agents**, in the context of machine learning, are a class of algorithms which make choices once deployed. These include everything from humble vacuum cleaners to stock-picking algorithms.
- **Supervised Learning** is the standard method of many statistical models and neural networks. It requires an (X, y) style training set, with inputs and desired outputs. For any kind of agent, this becomes a limitation, as the agent will be limited by the input data. That being said, we demonstrate supervised learning in the [Neural.Swarms](https://github.com/thetabor/Neural.Swarms) simulation. For the simple task of reaching a goal position in a deterministic environment, it performs very well after a short training period. We obtain data for this training from a deterministic strategy, so the neural network is limited to that performance level on the game. Simply adding a small barrier is enough to reduce the agents' performance.
- **Reinforcement Learning** is a method of training machine learning algorithms with rewards. Rather than learning from an (X, y) training set, it learns from experience. Each experience comes with certain rewards, and each time a reward is received, the algorithm can learn.
- **Deep-Q Networks** are a way to deploy reinforcement learning to neural networks. The network predicts Q-values for each action the network is allowed. A Q-value is a **quality** of a state, or the expected sum of rewards as we play the game from that state. We (almost) always select the max Q-value we predict.
- **RL Data:** Initially, the agent has absolutely no knowledge of the environment, so Q-values are effectively random. At each step, it updates the Q-value using the actual reward, plus the Q-value of the next step it plans on taking. So, our model fits (X, y) data, but each y is actually self-generated and often very inaccurate. But since a part of it is ground truth, the model eventually learns something close enough to real Q-values to function.

# Simulation

Using the [Neural.Swarms](https://github.com/thetabor/Neural.Swarms) simulation engine I can implement either supervised or reinforcement learning. Here we see how quickly the supervised learner can perform well on the simple task. In contrast, the reinforcement learner struggles to perform well, but it is does show potential. Here are some examples of simulation performance:

| Deterministic Strategy | Almost trained supervised model | Trained supervised | Typical supervised learning curve |
| --- | --- | --- | --- |
| ![Deterministic](https://github.com/thetabor/Neural.Swarms/blob/master/notes/gifs/deterministic_strategy_test.gif) | ![Almost trained](https://github.com/thetabor/Neural.Swarms/blob/master/notes/gifs/supervised/slight_undertrained_supervised.gif) | ![Fully trained](https://github.com/thetabor/Neural.Swarms/blob/master/notes/gifs/supervised/trained_supervised.gif) | ![Supervised learning curve](https://github.com/thetabor/Neural.Swarms/blob/master/notes/images/supervised_curve_0.png)|

The supervised network learns from the deterministic strategy on the left, and eventually learns to mimic it perfectly.

This also means that the supervised learner is limited by the strategy it learns from. So, enter *reinforcement learning*!

Reinforcement learning allows the agent to explore strategies on its own, and by receiving rewards from its environment, learns which are better. With reinforcement learning, I've struggled to get good results on the large grid, so I focused on a small game for now.

| RL early training | RL mid training | RL late training |
| --- | --- | --- |
| ![RL1](https://github.com/thetabor/Neural.Swarms/blob/master/notes/gifs/three_stages_rl/trained_guided_rl_1.gif) | ![RL2](https://github.com/thetabor/Neural.Swarms/blob/master/notes/gifs/three_stages_rl/trained_guided_rl_2.gif) | ![RL3](https://github.com/thetabor/Neural.Swarms/blob/master/notes/gifs/three_stages_rl/trained_guided_rl_3.gif) |

Above we see the progression of the agents learning. In the first, it had seen about 50,000 game steps. The next was an additional 500,000, and the final saw  another 1,000,000 steps. By contrast, the supervised learner above only required about 10,000 steps to achieve nearly-perfect imitation of the deterministic strategy. So why bother with reinforcement learning? I'll discuss once I have a chance to fully update this readme!

# References

- ![Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- Excellent materials in Georgia Tech's [Reinforcement Learning](https://www.udacity.com/course/reinforcement-learning--ud600) course on Udacity.
- [Keras Plays Catch](https://edersantana.github.io/articles/keras_rl/)
- [Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/)
- Nervanasys blog post (linked in Karpathy):
    - [Deep Reinforcement Learning](https://www.nervanasys.com/demystifying-deep-reinforcement-learning/)
