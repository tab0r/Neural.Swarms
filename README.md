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

For detailed development notes, see
```
/notes/
```


- ** Supervised Learning ** is the standard method of many statistical models and neural networks. It requires an (X, y) style training set, with inputs and desired outputs. For any kind of agent, this becomes a limitation, as the agent will be limited by the input data. That being said, we demonstrate supervised learning in the [Neural.Swarms](https://github.com/thetabor/Neural.Swarms) simulation. For the simple task of reaching a goal position in a deterministic environment, it performs very well after a short training period. We obtain data for this training from a deterministic strategy.
- ** Reinforcement Learning ** is a method of training machine learning algorithms with rewards. Rather than learning from an (X, y) training set, it learns from experience. Each experience comes with certain rewards, and each time a reward is received, the algorithm can learn. A Deep-Q Network implements this method as neural network Q-estimator. A thorough explanation of the DQN and Q-estimation is available in the [references](https://www.nervanasys.com/demystifying-deep-reinforcement-learning/) above.
- ** RL Data: ** At each step of the game, the agent experiences something and learns from it. Exactly what it is learning requires some subtle ideas. The reward is a 'ground truth' on which the agent can based its understanding of the world. But, we're interested in more than short-term rewards. We'd like to encourage the agent into complex, intelligent behavior. So, it must look forward. Enter the Q-value. A Q-value is a ** quality ** of a state, or the expected sum of rewards as we play the game from that state. Initially, the agent has absolutely no knowledge of the environment, so Q-values are effectively random. At each step, it updates the Q-value using the actual reward, plus the Q-value of the next step it plans on taking. So, our model fits (X, y) data, but each y is actually self-generated and often very innacurate. But since a part of it is ground truth, the model eventually learns something close enough to real Q-values to function.
