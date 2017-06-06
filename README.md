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
