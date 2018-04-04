
## Python Swarm of Neural Units
**Goal:** develop a swarm intelligence model with agents which
1. Can navigate a local space using internal knowledge
2. Can update that knowledge using sensor data
3. Can navigate a larger space using swarm knowledge

**Development Pathways**

Acheiving all the goals above involves several interrelated development pathways. 
1. **Agent** 
development involves designing and training agents which can perform as desired in simulations.
2. **Swarm** 
development involves testing groups of agents in simulation, and showing how the swarm performs against individual units.
3. **Hardware** 
development entails preparing connectivity functions to collect data from hardware agents (Sphero robotic balls) and passing them into the swarm model for processing. To limit development requirements, the entire hardware swarm will be run virtually on a computer, and commands passed back to individual units. This will simulate each agent having its own intelligence without having to actually design a version of the software that is compatible with Sphero.

## Development Plan
**Simulation Stage**
1. Design agent
2. Test agent
3. Test swarm of agents
4. Repeat until goal #3 is acheived

**Deployment Stage**
1. Modify agent classes until they can communicate over Bluetooth to Sphero robotic balls
2. Test swarm of Spheros in basic environments
3. Two goals: 
    -individual navigation 
    -collective cohesion

**Experimentation Stage**
- In simulations, design more agent types and test swarms with different compositions
- In Sphero Swarm, design more task tests

**This Document**

This document records my research on swarms of units with neural network intelligence. Style guide
1. each experiment description should have a reproducable simulation
2. each conclusion should drive into the next experiment
3. mathematical formulations should be well-formed
4. break up discussion and computational cells
5. break discussion cells by paragraph

**Swarm Basics**

   For development, I focus on simulation. However, I may be able to deploy the project to a small swarm of Sphero robot balls. Deployment may or may not occur depending on how well development goes. I will use <a href="https://github.com/zdanowiczkonrad/PythonSphero">PythonSphero</a> which implements the <a href="https://github.com/karol-szuster/kulka">kulka</a> Sphero API. The swarm simulation is based on basic cellular automata. Rather than following formulated rules, behavior is governed by a neural network. I am using a base swarm class from github user <a href='https://github.com/elmar-hinz/'>elmar-hinz</a> called <a href='https://github.com/elmar-hinz/Python.Swarms'>Python.Swarms</a>. It was developed in Python 2.7 for visualization in a terminal class called curses. So, I have adapted it for Python 3 and display in Jupyter Notebooks. 

## Experiment 0: Navigation with stationary "sensors"
- **Agent** Stationary "sensor" that can "see" a radius r- with the taxicab metric, mobile navigator.
- **Quantitiy** 1 neural agent, 1 dumb agent. Since this test will be done with "perfect" sensors, no need to train every tower. Only the 'navigator' is trained.
- **Training Method** Failed reinforcement on an MLP
- **Notes** The pieces all worked together: game backend, neural net, and my visualizations. But, everything got bloated and convoluted (not as in the nn...) so moving on to another experiment.
- **File**: `github.com/thetabor/Python.Swarms/swarm_game.py`

## Experiment 1: Navigation with a vector
- **Agent** Navigator with an MLP policy evaluator. Input is a vector to target, output is an estimate of value of one of five actions. Navigator chooses highest value.
- **Quantitiy** 1 neural agent, 2 => hidden => 5
- **Training Method** Reinforcement/Q learning on an MLP
- **File**: `github.com/thetabor/Python.Swarms/navi_game.py`
- May 14, 2017 Notes:
    - Pieces in place, can train a model on one goal at a time.
    - Seems to kinda work? Very slowly. Now trying to train it a lot and see if I can get it moving faster.
    - Realized we can move up to an RNN by feeding the rewards back into it.
    - Also really need to implement a "game over" method, stop training on non-interesting states.
    - But is that relevant when it trains on a random path?
    - Was working with an input of (pos, goal), so that one model can play multiple games.
    - To contextualize the whole "game over" idea, the game is not over until the agent learns what it's supposed to do.
    - SO. The game is training the model.
    - And now only training a network with (pos) for input
    - I think I've just had my first major success! I must consider how to use this...
    - The network succesfully trains with goals at (7, 7) and (3, 3) on a 16 by 16 px board.
- Final Details
    - Three-layer ReLu MLP with two input neurons (for position)
    - 20-neuron hidden layer
    - Five output neurons (one for each action/value)
    - Convergence occured with 1000 game steps and 20 epochs
    - using SGD and a learning rate 0.005.
    - file: `navi_game1.py`


## Experiment 2: Navigation with position and goal
- **File** `navi_game2.py`
- **Agent** Navigator with an MLP policy evaluator. Input is a vector to target, output is an estimate of value of one of five actions. Navigator chooses highest value. The main difference between this and the previous experiment is that the goal is an input to our neural network. So, the model will be trained and tested on with multiple flag locations.
- **Quantitiy** 1 neural agent, 4 => hidden => hidden => 5
- **Training Method** Reinforcement/Q learning on an MLP
- May 14, 2017
    - Using the training regime discovered above to train the model on multiple games
    - With 100 games, the network had come close to convergence, and stopped at a point that was not the flag
    - With 500 games, similar results. Validating performance, I found that the model still played the very first game fine, but subsetquent games were terrible. So, back to 100 games, but different training regimes for the later ones vs the first one. Or maybe more games, shorter runs...? Yeah the latter first.
    - 5000 games, 100 steps each... ? Not much better
    - Added a layer, now running 50k games with 1000 steps each, so 5 million data points. We will see in the morning!
- May 15, 2017
    - So, that didn't work much better than some of the other shorter runs. Worth it, cuz it gave me a chance to sleep.
    - As that run most likely contained every game of this variation (there are only 256) we can safely say that the number of games isn't improving performance. We need to train on the data differently, and maybe use a different optimizer.
    - We'll use no more than 500 games (about two of each, 99% chance of having at least two thirds of the games)
    - We'll shuffle merge their training logs into a single log, and train on that
    - Slightly better results with the above, but still performing poorly on the "final boss" game.
    - Go deeper?
- May 16, 2017
    - Still struggling to achieve positive results. Some parameters to think about:
    - We are playing on a `16 x 16` board with only two pieces - the `Flag` and the `Navigator`
    - This gives 256 possible games, with 256 possible states
    - Each game is equally likely, and each initial state is equally likely
    - Any given initial state of a given game is equally likely, but the initial state determines what internal states can be seen, since we generate them by randomly making allowed moves
    -However, the reward generator is game-agnostic, meaning it can be called with a totally random `Navigator` and `Flag` position
    - Perhaps that is how we should generate data?
    - Also, avoid training on more than 100 games. As many states as desired from each game, but try to minimize the number of games so we can present the network with new games. It is critical that the agent be able to navigate to arbitrary positions, not just positions it trained to.
    - Moderate sucesses continue... Finding the benchmark goal but not necessarily the other goals. So, and explicit benchmark: **Train** on 255 games or less with any number of steps from each game. **Test** on the benchmark game, and find the goal quickly.
    - Then, **Train on 100 games** or less, and **Test** on all possible games. Be able to find the goal!

`import math`

`print("Probability of seeing all games in a cycle of 256 games: ",math.factorial(255)/(256**255))`

- May 17, 2017
    - That wee tiny number up there is the probability of seeing all 256 games in a cycle of playing 256 games.
    - I can train on any number of games, with any number of steps in each game, with a randomness factor in there, too.
    - So, I do not feel like my problem is bias or variance. It is lack of either. So... Train a deeper network on less data?
    - Still not much progress. I think the next step needs to be building two different versions of the game; one for training, one for testing
- May 18, 2017
    - Reviewing some material on reinforcement learning.
    - Things to implement:
      - Easy reward adjustment
      - Multi-game viewer
      - Online learning
    - Also, been working with five outputs- one for each direction and one for hold. Maybe I should set a threshold probability for movement, then only have four?
    - Time to build a new version of navi_game.
    - Separating the game from the model and model training entirely.
    - Since a winning strategy is relatively easy to define, the game should run just fine without a neural net.
    -k thx bai!


## Experiment 3: Navigation with position and goal continued
- **File** `reinforcement_training.py`
- **Agent** Navigator with an RNN Q-function. Input is a position, goal, and last reward, and output is an estimate of quality of one of five actions. Navigator chooses highest value. 
- **Quantitiy** 1 neural agent, 5 => hiddens => 5
- **Training Method** Reinforcement/Q learning on an RNN
- **Notes** Many improvements since last experiment in the codebase
- June 6, 2017
    - Uncertain results so far.
