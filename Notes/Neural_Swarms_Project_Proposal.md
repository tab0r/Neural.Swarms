
## Project Proposal: Neural.Swarms
Tabor Henderson, May 23, 2017


**High Level Description**

Through the problem of swarm intelligence, I hope to answer fundamental questions about neural networks. Swarm intelligence problems are problems of cooperation, in which a collective can do better than individual units, or even achieve things impossible for an individual unit. Sphero robotic balls, with their accurate sensors but inaccurate locomation make an ideal test bed.
- *First goal:* navigate as a swarm
- *Second goal:* map a room, scout
- *Third goal:* locate a movable object, move it cooperatively


**Presentations**
- The work will be demonstrated live
- Either a swarm simulation or an actual demonstration with Sphero toys
- I would like to give a brief presentation to outline the project


**Next Step**

I have a simulation engine in which I can build agents and deterministic strategies. Using this engine I have succesfully trained basic navigator neural agents which perform as well as the deterministic strategy. The next step is to implement reinforcement learning in the engine and hopefully train neural agents which outperform the deterministic strategy. Performance is measured case-by-case, as different simulations have different goals.


**Data**
- I generate data with the simulation engine.
- Using materials in Georgia Tech's [Reinforcement Learning](https://www.udacity.com/course/reinforcement-learning--ud600) course on Udacity.
- Referencing Karpathy's blog in two places:
    - [Keras Plays Catch](https://edersantana.github.io/articles/keras_rl/)
    - [Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/)


## Project Details
** Techniques **

- Neural networks
- Supervised learning
- Reinforcement learning

** Anticipated Bottlenecks **

- Simulation construction: I have simulations online, but fine tuning it to a state where it can model space will take substantial time. I am taking the approach of incremental developments. Currently, I could even freeze my simulation code and focus on studying the fundamental neural network questions mentioned above.
- Reinforcement learning implementation: So far, the trainer uses reinforcement from a deterministic strategy. This, of course, requires that strategy, and there's no reason to expect the network to outperform that data. So, I need to switch to reinforcement learning as soon as possible. This will allow me to implement a explore/exploit function in the model so it can discover more optimal strategies. I am going through the Udacity course as fast as possible, and hope to have it implemented within a week.
- Performance metrics: Since each simulation has slightly different metrics of success, I have no way to compare models yet. But, since the final goal is a robust swarm AI, it's not necessary to focus on metrics until I achieve basic successes on differing tasks with a single model. Then, models can be rated on categories of tasks, like cooperation, navigation, and memory.
- Technology deployment: I have sucessfully connected and sent commands to the Sphero robots, but have yet to set up sensor streaming. I will have to add the functionality to the Kulka module.
- Technology incompatibilities: The Bluetooth libraries I need to connect to the robots are not available on macOS, and I the Linux kernel does not support my model of MacBook. So, I'll need to test code on the Mac Minis or an old laptop my brother dug up for me.

** Time Frame **

- I have simulations running, and the ability to train neural nets from the simulation
- Each new simulation takes a few hours to build, and I've been iterating rapidly
- Need to build at least 3 - 5 new simulations before considering deployment
- I want to perform detailed analysis of network performance and training before deploying
- So hoping to consider deployment within two weeks

** Technology Used **
- Keras & Theano
- Python.Swarms
- Kulka
- Sphero

** Data Samples **
- Supervised: Inputs are two pairs of coordinates, one is the agents position, the second is the goal position. Targets are the action the deterministic strategy takes
    - Inputs: `[3, 4, 0, 3], [2, 4, 0, 3], [1, 4, 0, 3], [0, 4, 0, 3], [0, 4, 0, 3]`
    - Targets: `[0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]`
- Reinforcement: Not yet available, but I am working on adapting the code to provide rewards as targets, rather than the choices of a deterministic strategy.
