## Experiment 0: Navigation with stationary "sensors"
- **Agent** Stationary "sensor" that can "see" a radius r- with the taxicab metric, mobile navigator.
- **Quantitiy** 1 neural agent, 1 dumb agent. Since this test will be done with "perfect" sensors, no need to train every tower. Only the 'navigator' is trained.
- **Training Method** Failed reinforcement on an MLP
- **Notes** The pieces all worked together: game backend, neural net, and my visualizations. But, everything got bloated and convoluted (not as in the nn...) so moving on to another experiment.
- **File**: `github.com/thetabor/Python.Swarms/swarm_game.py`
