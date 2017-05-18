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
