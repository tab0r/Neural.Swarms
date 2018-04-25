# utilities
import numpy as np
import pickle, os, sys
sys.path.append(os.path.abspath("../../../Python.Swarms/Source"))
from navi_game import NaviStrategy

class Q_Learner(NaviStrategy):
    def __init__(self, goal):
        # Q learner
        self.q_table = dict()
        self.online_learning = False
        self.pixel_input_mode = False
        self.reset_memory()
        NaviStrategy.__init__(self, goal)

    def reset_memory(self):
        self.memory = {"inputs": [], "choices": [], "rewards": []}
        self.epsilon = 0.01

    def get_reward(self, reward):
        self.memory['rewards'].append(reward)

    def step(self, choice = None):
        # make predictions
        # get this frames input
        s_prime, dist = self.get_input()
        # lookup the quality values
        if str(s_prime) in self.q_table.keys(): # check if we've seen this state
            quality = self.q_table[str(s_prime)] # grab the values
        else: # if not, generate some random numbers with mean zero
            quality = 2*np.random.random((1,5)) - 1

        # perform training step
        if (self.online_learning == True) and (len(self.memory['inputs']) > 0):
            s = self.memory['inputs'][-1]
            a = self.memory['choices'][-1] # these are -1 because the game gives rewards
            r = self.memory['rewards'][-1] # after the figure.step() has occured
            if str(s) in self.q_table.keys(): # check if we've seen this state
                target = self.q_table[str(s)] # grab the values
            else: # if not, generate some random numbers with mean zero
                target = np.random.random((1,5)) - 1
            gamma = 0.1
            target[0][a] = r + gamma * np.max(quality[0])
            self.q_table[str(s)] = target
        # store this frames' input
        self.memory['inputs'].append(s_prime)

        if choice == None:
            d = np.random.random()
            # explore some of the time
            if d < self.epsilon:
                choice = np.random.randint(0, 4)
            # exploit current Q-function
            else:
                choice = np.argmax(quality)

        e_choice = NaviStrategy.step(self, choice) # since the model may attempt
        # to wander off the board, or into the flag, e_choice may be different
        # from choice. maybe do something if so, but so far handle it well
        self.memory['choices'].append(choice) # store a'

class DeepQ(Q_Learner):
    def __init__(self, goal, model):
        # Deep-Q MLP
        self.model = model
        Q_Learner.__init__(self, goal)

    def step(self, choice = None):
        # get this frames input
        ipt, dist = self.get_input(pixel_ipt = self.pixel_input_mode)
        s_prime = np.array([ipt,])
        # make the quality prediction
        quality = self.model.predict(s_prime)

        # perform training step
        if (self.online_learning == True) and len(self.memory['inputs']) > 0:
            s = self.memory['inputs'][-1]
            a = self.memory['choices'][-1]
            r = self.memory['rewards'][-1]
            target = self.model.predict(s)
            gamma = 0.1 # hard coding discount at 0.1
            target[0][a] = r + gamma * np.max(quality[0])
            self.model.train_on_batch(s, target)
        # store this frames' input
        self.memory['inputs'].append(s_prime)

        if choice == None:
            d = np.random.random()
            # explore some of the time
            if d < self.epsilon:
                choice = np.random.randint(0, 4)
            # exploit current Q-function
            else:
                choice = np.argmax(quality)

        e_choice = NaviStrategy.step(self, choice) # since the model may attempt to wander off the board
        self.memory['choices'].append(choice) # store a'

class DeepQxr(DeepQ):
    def __init__(self, goal, model):
        # Deep-Q MLP with episodic memory and experience replay
        DeepQ.__init__(self, goal, model)
        self.lt_memory = []

    def step(self, choice = None):
        DeepQ.step(self, choice)
        # store memory when it's about to be erased
        # replace 5 with game length
        if len(self.memory['inputs']) % 5 == 0:
            self.lt_memory.append(self.memory)
        # experience replay
        if len(self.lt_memory) > 0:
            # experience replay
            v = np.random.randint(0, len(self.lt_memory))
            exp = self.lt_memory[v]
            for i in range(len(exp['inputs'])-1):
                s_prime = exp['inputs'][i+1]
                quality = self.model.predict(s_prime)
                s = exp['inputs'][i]
                a = exp['choices'][i]
                r = exp['rewards'][i]
                target = self.model.predict(s)
                gamma = 0.1 # hard coding discount at 0.1
                target[0][a] = r + gamma * np.max(quality[0])
                self.model.train_on_batch(s, target)

class LSTMDeepQ(Q_Learner):
    def __init__(self, goal, model):
        # Deep-Q LSTM
        self.model = model
        self.sequence_length = 5
        self.experience_replay = False
        Q_Learner.__init__(self, goal)

    def reset_memory(self):
        Q_Learner.reset_memory(self)
        self.memory['input_sequences'] = []
        self.memory['predicts']=[]

    def prep_sequences(self):
        # set up initial sequence values - just the agent sitting in
        # its starting position, accumulating rewards
        ipt, dist = self.get_input(pixel_ipt = self.pixel_input_mode)
        for _ in range(self.sequence_length):
            self.memory['inputs'].append(ipt)
        self.memory['choices'].append(4)
        if dist == 1:
            self.memory['rewards'].append(1)
        else:
            self.memory['rewards'].append(-0.1)
        self.memory['input_sequences'].append(np.array([self.memory['inputs'][-self.sequence_length:]],))
        q_0 = self.model.predict(self.memory['input_sequences'][-1])
        self.memory['predicts'].append(q_0)

    def placeIt(self):
        NaviStrategy.placeIt(self)
        self.prep_sequences()

    def step(self, choice = None):
        # get this frames input
        ipt, dist = self.get_input(pixel_ipt = self.pixel_input_mode)
        if len(self.memory['inputs']) == 0:
            self.prep_sequences()
        # store this frames' input
        self.memory['inputs'].append(ipt)
        s_prime = np.array([self.memory['inputs'][-self.sequence_length:]],)
        # make the quality prediction
        quality = self.model.predict(s_prime)

        # perform training step
        if (self.online_learning == True):
            s = self.memory['input_sequences'][-1]
            a = self.memory['choices'][-1]
            r = self.memory['rewards'][-1]
            target = self.memory['predicts'][-1]
            gamma = 0.1 # hard coding discount at 0.1
            target[0][a] = r + gamma * np.max(quality[0])
            self.model.train_on_batch(s, target)
        self.memory['input_sequences'].append(s_prime)
        self.memory['predicts'].append(quality)

        if choice == None:
            d = np.random.random()
            # explore some of the time
            if d < self.epsilon:
                choice = np.random.randint(0, 4)
            # exploit current Q-function
            else:
                choice = np.argmax(quality)

        e_choice = NaviStrategy.step(self, choice) # since the model may attempt to wander off the board
        self.memory['choices'].append(e_choice) # store a'

        # experience replay
        if self.experience_replay == True:
            # experience replay
            print("experience_replay - Not yet implemented")
            pass
