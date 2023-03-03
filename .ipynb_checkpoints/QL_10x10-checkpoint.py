# Q-learning
import gym
from random import randint
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict # initiatlize:
from helper import *
import pprint

env = gym.make('FrozenLake-v1', desc=generate_random_map(size=10, p = 0.75))
env.render()
pp = pprint.PrettyPrinter(indent=10)


# check to see if you can tune these values and how to tune them
alpha = 0.1
epsilon = 0.1
discount_rate = 0.9

Q = defaultdict(lambda: np.zeros(env.action_space.n))
# print(env.observation_space.n)
policy = defaultdict(lambda: 0)
state = defaultdict(lambda: 0)

def choose_action_q_learning(state):
    action = 0 # initiate arbitary action
    p = np.random.random()
    if p < epsilon:
        # print('ep')
        action = env.action_space.sample()
    else:
        # print("max")
        action = np.argmax(Q[(state, action)])
        # action = np.argmax((st, act) for )
    return action

n_episodes = 50000

max_steps = 200

for i_episode in range(n_episodes):
    state = env.reset()
    count = 0
    while(True):
        # choose A from S using policy derived from Q
        action =  choose_action_q_learning(state)
        # take action A, observe reward and next state
        next_state, reward, end, probability = env.step(action)
        count += 1
        if(next_state == 15):
            reward = 1
        elif(next_state in [5, 7, 11, 12]):
            reward = -1
        else:
            reward = 0
        Q[(state, action)] = Q[(state, action)] + alpha * (reward + discount_rate * np.argmax(Q[(next_state, action)]) - Q[(state, action)])
        if(i_episode % 10000 == 0):
            print(i_episode)
        if end or (count > max_steps):
            # print("Reaching steps in: ", count)
            break
        state = next_state
        
        
pp.pprint(Q)