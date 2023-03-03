# sarsa implementation
import gym
from random import randint
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict # initiatlize:
import pprint

env = gym.make('FrozenLake-v1')
env.render()

# check to see if you can tune these values and how to tune them
alpha = 0.1
epsilon = 0.1
discount_rate = 0.9

Q = defaultdict(lambda: np.zeros(env.action_space.n))
# print(env.observation_space.n)
policy = defaultdict(lambda: 0)
state = defaultdict(lambda: 0)

n_episodes = 100

max_steps = 100

def choose_action_sarsa(state):
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

for i_episode in range(n_episodes):
    state =  env.reset()
    # if the action state action does not exist, we create it, else use the existing ones
    # if state not in policy:
    # for a in range(env.action_space.n):
    #         Q[(state, a)]
    action = choose_action_sarsa(state)
    count = 0
    while(True):
        # take action, observe reward and next state
        next_state, reward, end, probability = env.step(action)
        # choose the action for the next state as well using the policy from Q
        next_state_action = choose_action_sarsa(next_state)
        if(next_state == 15):
            reward = 1
        elif(next_state in [5, 7, 11, 12]):
            reward = -1
        else:
            reward = 0
        Q[(state, action)] = Q[(state, action)] + alpha * (reward + discount_rate * Q[(next_state, next_state_action)] - Q[(state, action)])
        count += 1
        if end or (count > max_steps):
            # print("Reached goal in steps: ", count)
            break
        state = next_state
        action = next_state_action
    
print(Q)