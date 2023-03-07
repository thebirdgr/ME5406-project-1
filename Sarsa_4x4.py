# sarsa implementation
import gym
from random import randint
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict # initiatlize:
from tqdm import tqdm
from helper import *

env = gym.make('FrozenLake-v1')
env.render()

# check to see if you can tune these values and how to tune them
alpha = 0.01
# epsilon = 1
discount_rate = 1.0
decayX = -0.0001
size = int(math.sqrt(env.observation_space.n))


Q = defaultdict(lambda: {"a": 0, "c": 0}) # action value and the count
# print(env.observation_space.n)
policy = defaultdict(lambda: 0)
state = defaultdict(lambda: 0)

n_episodes = 10000

max_steps = 100

def choose_action_sarsa(Q, state, decay):
    no_actions = env.action_space.n
    es1 = 1 - decay + (decay / no_actions)
    es2 = decay / no_actions
    max_action = env.action_space.sample()
    for action in range(env.action_space.n):
        if Q[(state, max_action)]["a"] < Q[(state, action)]["a"]:
            max_action = action
    if es1 > es2:
        if np.random.random() < es2:
            return env.action_space.sample()
        else:
            return max_action
    else:
        if np.random.random() < es1:
            return env.action_space.sample()
        else:
            return max_action
    
epsilon = 1.0
MINIMUM_EPSILON = 0.0
REWARD_TARGET = 7 # reach goal in 7 steps
STEPS_TO_TAKE = REWARD_TARGET
REWARD_INCREMENT = 0.1
REWARD_THRESHOLD = 0
EPSILON_DELTA = (epsilon - MINIMUM_EPSILON)/STEPS_TO_TAKE
state =  env.reset()
action = choose_action_sarsa(Q, state, epsilon)
steps_needed = []

for i_episode in tqdm(range(n_episodes)):
    state =  env.reset()
    total_reward = 0
    count = 0
    while(True):
        # take action, observe reward and next stateS
        next_state, reward, end, probability = env.step(action)
        # choose the action for the next state as well using the policy from Q
        next_state_action = choose_action_sarsa(Q, next_state, epsilon)
        
        if(env.desc[next_state//size][next_state%size] == b"G"):
            # print(len(steps_per_episode_goal))
            # print("Steps to reach goal: ", steps)
            steps_needed.append((i_episode, count))
            # steps_goal.append(steps)
            reward = 1
        elif(env.desc[next_state//size][next_state%size] == b"H"): # need to add the holes
            # print("Steps to reach hsoles: ", steps)
            # steps_end.append(steps)
            reward = -1
        else:
            reward = 0
        total_reward += reward
        Q[(state, action)]["a"] = Q[(state, action)]["a"] + alpha * (total_reward + discount_rate * Q[(next_state, next_state_action)]["a"] - Q[(state, action)]["a"])
        Q[(state, action)]["c"] += 1 # number of time the state action was visited              
        count += 1
        if end:
            # print("Reached goal in steps: ", count)
            break
        state = next_state
        action = next_state_action
        
    # epsilon = epsilon + decayX
    # print(epsilon)
    # if epsilon > MINIMUM_EPSILON and reward >= REWARD_THRESHOLD:    # works 10x10 100k reward target 25
    #         epsilon = epsilon - EPSILON_DELTA    # lower the epsilon
    #         REWARD_THRESHOLD = REWARD_THRESHOLD + REWARD_INCREMENT
    
# print(Q)

steps_goal = []
steps_end = []
for i in range(1000):
        state = env.reset()
        steps = 0
        size = int(math.sqrt(env.observation_space.n))
        done = False
        while not done:
            max_action = env.action_space.sample()
            for action in range(env.action_space.n):
                if Q[(state, max_action)]["a"] < Q[(state, action)]["a"]:
                    max_action = action

            next_state, reward, done, info = env.step(max_action)
            steps += 1
            if(env.desc[next_state//size][next_state%size] == b"G"):
                # print(len(steps_per_episode_goal))
                # print("Steps to reach goal: ", steps)
                steps_goal.append(steps)
                reward = 1
            elif(env.desc[next_state//size][next_state%size] == b"H"): # need to add the holes
                # print("Steps to reach hsoles: ", steps)
                steps_end.append(steps)
                reward = -1
            else:
                reward = 0
            state = next_state

txt = f'Times reached goal vs times failed ratio: {len(steps_goal)/(len(steps_end)+len(steps_goal)+0.001)}'
print(txt)

plt.plot(*zip(*steps_needed))
plt.xlabel("Number of Episodes")
plt.ylabel("Number of Steps needed to reach Goal")
plt.title("4x4 Sarsa with RBED")
text = f'Number of times reached goal during training {n_episodes} episodes: {len(steps_needed)}'
plt.figtext(0.5, 0.06, text, wrap=True, horizontalalignment='center', fontsize=24)
# plt.savefig('sarsa-4-peculiar-with-epislon-decay.png')
plt.show()