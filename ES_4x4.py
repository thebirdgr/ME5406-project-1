import gym
from gym import wrappers
from gym import envs
import pprint
import plotting
from tqdm import tqdm
from random import randint
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict# initiatlize:
from helper import *

env = gym.make('FrozenLake-v1')
# , desc=generate_random_map(size=10, p = 0.75)
pp = pprint.PrettyPrinter(indent=4)

env.render()

# policy
Q = defaultdict(lambda: 0)
policy = defaultdict(lambda: randint(0,3)) # initialize random actions to the start initital policy, lambda: 0
state_action_returns = defaultdict(lambda: [])
state_action_count = defaultdict(lambda: 0) # since later we need the average, we need to make sure that we know how many times the state was visited, so as not to overbias with it's high values
size = int(math.sqrt(env.observation_space.n))
n_episodes = 10000

 # Reward schedule:
 #    - Reach goal(G): +1
 #    - Reach hole(H): -1
 #    - Reach frozen(F): 0

# def choose_action_es(policy, state, decay):
    
#     if np.random.random() < decay:
#         return env.action_space.sample() # lesser explored
#     else:
#         return policy[state] # exploitation
    
def choose_action_es(policy, state, decay):
    no_actions = env.action_space.n
    # these are probabilities
    es1 = 1 - decay + (decay / no_actions)
    es2 = decay / no_actions
    if es1 > es2:
        if np.random.random() < es2:
            return env.action_space.sample()
        else:
            return policy[state]
    else:
        if np.random.random() < es1:
            return env.action_space.sample()
        else:
            return policy[state]

decayX = -0.0001

# init_epsilon = 1

timesgoal = 0
gamma = 0.9

epsilon = 1.0
MINIMUM_EPSILON = 0.0
REWARD_TARGET = 7 # reach goal in 7 steps
STEPS_TO_TAKE = REWARD_TARGET
REWARD_INCREMENT = 0.1
REWARD_THRESHOLD = 0
EPSILON_DELTA = (epsilon - MINIMUM_EPSILON)/STEPS_TO_TAKE

steps_needed = []

for i_episode in tqdm(range(n_episodes)):
    state =  env.reset()
    start = True # for the starting taking a randoming action, to give us exploring starts, this so that whatever direction it takes, it can reach the goal
    state_action_returns_episode = [] # to track the Q(s,a) in each episode
    states_in_episode = [] # to track S in each episode
    episode = []
    count  = 0
    total_reward = 0
    while(True): # proceed until you reach you end goals
        # action = env.action_space.sample()
        # 3
        action = choose_action_es(policy, state, epsilon)

        next_state, reward, end, probability = env.step(action)
        count += 1
        if(env.desc[next_state//size][next_state%size] == b"G"):
            reward = 1
            steps_needed.append((i_episode, count))
        elif(env.desc[next_state//size][next_state%size] == b"H"): # need to add the holes
            reward = -1
        else:
            reward = 0
        total_reward += reward
        episode.append(((state, action),total_reward))
        if end:
            break
        state = next_state
    # epsilon = epsilon + decayX
    if epsilon > MINIMUM_EPSILON and reward >= REWARD_THRESHOLD:    
                epsilon = epsilon - EPSILON_DELTA    # lower the epsilon
                REWARD_THRESHOLD = REWARD_THRESHOLD + REWARD_INCREMENT
    # update the state-action pair values 
    g = 0
    for ((curr_state, action),reward) in episode:
        g = gamma*g + reward
        state_action_returns[(curr_state, action)].append(g)
        Q[(curr_state, action)] = np.average(state_action_returns[(curr_state, action)])
        max_action = env.action_space.sample()
        for action in range(env.action_space.n):
            if Q[(state, max_action)] < Q[(state, action)]:
                max_action = action
        policy[curr_state] = max_action

steps_goal = []
steps_end = []
ratio = []
for i in range(1000):
    state = env.reset()
    steps = 0
    size = int(math.sqrt(env.observation_space.n))
    done = False
    while not done:
        action = policy[state]

        next_state, reward, done, info = env.step(action)
        
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
    ratio.append(len(steps_goal)/(len(steps_goal) + len(steps_end)))       

# Plotting
txt = f'Evaluation Success Rate: {len(steps_goal)/(len(steps_end)+len(steps_goal))}'
plt.rcParams["figure.figsize"] = (30,20)

# bar plot
title = "4x4 MCWES with Epsilon Decay"
counts, edges, bars = plt.hist(steps_goal, color = 'r', rwidth=0.7)
plt.bar_label(bars)
plt.title(f'Without Decay')
plt.axis(xmin=0,xmax=100)
plt.xlabel("Steps Taken to Reach Goal", fontsize=20)
plt.ylabel("Success Count", fontsize=20)
plt.title(f'{title} - Evaluation', fontsize=24)
plt.figtext(0.5, 0.03, txt, wrap=True, horizontalalignment='center', fontsize=20)
# plt.savefig('./Graphs/es-4-evaluation-rbed.png')
plt.figure()

# Training Plot
plt.plot(*zip(*steps_needed))
plt.xlabel("Number of Episodes", fontsize=20)
plt.ylabel("Number of Steps needed to reach Goal", fontsize=20)
plt.title(f'{title} - Training')
t = f'Training Success Rate: {len(steps_needed)/n_episodes}'
text = f'Number of times reached goal during training {n_episodes} episodes: {len(steps_needed)}\n {t}'
plt.figtext(0.5, 0.03, text, wrap=True, horizontalalignment='center', fontsize=20)

# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.savefig('./Graphs/es-4-training-rbed.png')
plt.show()
    