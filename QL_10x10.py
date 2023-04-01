# Q-learning
import gym
from random import randint
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict # initiatlize:
from tqdm import tqdm
from helper import *
import csv

# f = open('./csv/ql-10-rewards-decay.csv', 'w')
# writer = csv.writer(f)

env = gym.make('FrozenLake-v0', desc=generate_random_map(size=10, p = 0.75))
# env.reset()
env.render()

# check to see if you can tune these values and how to tune them
alpha = 0.1

discount_rate = 0.9

Q_list = []
policy = defaultdict(lambda: 0)
state = defaultdict(lambda: 0)

# epsilon soft-max greedy policy    
def choose_action_q_learning(Q, state, decay):
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

max_steps = 200

steps_per_episode = []
steps_per_episode_goal = []
size = int(math.sqrt(env.observation_space.n))


# Q_table to store (state, action) pair
Q = defaultdict(lambda: {"a": 0, "c": 0}) # action value and the count
decayX = -0.00001
n_episodes = 100000 # 10000 doesn't work
epsilon = 1.0

#information needed for Reward Based Epsilon Decay
MINIMUM_EPSILON = 0.0
REWARD_TARGET = 50 # reach goal in 50 steps
STEPS_TO_TAKE = REWARD_TARGET
REWARD_INCREMENT = 0.1
REWARD_THRESHOLD = 0
EPSILON_DELTA = (epsilon - MINIMUM_EPSILON)/STEPS_TO_TAKE

steps_needed = [] # steps needed to reach the goal per episode
rewards_list = [] # list of cumulative rewards
tr = 0 # total reward 

# Training
for i_episode in tqdm(range(n_episodes)):
        state = env.reset()
        # state = state[0]
        count = 0
        total_reward = 0
        while(True):
            # choose A from S using policy derived from Q
            action =  choose_action_q_learning(Q, state, epsilon)
            # take action A, observe reward and next state
            next_state, reward, end, info = env.step(action)
            count += 1 # steps
            # Reward schedule:
            #    - Reach goal(G): +1
            #    - Reach hole(H): -1
            #    - Reach frozen(F): 0
            if(env.desc[next_state//size][next_state%size] == b"G"):
                # if(n_episodes>750000):
                steps_needed.append((i_episode, count))
                # print(len(steps_per_episode_goal))
                reward = 1
            elif(env.desc[next_state//size][next_state%size] == b"H"): # need to add the holes
                reward = -1
            else:
                reward = 0
            total_reward += reward
            next_state_max_action = env.action_space.sample()
            for a in range(env.action_space.n):
                if Q[(next_state, next_state_max_action)]["a"] < Q[(next_state, a)]["a"]:
                    next_state_max_action = a
            Q[(state, action)]["a"] = Q[(state, action)]["a"] + alpha * (reward + discount_rate * Q[(next_state, next_state_max_action)]["a"] - Q[(state, action)]["a"])
            Q[(state, action)]["c"] += 1 # number of time the state action was visited                         
            if end: # don't include max_steps first
                # print("Reaching steps in: ", count)
                if(env.desc[next_state//size][next_state%size] == b"G"):
                    steps_per_episode_goal.append(count)
                    # steps_per_episode_goal.append((i_episode, count))
                    # print("goal")
                else:
                    steps_per_episode.append((i_episode, count))
                tr += total_reward
                rewards_list.append(tr)
                # writer.writerow([i_episode, tr])
                # print(total_reward)
                break
            state = next_state
        # ---------------- Uncomment This for Epsilon decay ----------------    
    
        # epsilon = epsilon + decayX

        # ---------------- Uncomment This for Reward Based Epsilon Decay----------------    

        # if epsilon > MINIMUM_EPSILON and reward >= REWARD_THRESHOLD:    
        #             epsilon = epsilon - EPSILON_DELTA    # lower the epsilon
        #             REWARD_THRESHOLD = REWARD_THRESHOLD + REWARD_INCREMENT


# Evaluation
steps_goal = []
steps_end = []
for i in range(1000):
        state = env.reset()
        # state = state[0]
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
        # ratio.append(len(steps_goal)/(len(steps_goal) + len(steps_end)))       



# Plotting
if(len(steps_end)+len(steps_goal) > 0):
    success_rate = len(steps_goal)/(len(steps_end)+len(steps_goal))
else:
    success_rate = 0.0
txt = f'Evaluation Success Rate: {success_rate}'
plt.rcParams["figure.figsize"] = (30,20)
print(txt)
# plt.plot(rewards_list)
# bar plot
title = "10x10 Q-Learnings without Epsilon Decay"
# counts, edges, bars = plt.hist(steps_goal, color = 'r', rwidth=0.7)
# plt.bar_label(bars)
# plt.axis(xmin=0,xmax=100)
# plt.xlabel("Steps Taken to Reach Goal", fontsize=20)
# plt.ylabel("Success Count", fontsize=20)
# plt.title(f'{title} - Evaluation', fontsize=24)
# plt.figtext(0.5, 0.03, txt, wrap=True, horizontalalignment='center', fontsize=20)
# # plt.savefig('./Graphs/ql-10-evaluation.png')
# plt.figure()

# Training Plot
plt.plot(*zip(*steps_needed))
plt.xlabel("Number of Episodes", fontsize=20)
plt.ylabel("Cumulative Rewards", fontsize=20)
plt.title(f'{title} - Training')
t = f'Training Success Rate: {len(steps_needed)/n_episodes}'
text = f'Number of times reached goal during training {n_episodes} episodes: {len(steps_needed)}\n {t}'
plt.figtext(0.5, 0.03, text, wrap=True, horizontalalignment='center', fontsize=20)

# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.savefig('./Graphs/ql-10-rewards.png')
plt.show()