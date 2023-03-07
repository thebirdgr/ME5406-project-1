# Q-learning
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
# epsilon = 0.8
discount_rate = 0.9

Q_list = []
# Q = defaultdict(lambda: np.zeros(env.action_space.n))
# print(env.observation_space.n)
policy = defaultdict(lambda: 0)
state = defaultdict(lambda: 0)

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

steps_needed = []

Q = defaultdict(lambda: {"a": 0, "c": 0}) # action value and the count
decayX = -0.0001
n_episodes = 10000
epsilon = 1.0
MINIMUM_EPSILON = 0.0
REWARD_TARGET = 7 # reach goal in 50 steps
STEPS_TO_TAKE = REWARD_TARGET
REWARD_INCREMENT = 0.1
REWARD_THRESHOLD = 0
EPSILON_DELTA = (epsilon - MINIMUM_EPSILON)/STEPS_TO_TAKE

    
for i_episode in tqdm(range(n_episodes)):
        state = env.reset()
        count = 0
        total_reward = 0
        while(True):
            # choose A from S using policy derived from Q
            action =  choose_action_q_learning(Q, state, epsilon)
            # take action A, observe reward and next state
            next_state, reward, end, probability = env.step(action)
            count += 1 # steps
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
            Q[(state, action)]["a"] = Q[(state, action)]["a"] + alpha * (total_reward + discount_rate * Q[(next_state, next_state_max_action)]["a"] - Q[(state, action)]["a"])
            Q[(state, action)]["c"] += 1 # number of time the state action was visited                         
            if end: # don't include max_steps first
                # print("Reaching steps in: ", count)
                if(env.desc[next_state//size][next_state%size] == b"G"):
                    steps_per_episode_goal.append(count)
                    # steps_per_episode_goal.append((i_episode, count))
                    # print("goal")
                else:
                    steps_per_episode.append((i_episode, count))
                break
            state = next_state

        epsilon = epsilon + decayX # works for 10x 10

        # if epsilon > MINIMUM_EPSILON and reward >= REWARD_THRESHOLD:    
        #         epsilon = epsilon - EPSILON_DELTA    # lower the epsilon
        #         REWARD_THRESHOLD = REWARD_THRESHOLD + REWARD_INCREMENT
        
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



        # end_list.append(steps)
        # print("Steps to reach goal: ", steps)
    # print(len(end_list))
    # y = [np.polyval(curve, p) for p in end_list]
    # axis[r//subplot_size, r%subplot_size].hist(steps_end, color = 'g', orientation='horizontal')
txt = f'Times reached goal vs times failed ratio: {len(steps_goal)/(len(steps_end)+len(steps_goal))}'
print(txt)
# print(steps_goal)

# m1 = np.array(steps_goal).mean(axis=0)
# st1 = np.array(steps_goal).std(axis=0)
# fig, ax = plt.subplots()
# bp = ax.boxplot(steps_goal, showmeans=True)

# for i, line in enumerate(bp['medians']):
#     x, y = line.get_xydata()[1]
#     text = f'μ={m1}\n σ={st1}'
#     ax.annotate(text, xy=(x, y))

# # plt.boxplot(steps_goal, showmeans=True)
# plt.xlabel("Number of Episodes")
# plt.ylabel("Number of Steps needed to reach Goal")
# # plt.title("4x4 Q-learning without Epsilon Decay")
# # text = f'Number of times reached goal during training {n_episodes} episodes: {len(steps_needed)}'
# # plt.figtext(0.5, 0.06, text, wrap=True, horizontalalignment='center', fontsize=12)
# # plt.savefig('ql-4-without-epsilon-decay.png')
# plt.show()

plt.plot(*zip(*steps_needed))
plt.xlabel("Number of Episodes")
plt.ylabel("Number of Steps needed to reach Goal")
plt.title("4x4 Sarsa with RBED")
text = f'Number of times reached goal during training {n_episodes} episodes: {len(steps_needed)}'
plt.figtext(0.5, 0.06, text, wrap=True, horizontalalignment='center', fontsize=24)
# plt.savefig('sarsa-4-peculiar-with-epislon-decay.png')
plt.show()