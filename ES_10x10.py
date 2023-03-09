import gym
from gym import wrappers
from gym import envs
import pprint
# import plotting
from tqdm import tqdm
from random import randint
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict# initiatlize:
from helper import *
import csv
# f = open('./csv/es-10-rewards-decay.csv', 'w')
# writer = csv.writer(f)

env = gym.make('FrozenLake-v1', desc=generate_random_map(size=10, p = 0.75))
env.reset()
env.render()

pp = pprint.PrettyPrinter(indent=10)

# policy
policy = defaultdict(lambda: randint(0,3)) # initialize random actions to the start initital policy, lambda: 0
state_action_returns = defaultdict(lambda: 0)
state_action_count = defaultdict(lambda: 0) # since later we need the average, we need to make sure that we know how many times the state was visited, so as not to overbias with it's high values
Q = defaultdict(lambda: np.zeros(env.action_space.n))
size = int(math.sqrt(env.observation_space.n))
n_episodes = 100000

 # Reward schedule:
 #    - Reach goal(G): +1
 #    - Reach hole(H): -1
 #    - Reach frozen(F): 0

def choose_action_es(policy, state, decay):
    
    if np.random.random() < decay:
        return env.action_space.sample() # lesser explored
    else:
        return policy[state] # exploitation

decayX = -0.00001

# init_epsilon = 1

timesgoal = 0

epsilon = 1.0
MINIMUM_EPSILON = 0.0
REWARD_TARGET = 50 # reach goal in 7 steps
STEPS_TO_TAKE = REWARD_TARGET
REWARD_INCREMENT = 0.1
REWARD_THRESHOLD = 0
EPSILON_DELTA = (epsilon - MINIMUM_EPSILON)/STEPS_TO_TAKE

steps_needed = []

rewards_list = []
tr = 0   

for i_episode in tqdm(range(n_episodes)):
    state =  env.reset()
    state = state[0]
    start = True # for the starting taking a randoming action, to give us exploring starts, this so that whatever direction it takes, it can reach the goal
    state_action_returns_episode = [] # to track the Q(s,a) in each episode
    states_in_episode = [] # to track S in each episode
    count  = 0
    total_reward = 0
    while(True): # proceed until you reach you end goals
        # 1
        # if(start):
        #     action = np.random.choice([0,1,2,3])
        #     start=False
        # else:
        #     action = policy[curr_state]
        # 2
        
        action = choose_action_es(policy, state, epsilon)
        # 3
        
        next_state, reward, end, trunc, info = env.step(action)
        count += 1
        if(env.desc[next_state//size][next_state%size] == b"G"):
            reward = 1
            steps_needed.append((i_episode, count))
        elif(env.desc[next_state//size][next_state%size] == b"H"): # need to add the holes
            reward = -1
        else:
            reward = 0
        total_reward += reward
        state_action_returns_episode.append(((state, action),reward)) # to match our state_action_returns
        states_in_episode.append(state)
        if end:
            if(i_episode > n_episodes - 5):
                # print(reward)
                print(timesgoal)
                # print("reached end goal in: ", len(states_in_episode))
            tr += total_reward
            rewards_list.append(tr)
            # writer.writerow([i_episode, tr])
            break
        state = next_state
    epsilon = epsilon + decayX
    # if epsilon > MINIMUM_EPSILON and reward >= REWARD_THRESHOLD:    
    #             epsilon = epsilon - EPSILON_DELTA    # lower the epsilon
    #             REWARD_THRESHOLD = REWARD_THRESHOLD + REWARD_INCREMENT
    # update the state-action pair values 
    for ((curr_state, action),reward) in state_action_returns_episode:
        first_occurence_idx = next(i for i,(s_a,r) in enumerate(state_action_returns_episode) if s_a==(curr_state,action))
        g = 0
        for event in state_action_returns_episode:
            # print(event)
            g += event[1] # at the first round rewards will be 0 as the goal is not reached and it's still exploring
        state_action_count[(curr_state, action)] += 1
        # if(i_episode > n_episodes - 5):
            # print(state_action_returns[(curr_state, action)])
        # new_val = (current_val * N + reward) / N + 1
        state_action_returns[(curr_state, action)] += (state_action_returns[(curr_state, action)] * state_action_count[(curr_state, action)] + g) / (state_action_count[(curr_state, action)] + 1)
        Q[curr_state][action] = state_action_returns[(curr_state, action)]
        
    for curr_state in states_in_episode:
        # get all the curr_state values
        curr_state_action_pairs = [(s,a) for ((s,a),r) in state_action_returns_episode if s==curr_state]
        maximum = 0
        for pair in curr_state_action_pairs:
            if Q[pair[0]][pair[1]] > maximum:
                maximum = Q[pair[0]][pair[1]]
                policy[curr_state] = pair[1]
                
# pp.pprint(steps_needed)
steps_goal = []
steps_end = []
ratio = []
for i in range(1000):
    state = env.reset()
    state = state[0]
    steps = 0
    size = int(math.sqrt(env.observation_space.n))
    done = False
    while not done:
        action = policy[state]

        next_state, reward, done, trunc, info = env.step(action)
        
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
    ratio.append(len(steps_goal)/len(steps_end))        
    
# Plotting
txt = f'Evaluation Success Rate: {len(steps_goal)/(len(steps_end)+len(steps_goal))}'
plt.rcParams["figure.figsize"] = (30,20)
print(txt)
plt.plot(rewards_list)

# bar plot
title = "10x10 ES without Epsilon Decay"
# counts, edges, bars = plt.hist(steps_goal, color = 'r', rwidth=0.7)
# plt.bar_label(bars)
# plt.axis(xmin=0,xmax=100)
# plt.xlabel("Steps Taken to Reach Goal", fontsize=20)
# plt.ylabel("Success Count", fontsize=20)
plt.title(f'{title} - Evaluation', fontsize=24)
# plt.figtext(0.5, 0.03, txt, wrap=True, horizontalalignment='center', fontsize=20)
# # plt.savefig('./Graphs/es-10-evaluation.png')
# plt.figure()

# Training Plot
# plt.plot(*zip(*steps_needed))
plt.xlabel("Number of Episodes", fontsize=20)
plt.ylabel("Cumulative Rewards", fontsize=20)
# plt.title(f'{title} - Training')
# t = f'Training Success Rate: {len(steps_needed)/n_episodes}'
# text = f'Number of times reached goal during training {n_episodes} episodes: {len(steps_needed)}\n {t}'
# plt.figtext(0.5, 0.03, text, wrap=True, horizontalalignment='center', fontsize=20)

# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.savefig('./Graphs/es-10-rewards.png')
plt.show()