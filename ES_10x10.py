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

# policy
policy = defaultdict(lambda: randint(0,3)) # initialize random actions to the start initital policy, lambda: 0
state_action_returns = defaultdict(lambda: 0)
state_action_count = defaultdict(lambda: 0) # since later we need the average, we need to make sure that we know how many times the state was visited, so as not to overbias with it's high values
Q = defaultdict(lambda: np.zeros(env.action_space.n))
pp = pprint.PrettyPrinter(indent=4)

n_episodes = 10000

 # Reward schedule:
 #    - Reach goal(G): +1
 #    - Reach hole(H): -1
 #    - Reach frozen(F): 0
for i_episode in range(n_episodes):
    curr_state =  env.reset()
    start = True # for the starting taking a randoming action, to give us exploring starts, this so that whatever direction it takes, it can reach the goal
    state_action_returns_episode = [] # to track the Q(s,a) in each episode
    states_in_episode = [] # to track S in each episode
    while(True): # proceed until you reach you end goals
        if(start):
            action = np.random.choice([0,1,2,3])
            start=False
        else:
            action = policy[curr_state]
        next_state, reward, end, probability = env.step(action)
        if(env.desc[next_state//10][next_state%10] == b"G"):
            reward = 1
        elif(env.desc[next_state//10][next_state%10] == b"H"): # need to add the holes
            reward = -1
        else:
            reward = 0
        state_action_returns_episode.append(((curr_state, action),reward)) # to match our state_action_returns
        states_in_episode.append(curr_state)
        if end:
            if(i_episode >= (n_episodes - 2)):
                print(reward)
                print("reached end goal in: ", len(states_in_episode))
            break
        curr_state = next_state
        
# update the state-action pair values 
    for ((curr_state, action),reward) in state_action_returns_episode:
        first_occurence_idx = next(i for i,(s_a,r) in enumerate(state_action_returns_episode) if s_a==(curr_state,action))
        g = 0
        for event in state_action_returns_episode:
            # print(event)
            g += event[1] # at the first round rewards will be 0 as the goal is not reached and it's still exploring
        state_action_count[(curr_state, action)] += 1
        # if(i_episode > n_episodes - 2):
        #     print(state_action_returns[(curr_state, action)])
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
                
pp.pprint(Q)
# pp.pprint(policy)