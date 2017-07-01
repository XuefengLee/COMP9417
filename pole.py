import gym
from gym import wrappers
import numpy as np
import random
import math
from time import sleep
from model import Model

import tensorflow as tf



ENVIRONMENT = 'CartPole-v0'
    #'MountainCar-v0'

MONITOR = False
UPLOAD = False
env = gym.make(ENVIRONMENT)
env1 = env.unwrapped

if(MONITOR):
    env = wrappers.Monitor(env, '/Users/smountain/Desktop/9417ASSN2/Monitor')


## Defining the simulation related constants
NUM_BUCKETS = (1, 1, 1, 1)  # (x, x', theta, theta')
NUM_ACTIONS = env.action_space.n
NUM_EPISODES = 1000
MAX_T = 250
NUM_TO_END = 100
SOLVED_T = 194
DEBUG_MODE = False

MIN_EXPLORE_RATE = 0.05
MIN_LEARNING_RATE = 0.01



def simulate(model):

    num_mem = 0
    num_streak = 0

    for episode in range(10000):

        # Reset the environment
        obv = env.reset()
        # the initial state

        for t in range(2000):
            env.render()

            # Select an action
            action = model.select_action(obv)

            # Execute the action, reward is not very helpful, so need to create our own
            next_obv, _, done, _ = env.step(action)
            x, x_dot, theta, theta_dot = next_obv

            # the smaller theta and closer to center the better
            r1 = (env1.x_threshold - abs(x)) / env1.x_threshold - 0.8
            r2 = (env1.theta_threshold_radians - abs(theta)) / env1.theta_threshold_radians - 0.5
            reward = r1 + r2

            model.store_transition(obv, action, reward, next_obv)

            if num_mem > 1000:
                model.learn()

            if done:
                print("Episode %d finished after %d time steps" % (episode, t))
                if (t >= SOLVED_T):
                    num_streak += 1
                    print(num_streak)
                else:
                    num_streak = 0
                break




            obv = next_obv
            num_mem += 1

        model.explore_rate = get_explore_rate(episode)
        model.learning_rate = get_learning_rate(episode)

        # solved condition may be changed in the future
        if num_streak > NUM_TO_END:
            break



    #close and upload
    env1.close()
    env.close()
    if(UPLOAD):
        gym.upload('/Users/smountain/Desktop/9417ASSN2/Monitor', api_key='sk_8xuCUXp9TOSW0DFIUgB12A')


def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(1, 1.0 - math.log10((t+1)/25)))

def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((t+1)/25)))




if __name__ == "__main__":
    ## Instantiating the learning related parameters
    model = Model(NUM_ACTIONS, len(NUM_BUCKETS), learning_rate=0.01, discount_rate=0.9, explore_rate=0.05)
    simulate(model)