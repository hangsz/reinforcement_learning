# coding: utf-8

__author__ = 'zhenhang.sun@gmail.com'
__version__ = '1.0.0'

import gym
from IPython import display
import matplotlib.pyplot as plt

from dqn import Agent

def plot(score, mean):
    
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.figure(figsize=(20,10))
    plt.clf()

    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(score)
    plt.plot(mean)
    plt.text(len(score)-1, score[-1], str(score[-1]))
    plt.text(len(mean)-1, mean[-1], str(mean[-1]))

if __name__ == '__main__':

    env = gym.make('CartPole-v0')

    params = {
        'gamma': 0.8,
        'epsi_high': 0.9,
        'epsi_low': 0.05,
        'decay': 200, 
        'lr': 0.001,
        'capacity': 10000,
        'batch_size': 64,
        'state_space_dim': env.observation_space.shape[0],
        'action_space_dim': env.action_space.n   
    }
    agent = Agent(**params)

    score = []
    mean = []

    for episode in range(1000):
        s0 = env.reset()
        total_reward = 1
        while True:
            env.render()
            a0 = agent.act(s0)
            s1, r1, done, _ = env.step(a0)
            
            if done:
                r1 = -1
                
            agent.put(s0, a0, r1, s1)
            
            if done:
                break

            total_reward += r1
            s0 = s1
            agent.learn()
            
        score.append(total_reward)
        mean.append( sum(score[-100:])/100)
        
        plot(score, mean)