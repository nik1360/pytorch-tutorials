from ddpg_agent import DDPGAgent
import gym
import numpy as np
import torch

env = gym.make('LunarLanderContinuous-v2')
device = "cuda" if torch.cuda.is_available() else "cpu"

agent = DDPGAgent(alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001, env=env,
    batch_size=64, layer1_size=400, layer2_size=300, n_actions=2)

np.random.seed(0)
agent.load_model()

score_history = []
for i in range(1000):
    done=False
    score = 0 
    obs = env.reset()
    while not done:
        act = agent.choose_action(obs)
        new_state, reward, done, info = env.step(act)
        
        agent.remember(obs, act, reward, new_state, int(done))
        #agent.learn()
        score += reward 
        obs = new_state
        env.render()
    score_history.append(score)
    print('episode ', i, 'score %.2f' %score,
        '100 game average %.2f' %np.mean(score_history[-100:]))
    
    #if i%25 ==0:
    #    agent.save_model()


