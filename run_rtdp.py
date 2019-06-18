import numpy as np
from env_rtdp import Environment
import matplotlib.pyplot as plt

class Agent:
    
    def __init__(self, env):
        self.train_mode = True
        self.states = set()
        self.actions = [i for i in range(9)]
        for _ in range(1000000):
            self.states.add(env.sample_state())
        self.states = list(self.states)
        self.Q = {}
        self.V = {}
        self.policy = {}
        for state in self.states:
        	self.V[state] = 0 
            self.Q[state] = {}
            for action in self.actions:
                self.Q[state][action] = 0
        for state in list(self.Q.keys()):
            self.policy[state] = max(self.Q[state])

    # def take_action(self, state, eps=0.05):
    	# for a in Q[state].keys():
    		# Q[state][a] = 
            
    def train(self, env, epochs=10000, eps=0.1):
        eps = 1
        frame_idx = 0
        episodic_rews = []
        returns = {}
        for state in self.states:
            returns[state] = {}
            for action in self.actions:
                returns[state][action] = []
        for epoch in range(epochs):
            done = False
            episode = []
            seen_states = set()
            s, pr_s = env.reset()
            # env.render(pr_s)
            tot_rew = 0
            frame_idx = 0
            while not done:
                frame_idx += 1
                a = self.take_action(s, eps)
                sp, r, done, pr_s = env.step(a)
                tot_rew += r
                eps = max(0.1, 1 - frame_idx/1000)
                # env.render(pr_s)
                # if frame_idx > 5000:
                #     r = -200
                #     break
                episode.append((s, a, r))
                s = sp
            episodic_rews.append(tot_rew)
            print("Epoch", epoch, "Total reward this episode: ",tot_rew)

        plt.plot([i for i in range(len(episodic_rews))], episodic_rews, c='orange')
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.show()
        print(self.policy)
        input()

    def gen_optimum_policy(self):
        for state in list(self.Q.keys()):
            self.policy[state] = max(self.Q[state])

def main():
    layout = [(2, 15), (5, 17), (2, 10), (7, 14),
          (3, 10), (5, 17), (7, 24), (2, 20),
          (6, 15), (8, 17), (5, 10), (2, 14),
          (1, 15), (5, 17), (2, 10), (7, 14),
          (2, 15), (5, 17), (1, 17), (5, 14),
          (2, 13), (7, 17), (2, 10), (7, 14),
          (1, 15), (5, 17), (8, 25), (10, 27),
          (12, 15), (15, 17), (17, 10), (18, 14)]
    env = Environment(32, layout)
    agent = Agent(env)
    agent.train(env)

if __name__ == "__main__":
    main()