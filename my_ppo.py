from my_network import FeedForwardNN
import gym
import time

import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal

class PPO:
    # init
    def __init__(self, env) -> None:
        # Initialize hyperparameters
        self._init_hyperparameters()

        # Get environment info
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # define our actor and critic
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)
        
        # define the actor and critic optimizer
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # Create a covariance to sample actions
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)


    def _init_hyperparameters(self):
        # Default values for hyperparameters, will need to change later.
        self.timesteps_per_batch = 4800            # timesteps per batch
        self.max_timesteps_per_episode = 1600      # timesteps per episode

        # Discount factor gamma
        self.gamma = 0.95

        # Epoch
        self.n_updates_per_iteration = 5

        # PPO Clip parameter
        self.clip = 0.2 # As recommended by the paper

        # Learning rate
        self.lr = 0.005

    def learn(self, total_timesteps):
        current_timestep = 0
        i_so_far = 0 # Iterations ran so far

        while current_timestep < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            current_timestep += np.sum(batch_lens)

            V, _ = self.evaluate(batch_obs, batch_acts)

            # Compute advantage estimate
            A_k = batch_rtgs - V.detach() # V here should not enter the gradient
            
            # Normalize advantages to make it more numerically stable
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

			# Increment the number of iterations
            i_so_far += 1

            # PPO_CLIP
            for _ in range(self.n_updates_per_iteration):
                # Calculate pi_theta(a_t | s_t), this is updated with every epoch
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                # Get the ratio by taking exponential
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                # Calculate actor loss
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                # Optimize actor
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                # Final step, update critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()


    def evaluate(self, batch_obs, batch_acts):

        # Compute critic
        V = self.critic(batch_obs).squeeze()

        # Compute actor
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts) # return the log probability of the chosen action

        return V, log_probs
    
    def rollout(self):
        # Batch data
        batch_obs = []             # batch observations
        batch_acts = []            # batch actions
        batch_log_probs = []       # log probs of each action
        batch_rews = []            # batch rewards
        batch_rtgs = []            # batch rewards-to-go
        batch_lens = []            # episodic lengths in batch

        # Number of timesteps run so far this batch
        t = 0

        ep_rews = []
        # sample batch
        while t < self.timesteps_per_batch:
            ep_rews = []

            obs = self.env.reset() # reset environment
            done = False
            # sample episode
            for ep_t in range(self.max_timesteps_per_episode):
                t += 1

                # Collect observation
                batch_obs.append(obs)

                # Collect actions
                action, log_prob = self.get_action(obs)

                # Step env forward
                obs, rew, done, _ = self.env.step(action)

                # Collect other info
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break
            

            batch_lens.append(ep_t + 1) # plus 1 because timestep starts at 0
            batch_rews.append(ep_rews)  # 
        # Gym use numpy, need to convert to tensor for later use
        batch_obs = np.array(batch_obs)
        batch_acts = np.array(batch_acts)
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)

        # Compute rewards to go
        batch_rtgs = self.compute_rtgs(batch_rews)
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def get_action(self, obs):

        # Use actor critic to compute mean
        mean = self.actor(obs)

        # Create normal distribution
        dist = MultivariateNormal(mean, self.cov_mat)

        # Sample from normal distribution
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach() 

    def compute_rtgs(self, batch_rews):
        batch_rtgs = []

        # Iterate through batch 
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        
        # Convert to tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    env = gym.make('Pendulum-v0')
    agent = PPO(env)
    
    agent.learn(10000)
    
        