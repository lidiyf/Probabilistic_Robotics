"""
PPO ALGORITHM
1. Initialise actor and critic networks
2. for k=0,1,2,... do
3. Collect set of trajectories dk=ti by running pi-k=pi-theta-k in the environment
4. compute rewards to go
5. Calculate advantage
6. Update policy by maximizing the PPO-Clip objective
7. Fit value function by regression on mean-squared error typically using a gradient descent algorithm
8. End for loop
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from network import FeedForwardNN
import numpy as np

class PPO:
    def __init(self,env):
        #Extract env info and initialize hyperparameters
        self._init_hyperparameters()

        self.env=env
        self.obs_dim=env.observation_sapce.shape[0]
        self.act_dim=env.action_space.shape[0]

        self.actor=FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic=FeedForwardNN(self.obs_dim,1)

        self.actor_optim=Adam(self.actor.parameters(),lr=self.lr)
        self.cov_var=torch.full(size=(self.act_time,),fill_value=0.5)
        #covariance matrix

        self.cov_mat=torch.diag(self.cov_var)
        self.cov_var-torch.full(size=[self.act_dim],fill_value=0.5)

    def learn (self, total_timesteps) :
        t_so_far = 0 # Timesteps simulated so far
        while t_so_far < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
            t_so_far+=np.sum(batch_lens)

        V = self.evaluate (batch_obs)

        #calculate advantage
        A_k = batch_rtgs - V. detach ()
        # Normalize advantages
        A_k = (A_k - A_k.mean ()) / (A_k.std() + 1e-10)

        for _ in range(self.n_updates_per_iteration):
            _, curr_log_probs =self.evaluation(batch_obs, batch_acts)

            ratios = torch. exp(curr_log_probs - batch_log_probs)

            surr1 = ratios * A_k
            surr2 = torch. clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

            actor_loss = (-torch.min(surr1, surr2)) . mean ()
            critic_loss=nn.MSELoss()(V,batch_rtgs)

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step ()

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

    def evaluate (self, batch_obs):
        # Query critic network for a value V for each obs in batch_obs.
        V = self.critic(batch_obs) . squeeze ()

        mean=self.actor(batch_obs)
        dist = MultivariateNormal(mean,self.cov_mat)
        log_probs=dist.log_prob(batch_acts)

        return V
    
    def rollout(self):
        batch_obs=[] #no. of timesteps per batch -> dimention of observation
        batch_acts=[] 
        batch_log_probs=[]
        batch_rews=[] #rewards
        batch_rtgs=[] #rewards to go
        batch_lens=[]

        t=0
        while t<self.timesteps_per_batch:
            ep_rews=[]
            obs=self.env.reset()
            done=False
            for ep_t in range(self.max_timesteps_per_episode):
                action = self.env.action.sample()
                obs,rew,done,_=self.env.step(action)
                if done:
                    break
                
            batch_lens.append(t+1)
            batch_rews.append(ep_rews)
        
        batch_obs=torch.tensor(batch_obs,dtype=torch.float)
        batch_acts=torch.tensot(batch_acts, dtype=torch.float)
        batch_log_probs=torch.tensor(batch_log_probs,dtype=torch.float)

        batch_rtgs=self.compute_rtgs(batch_rews)

        return batch_obs, batch_acts, batch_log_probs, batch_lens

    def get_action(self,obs):
        mean=self.actor(obs)
        dist=MultivariateNormal(mean,self.cov_mat)
        action=dist.sample()
        log_prob=dist.log_prob(action)

        return action.detatch().numpy(), log_prob.detatch()
    
    def compute_rtgs(self,batch_rews):
        # The shape will be (num timesteps per episode)
        batch_rtgs = []
        # Iterate through each episode
        for ep_rews in reversed (batch_rews) :
            discounted_reward = 0 # The discounted reward so far
            for rew in reversed (ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs. insert(0, discounted_reward)
# Convert the rewards-to-go into a tensor
        batch_rtgs = torch. tensor (batch_rtgs, dtype-torch. float)
        return batch_rtgs
    
    def _init_hyperparameters(self):
        self.timesteps_per_batch=4800
        self.max_timesteps_per_episode=1600
        self.gamma=0.95 #discount factor
        self.n_updates_per_iteration=5 #number of epochs per episode
        self.clip=0.2 #clip threshold
        self.lr=0.005 #learning rate of optimizers
    
    
   
    
