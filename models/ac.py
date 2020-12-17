import numpy as np
import scipy
import torch

from models.pg import PolicyNetwork, ValueNetwork


class ActorCritic:
    def __init__(
        self,
        state_dim,
        action_dim,
        discrete,
        action_std=0.1,
        train_config=None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.action_std = action_std
        self.train_config = train_config

        self.pi = PolicyNetwork(self.state_dim, self.action_dim, self.discrete)
        self.v = ValueNetwork(self.state_dim)
    
    def get_networks(self):
        return [self.pi, self.v]

    def act(self, state, cuda=False):
        self.pi.eval()

        state = torch.FloatTensor(state)
        if cuda:
            state = state.cuda()

        if self.discrete:
            probs = self.pi(state)
            m = torch.distributions.Categorical(probs)
            action = m.sample().detach().numpy()
        else:
            mean = self.pi(state)
            cov_mtx = torch.eye(self.action_dim) * (self.action_std ** 2)
            m = torch.distributions.MultivariateNormal(mean, cov_mtx)
            action = m.sample().detach().numpy()

        return action
    
    def train(self, env, render=False, cuda=False):
        lr = self.train_config["lr"]
        num_iters = self.train_config["num_iters"]
        num_eps_per_iter = self.train_config["num_eps_per_iter"]
        horizon = self.train_config["horizon"]
        discount = self.train_config["discount"]
        normalize_advantage = self.train_config["normalize_advantage"]

        opt_pi = torch.optim.Adam(self.pi.parameters(), lr)
        opt_v = torch.optim.Adam(self.v.parameters(), lr)

        rwd_iter_means = []
        for i in range(num_iters):
            rwd_iter = []
            for _ in range(num_eps_per_iter):
                obs = []
                acts = []
                rwds = []
                disc_rwds = []
                disc = []

                if render:
                    env.render()

                ob = env.reset()
                act = self.act(ob)

                obs.append(ob)
                acts.append(act)

                ob, rwd, done, info = env.step(act)
                
                rwds.append(rwd)
                disc_rwds.append(rwd)
                disc.append(discount ** 0)

                t = 1
                while True:
                    act = self.act(ob)

                    obs.append(ob)
                    acts.append(act)

                    ob, rwd, done, info = env.step(act)

                    rwds.append(rwd)
                    disc_rwds.append(rwd * (discount ** t))
                    disc.append(discount ** t)
                        
                    t += 1

                    if done:
                        break
                    if horizon is not None:
                        if t >= horizon:
                            break

                rwd_iter.append(np.sum(rwds))

                obs = torch.FloatTensor(obs)
                acts = torch.FloatTensor(np.array(acts))
                rwds = torch.FloatTensor(rwds)

                disc = torch.FloatTensor(disc)

                if cuda:
                    obs, acts, rwds, disc = obs.cuda(), acts.cuda(), rwds.cuda(), disc.cuda()
                
                self.v.eval()
                curr_vals = self.v(obs)
                next_vals = torch.cat((self.v(obs)[1:], torch.FloatTensor([[0.]])))
                advantage = (rwds + discount * next_vals - curr_vals).detach()
                if normalize_advantage:
                    advantage = (advantage - advantage.mean()) / advantage.std()

                self.v.train()

                opt_v.zero_grad()
                loss = (-1) * disc * advantage * self.v(obs)
                loss.mean().backward()
                opt_v.step()

                self.pi.train()

                if self.discrete:
                    probs = self.pi(obs)
                    m = torch.distributions.Categorical(probs)
                else:
                    mean = self.pi(obs)
                    cov_mtx = torch.eye(self.action_dim) * (self.action_std ** 2)
                    m = torch.distributions.MultivariateNormal(mean, cov_mtx)
                
                opt_pi.zero_grad()
                loss = (-1) * disc * advantage * m.log_prob(acts)
                loss.mean().backward()
                opt_pi.step()

            rwd_iter_means.append(np.mean(rwd_iter))
            print("Iterations: %i,   Reward Mean: %f" % (i + 1, np.mean(rwd_iter)))
        
        return rwd_iter_means