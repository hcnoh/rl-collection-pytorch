import numpy as np
import scipy
import torch


class PolicyNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, discrete):
        super(PolicyNetwork, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50, 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50, 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50, action_dim),
            # torch.nn.Tanh()
        )
        self.discrete = discrete
    
    def forward(self, states):
        if self.discrete:
            return torch.nn.functional.softmax(self.net(states))
        else:
            return self.net(states)# * self.action_range


class ValueNetwork(torch.nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50, 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50, 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50, 1),
        )
        
    def forward(self, states):
        return self.net(states)


class PolicyGradient:
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
        if self.train_config["use_baseline"]:
            self.v = ValueNetwork(self.state_dim)
    
    def get_networks(self):
        if self.train_config["use_baseline"]:
            return [self.pi, self.v]
        else:
            return [self.pi]

    def act(self, state):
        self.pi.eval()

        state = torch.FloatTensor(state)

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
    
    def train(self, env, render=False):
        lr = self.train_config["lr"]
        num_iters = self.train_config["num_iters"]
        num_eps_per_iter = self.train_config["num_eps_per_iter"]
        horizon = self.train_config["horizon"]
        discount = self.train_config["discount"]
        normalize_return = self.train_config["normalize_return"]
        use_baseline = self.train_config["use_baseline"]

        opt = torch.optim.Adam(self.pi.parameters(), lr)
        if use_baseline:
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

                disc = torch.FloatTensor(disc)

                disc_rets = torch.FloatTensor(
                    [sum(disc_rwds[i:]) for i in range(len(disc_rwds))]
                )
                rets = disc_rets / disc
                
                if normalize_return:
                    rets = (rets - rets.mean()) / rets.std()
                
                if use_baseline:
                    self.v.eval()
                    delta = (rets - self.v(obs)).detach()

                    self.v.train()

                    opt_v.zero_grad()
                    loss = (-1) * disc * delta * self.v(obs)
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
                
                opt.zero_grad()
                if use_baseline:
                    loss = (-1) * disc * delta * m.log_prob(acts)
                else:
                    loss = (-1) * disc * m.log_prob(acts) * rets
                loss.mean().backward()
                opt.step()

            rwd_iter_means.append(np.mean(rwd_iter))
            print("Iterations: %i,   Reward Mean: %f" % (i + 1, np.mean(rwd_iter)))
        
        return rwd_iter_means