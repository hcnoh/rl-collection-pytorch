import numpy as np
import scipy
import torch

from scipy.sparse.linalg import cg

from models.pg import PolicyNetwork, ValueNetwork

if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor


class TRPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        discrete,
        train_config=None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.train_config = train_config

        self.pi = PolicyNetwork(self.state_dim, self.action_dim, self.discrete)
        if self.train_config["use_baseline"]:
            self.v = ValueNetwork(self.state_dim)
        
        if torch.cuda.is_available():
            for net in self.get_networks():
                net.to(torch.device("cuda"))
    
    def get_networks(self):
        if self.train_config["use_baseline"]:
            return [self.pi, self.v]
        else:
            return [self.pi]

    def act(self, state):
        self.pi.eval()

        state = FloatTensor(state)
        distb = self.pi(state)
        
        action = distb.sample().detach().cpu().numpy()

        return action
    
    def train(self, env, render=False):
        lr = self.train_config["lr"]
        num_iters = self.train_config["num_iters"]
        num_eps_per_iter = self.train_config["num_eps_per_iter"]
        horizon = self.train_config["horizon"]
        discount = self.train_config["discount"]
        kl_stepsize = self.train_config["kl_stepsize"]
        normalize_return = self.train_config["normalize_return"]
        use_baseline = self.train_config["use_baseline"]

        opt_pi = torch.optim.Adam(self.pi.parameters(), lr)
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

                ob = env.reset()
                try:
                    act = self.act(ob)
                except:
                    for param in self.pi.parameters():
                        print(param)

                obs.append(ob)
                acts.append(act)

                if render:
                    env.render()
                ob, rwd, done, info = env.step(act)
                
                rwds.append(rwd)
                disc_rwds.append(rwd)
                disc.append(discount ** 0)

                t = 1
                while True:
                    act = self.act(ob)

                    obs.append(ob)
                    acts.append(act)

                    if render:
                        env.render()
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

                obs = FloatTensor(obs)
                acts = FloatTensor(np.array(acts))

                disc = FloatTensor(disc)

                disc_rets = FloatTensor(
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
                distb = self.pi(obs)
                
                opt_pi.zero_grad()
                if use_baseline:
                    loss = (-1) * disc * delta * distb.log_prob(acts)
                else:
                    loss = (-1) * disc * distb.log_prob(acts) * rets
                grads = torch.autograd.grad(loss.mean(), self.pi.parameters(), retain_graph=True)

                log_pi = distb.log_prob(acts)
                for param, g in zip(self.pi.parameters(), grads):
                    g = g.flatten().detach()

                    lp, d = log_pi[0], disc[0]
                    grad_lp = torch.autograd.grad(lp, param, retain_graph=True)[0].flatten().detach()
                    grad_lp = grad_lp.unsqueeze(-1)
                    H = d * torch.matmul(grad_lp, grad_lp.T)
                    for lp, d in zip(log_pi[1:], disc[1:]):
                        grad_lp = torch.autograd.grad(lp, param, retain_graph=True)[0].flatten().detach()
                        grad_lp = grad_lp.unsqueeze(-1)
                        H += d * torch.matmul(grad_lp, grad_lp.T)
                    H /= len(disc)

                    s, _ = cg(H.cpu().numpy(), g.cpu().numpy(), tol=1e-10, maxiter=10)
                    s = FloatTensor(s).unsqueeze(-1)
                    beta = torch.sqrt((2 * kl_stepsize) / (torch.matmul(s.T, torch.matmul(H, s)) + 1e-10))

                    s = beta * s
                    # print(s, beta)

                    param.grad = s.unflatten(0, param.shape).squeeze()
                
                opt_pi.step()
                print("eeeeeeeeeeeeeeeeeeeeeeeeeeeee")

            rwd_iter_means.append(np.mean(rwd_iter))
            print("Iterations: %i,   Reward Mean: %f" % (i + 1, np.mean(rwd_iter)))
        
        return rwd_iter_means