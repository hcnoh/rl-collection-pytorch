import numpy as np
import torch

from torch.nn import Module

from models.nets import PolicyNetwork, ValueNetwork
from utils.funcs import get_flat_grads, get_flat_params, set_params, \
    conjugate_gradient, rescale_and_linesearch

if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor


class GAE(Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        discrete,
        train_config=None
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.train_config = train_config

        self.pi = PolicyNetwork(self.state_dim, self.action_dim, self.discrete)
        self.v = ValueNetwork(self.state_dim)

    def get_networks(self):
        return [self.pi, self.v]

    def act(self, state):
        self.pi.eval()

        state = FloatTensor(state)
        distb = self.pi(state)

        action = distb.sample().detach().cpu().numpy()

        return action

    def train(self, env, render=False):
        num_iters = self.train_config["num_iters"]
        num_steps_per_iter = self.train_config["num_steps_per_iter"]
        horizon = self.train_config["horizon"]
        gamma_ = self.train_config["gamma"]
        lambda_ = self.train_config["lambda"]
        eps = self.train_config["epsilon"]
        max_kl = self.train_config["max_kl"]
        cg_damping = self.train_config["cg_damping"]
        normalize_advantage = self.train_config["normalize_advantage"]

        rwd_iter_means = []
        for i in range(num_iters):
            rwd_iter = []

            obs = []
            acts = []
            rets = []
            advs = []
            gms = []

            steps = 0
            while steps < num_steps_per_iter:
                ep_obs = []
                ep_rwds = []
                ep_disc_rwds = []
                ep_gms = []
                ep_lmbs = []

                t = 0
                done = False

                ob = env.reset()

                while not done and steps < num_steps_per_iter:
                    act = self.act(ob)

                    ep_obs.append(ob)
                    obs.append(ob)
                    acts.append(act)

                    if render:
                        env.render()
                    ob, rwd, done, info = env.step(act)

                    ep_rwds.append(rwd)
                    ep_disc_rwds.append(rwd * (gamma_ ** t))
                    ep_gms.append(gamma_ ** t)
                    ep_lmbs.append(lambda_ ** t)

                    t += 1
                    steps += 1

                    if horizon is not None:
                        if t >= horizon:
                            done = True
                            break

                if done:
                    rwd_iter.append(np.sum(ep_rwds))

                ep_obs = FloatTensor(np.array(ep_obs))
                ep_rwds = FloatTensor(ep_rwds)
                ep_disc_rwds = FloatTensor(ep_disc_rwds)
                ep_gms = FloatTensor(ep_gms)
                ep_lmbs = FloatTensor(ep_lmbs)

                ep_disc_rets = FloatTensor(
                    [sum(ep_disc_rwds[i:]) for i in range(t)]
                )
                ep_rets = ep_disc_rets / ep_gms

                rets.append(ep_rets)

                self.v.eval()
                curr_vals = self.v(ep_obs).detach()
                next_vals = torch.cat(
                    (self.v(ep_obs)[1:], FloatTensor([[0.]]))
                ).detach()
                ep_deltas = ep_rwds.unsqueeze(-1)\
                    + gamma_ * next_vals\
                    - curr_vals

                ep_advs = FloatTensor([
                    ((ep_gms * ep_lmbs)[:t - j].unsqueeze(-1) * ep_deltas[j:])
                    .sum()
                    for j in range(t)
                ])
                advs.append(ep_advs)

                gms.append(ep_gms)

            rwd_iter_means.append(np.mean(rwd_iter))
            print(
                "Iterations: {},   Reward Mean: {}"
                .format(i + 1, np.mean(rwd_iter))
            )

            obs = FloatTensor(np.array(obs))
            acts = FloatTensor(np.array(acts))
            rets = torch.cat(rets)
            advs = torch.cat(advs)
            gms = torch.cat(gms)

            if normalize_advantage:
                advs = (advs - advs.mean()) / advs.std()

            self.v.train()
            old_params = get_flat_params(self.v).detach()
            old_v = self.v(obs).detach()

            def constraint():
                return ((old_v - self.v(obs)) ** 2).mean()

            grad_diff = get_flat_grads(constraint(), self.v)

            def Hv(v):
                hessian = get_flat_grads(torch.dot(grad_diff, v), self.v)\
                    .detach()

                return hessian

            g = get_flat_grads(
                ((-1) * (self.v(obs).squeeze() - rets) ** 2).mean(), self.v
            ).detach()
            s = conjugate_gradient(Hv, g).detach()

            Hs = Hv(s).detach()
            alpha = torch.sqrt(2 * eps / torch.dot(s, Hs))

            new_params = old_params + alpha * s

            set_params(self.v, new_params)

            self.pi.train()
            old_params = get_flat_params(self.pi).detach()
            old_distb = self.pi(obs)

            def L():
                distb = self.pi(obs)

                return (advs * torch.exp(
                            distb.log_prob(acts)
                            - old_distb.log_prob(acts).detach()
                        )).mean()

            def kld():
                distb = self.pi(obs)

                if self.discrete:
                    old_p = old_distb.probs.detach()
                    p = distb.probs

                    return (old_p * (torch.log(old_p) - torch.log(p)))\
                        .sum(-1)\
                        .mean()

                else:
                    old_mean = old_distb.mean.detach()
                    old_cov = old_distb.covariance_matrix.sum(-1).detach()
                    mean = distb.mean
                    cov = distb.covariance_matrix.sum(-1)

                    return (0.5) * (
                            (old_cov / cov).sum(-1)
                            + (((old_mean - mean) ** 2) / cov).sum(-1)
                            - self.action_dim
                            + torch.log(cov).sum(-1)
                            - torch.log(old_cov).sum(-1)
                        ).mean()

            grad_kld_old_param = get_flat_grads(kld(), self.pi)

            def Hv(v):
                hessian = get_flat_grads(
                    torch.dot(grad_kld_old_param, v),
                    self.pi
                ).detach()

                return hessian + cg_damping * v

            g = get_flat_grads(L(), self.pi).detach()

            s = conjugate_gradient(Hv, g).detach()
            Hs = Hv(s).detach()

            new_params = rescale_and_linesearch(
                g, s, Hs, max_kl, L, kld, old_params, self.pi
            )

            set_params(self.pi, new_params)

        return rwd_iter_means
