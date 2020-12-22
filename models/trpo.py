import numpy as np
import torch

from models.pg import PolicyNetwork, ValueNetwork

if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor


def get_flat_grads(f, net):
    flat_grads = torch.cat([
        grad.view(-1)
        for grad in torch.autograd.grad(f, net.parameters(), create_graph=True)
    ])

    return flat_grads


def get_flat_params(net):
    return torch.cat([param.view(-1) for param in net.parameters()])


def set_params(net, new_flat_params):
    start_idx = 0
    for param in net.parameters():
        end_idx = start_idx + np.prod(list(param.shape))
        param.data = torch.reshape(
            new_flat_params[start_idx:end_idx], param.shape
        )

        start_idx = end_idx


def conjugate_gradient(Av_func, b, max_iter=10, residual_tol=1e-10):
    x = torch.zeros_like(b)
    r = b - Av_func(x)
    p = r
    rsold = r.norm() ** 2

    for _ in range(max_iter):
        Ap = Av_func(p)
        alpha = rsold / torch.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r.norm() ** 2
        if torch.sqrt(rsnew) < residual_tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x


def rescale_and_linesearch(
    g, s, Hs, kl_stepsize, L, kld, old_params, pi, max_iter=10,
    success_ratio=0.1
):
    set_params(pi, old_params)
    L_old = L().detach()

    beta = torch.sqrt((2 * kl_stepsize) / torch.dot(s, Hs))

    for _ in range(max_iter):
        new_params = old_params + beta * s

        set_params(pi, new_params)
        kld_new = kld().detach()

        L_new = L().detach()

        actual_improv = L_new - L_old
        approx_improv = torch.dot(g, beta * s)
        ratio = actual_improv / approx_improv

        if ratio > success_ratio \
            and actual_improv > 0 \
                and kld_new < kl_stepsize:
            return new_params

        beta *= 0.5

    print("The line search was failed!")
    return old_params


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
        num_steps_per_iter = self.train_config["num_steps_per_iter"]
        horizon = self.train_config["horizon"]
        discount = self.train_config["discount"]
        kl_stepsize = self.train_config["kl_stepsize"]
        cg_damping = self.train_config["cg_damping"]
        normalize_return = self.train_config["normalize_return"]
        use_baseline = self.train_config["use_baseline"]

        if use_baseline:
            opt_v = torch.optim.Adam(self.v.parameters(), lr)

        rwd_iter_means = []
        for i in range(num_iters):
            rwd_iter = []

            obs = []
            acts = []
            rets = []
            disc = []

            steps = 0
            while steps < num_steps_per_iter:
                ep_rwds = []
                ep_disc_rwds = []
                ep_disc = []

                t = 0
                done = False

                ob = env.reset()

                while not done and steps < num_steps_per_iter:
                    act = self.act(ob)

                    obs.append(ob)
                    acts.append(act)

                    if render:
                        env.render()
                    ob, rwd, done, info = env.step(act)

                    ep_rwds.append(rwd)
                    ep_disc_rwds.append(rwd * (discount ** t))
                    ep_disc.append(discount ** t)

                    t += 1
                    steps += 1

                    if horizon is not None:
                        if t >= horizon:
                            break

                ep_disc = FloatTensor(ep_disc)

                ep_disc_rets = FloatTensor(
                    [sum(ep_disc_rwds[i:]) for i in range(len(ep_disc_rwds))]
                )
                ep_rets = ep_disc_rets / ep_disc

                rets.append(ep_rets)
                disc.append(ep_disc)

                if done:
                    rwd_iter.append(np.sum(ep_rwds))

            rwd_iter_means.append(np.mean(rwd_iter))
            print(
                "Iterations: %i,   Reward Mean: %f"
                % (i + 1, np.mean(rwd_iter))
            )

            obs = FloatTensor(obs)
            acts = FloatTensor(np.array(acts))
            rets = torch.cat(rets)
            disc = torch.cat(disc)

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
            old_params = get_flat_params(self.pi).detach()
            old_distb = self.pi(obs)

            def L():
                distb = self.pi(obs)

                if use_baseline:
                    return (disc * delta * torch.exp(
                        distb.log_prob(acts)
                        - old_distb.log_prob(acts).detach()))\
                        .mean()
                else:
                    return (disc * rets * torch.exp(
                        distb.log_prob(acts)
                        - old_distb.log_prob(acts).detach()))\
                        .mean()

            def kld():
                distb = self.pi(obs)

                if self.discrete:
                    old_p = old_distb.probs.detach()
                    p = distb.probs

                    return (old_p * (torch.log(old_p) - torch.log(p)))\
                        .sum(-1)\
                        .mean()
                    # return \
                    #     (disc *
                    #         (old_p * (torch.log(old_p) - torch.log(p)))
                    #         .sum(-1))\
                    #     .mean()

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
                g, s, Hs, kl_stepsize, L, kld, old_params, self.pi
            )

            set_params(self.pi, new_params)

        return rwd_iter_means
