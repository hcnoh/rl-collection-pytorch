import numpy as np
import torch

from models.nets import PolicyNetwork, ValueNetwork

if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor


class PPO:
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
        self.v = ValueNetwork(self.state_dim)

        if torch.cuda.is_available():
            for net in self.get_networks():
                net.to(torch.device("cuda"))

    def get_networks(self):
        return [self.pi, self.v]

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
        num_epochs = self.train_config["num_epochs"]
        minibatch_size = self.train_config["minibatch_size"]
        horizon = self.train_config["horizon"]
        gamma_ = self.train_config["gamma"]
        lambda_ = self.train_config["lambda"]
        eps = self.train_config["epsilon"]
        c1 = self.train_config["vf_coeff"]
        c2 = self.train_config["entropy_coeff"]
        normalize_advantage = self.train_config["normalize_advantage"]

        opt_pi = torch.optim.Adam(self.pi.parameters(), lr)
        opt_v = torch.optim.Adam(self.v.parameters(), lr)

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
                            break

                if done:
                    rwd_iter.append(np.sum(ep_rwds))

                ep_obs = FloatTensor(ep_obs)
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

                ep_advs = torch.FloatTensor([
                    ((ep_gms * ep_lmbs)[:t - j].unsqueeze(-1) * ep_deltas[j:])
                    .sum()
                    for j in range(t)
                ])
                advs.append(ep_advs)

                gms.append(ep_gms)

            rwd_iter_means.append(np.mean(rwd_iter))
            print(
                "Iterations: %i,   Reward Mean: %f"
                % (i + 1, np.mean(rwd_iter))
            )

            obs = FloatTensor(obs)
            acts = FloatTensor(np.array(acts))
            rets = torch.cat(rets)
            advs = torch.cat(advs)
            gms = torch.cat(gms)

            if normalize_advantage:
                advs = (advs - advs.mean()) / advs.std()

            self.pi.eval()
            old_log_pi = self.pi(obs).log_prob(acts).detach()

            self.pi.train()
            self.v.train()

            max_steps = num_epochs * (num_steps_per_iter // minibatch_size)

            for _ in range(max_steps):
                minibatch_indices = np.random.choice(
                    range(steps), minibatch_size, False
                )
                mb_obs = obs[minibatch_indices]
                mb_acts = acts[minibatch_indices]
                mb_advs = advs[minibatch_indices]
                mb_rets = rets[minibatch_indices]

                mb_log_pi = self.pi(mb_obs).log_prob(mb_acts)
                mb_old_log_pi = old_log_pi[minibatch_indices]

                r = torch.exp(mb_log_pi - mb_old_log_pi)

                L_clip = torch.minimum(
                    r * mb_advs, torch.clip(r, 1 - eps, 1 + eps) * mb_advs
                )

                L_vf = (self.v(mb_obs).squeeze() - mb_rets) ** 2

                S = (-1) * mb_log_pi

                opt_pi.zero_grad()
                opt_v.zero_grad()
                loss = (-1) * (L_clip - c1 * L_vf + c2 * S).mean()
                loss.backward()
                opt_pi.step()
                opt_v.step()

        return rwd_iter_means
