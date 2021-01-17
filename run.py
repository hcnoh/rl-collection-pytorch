import json
import argparse

import numpy as np
import torch
import gym

from models.pg import PolicyGradient
from models.ac import ActorCritic
from models.trpo import TRPO
from models.gae import GAE
from models.ppo import PPO


def main(env_name, model_name, num_episodes, render):
    ckpt_path = ".ckpts/%s/%s/" % (model_name, env_name)

    with open(ckpt_path + "model_config.json") as f:
        config = json.load(f)

    if env_name not in ["CartPole-v1", "Pendulum-v0", "BipedalWalker-v3"]:
        print("The environment name is wrong!")
        return

    env = gym.make(env_name)
    env.reset()

    state_dim = len(env.observation_space.high)
    if env_name in ["CartPole-v1"]:
        discrete = True
        action_dim = env.action_space.n
    else:
        discrete = False
        action_dim = env.action_space.shape[0]

    if model_name == "pg":
        model = PolicyGradient(state_dim, action_dim, discrete, **config)
    elif model_name == "ac":
        model = ActorCritic(state_dim, action_dim, discrete, **config)
    elif model_name == "trpo":
        model = TRPO(state_dim, action_dim, discrete, **config)
    elif model_name == "gae":
        model = GAE(state_dim, action_dim, discrete, **config)
    elif model_name == "ppo":
        model = PPO(state_dim, action_dim, discrete, **config)

    if hasattr(model, "pi"):
        model.pi.load_state_dict(torch.load(ckpt_path + "policy.ckpt"))

    rwd_mean = []
    for i in range(1, num_episodes + 1):
        rwds = []

        done = False
        ob = env.reset()

        while not done:
            act = model.act(ob)
            if render:
                env.render()
            ob, rwd, done, info = env.step(act)
            rwds.append(rwd)

        rwd_sum = sum(rwds)
        print("The total reward of the episode %i = %f" % (i, rwd_sum))
        rwd_mean.append(rwd_sum)

    env.close()

    rwd_std = np.std(rwd_mean)
    rwd_mean = np.mean(rwd_mean)
    print("Mean = %f" % rwd_mean)
    print("Standard Deviation = %f" % rwd_std)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        default="CartPole-v1",
        help="Type the environment name to run. \
            The possible environments are \
                [CartPole-v1, Pendulum-v0, BipedalWalker-v3]"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="pg",
        help="Type the model name to train. \
            The possible models are [pg, ac, trpo, gae, ppo]"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help="Type the number of episodes to run this agent"
    )
    parser.add_argument(
        "--render",
        type=str,
        default="True",
        help="Type whether the render is on or not"
    )
    args = parser.parse_args()

    if args.render == "True":
        render = True
    else:
        render = False

    main(args.env_name, args.model_name, args.num_episodes, render)
